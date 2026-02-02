# optimized_translator.py
import asyncio
import queue
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
import torch
import torchaudio
from TTS.api import TTS  # Coqui TTS - much faster than Bark

class RealtimeTranslator:
    def __init__(self, source_lang="en", target_lang="hi"):
        # Use Faster-Whisper (3-4x speedup)
        self.whisper = WhisperModel(
            "medium.en" if source_lang == "en" else "medium",
            device="cpu",  # or "cuda" for GPU
            compute_type="int8"  # Quantization for speed
        )
        
        # Lightweight TTS (Coqui is 10x faster than Bark)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Speaker embeddings for real-time ID
        self.speaker_tracker = OnlineSpeakerTracker()
        
        # Translation queue for batching
        self.translation_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def process_audio_chunk(self, audio_chunk: torch.Tensor, sample_rate: int):
        """Process a 2-3 second audio chunk in real-time"""
        
        # 1. VAD - Skip silence
        if self._is_silence(audio_chunk):
            return None
        
        # 2. Speaker identification (fast) + reference audio collection
        speaker_id = self.speaker_tracker.identify(audio_chunk)
        
        # 3. Transcription (streaming)
        segments, info = self.whisper.transcribe(
            audio_chunk.numpy(),
            language=self.source_lang,
            beam_size=1,  # Faster than 5
            vad_filter=True,  # Skip non-speech
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        text = " ".join([seg.text for seg in segments])
        if not text.strip():
            return None
        
        # 4. Translation (async batching)
        translated = self._translate_async(text)
        
        # 5. TTS with voice cloning
        reference_audio = self.speaker_tracker.get_reference_audio(speaker_id)
        
        if reference_audio is not None:
            # We have a voice profile - use it for cloning
            audio_output = self.tts.tts(
                text=translated,
                speaker_wav=reference_audio,  # Clone this speaker's voice
                language=self.target_lang
            )
        elif self.speaker_tracker.has_voice_profile(speaker_id):
            # Profile exists, use it
            audio_output = self.tts.tts(
                text=translated,
                speaker_wav=self.speaker_tracker.get_reference_audio(speaker_id),
                language=self.target_lang
            )
        else:
            # Still collecting voice profile - use preset voice temporarily
            preset_voice = self._get_preset_voice(speaker_id)
            audio_output = self.tts.tts(
                text=translated,
                speaker=preset_voice,  # Use preset until we have real voice
                language=self.target_lang
            )
            print(f"⏳ {speaker_id}: Collecting voice profile... ({translated[:30]}...)")
        
        return {
            'speaker': speaker_id,
            'original': text,
            'translated': translated,
            'audio': audio_output
        }
    
    def _get_preset_voice(self, speaker_id):
        """Temporary preset voice while collecting reference audio"""
        # Map speakers to different preset voices
        presets = ["male_1", "female_1", "male_2", "female_2"]
        speaker_num = int(speaker_id.split("_")[-1])
        return presets[speaker_num % len(presets)]
    
    def _is_silence(self, audio_chunk):
        """Quick VAD check"""
        energy = torch.abs(audio_chunk).mean()
        return energy < 0.01
    
    def _translate_async(self, text):
        """Batch translations for efficiency"""
        # Use local translation model for lower latency
        # Or batch multiple requests to Google Cloud
        from transformers import pipeline
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
        result = translator(text)[0]['translation_text']
        return result


class OnlineSpeakerTracker:
    """Real-time speaker identification with voice reference collection"""
    def __init__(self, sample_rate=16000):
        from speechbrain.pretrained import EncoderClassifier
        self.embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        self.known_speakers = []
        self.speaker_embeddings = []
        self.speaker_reference_audio = {}  # Store reference audio per speaker
        self.speaker_audio_buffers = {}  # Accumulate audio until we have enough
        self.similarity_threshold = 0.75
        self.reference_duration_target = 8.0  # seconds
        self.sample_rate = sample_rate
    
    def identify(self, audio_chunk):
        """Identify speaker from audio chunk"""
        # Extract embedding
        embedding = self.embedding_model.encode_batch(audio_chunk.unsqueeze(0))
        
        # Compare to known speakers
        if not self.speaker_embeddings:
            speaker_id = "SPEAKER_0"
            self.known_speakers.append(speaker_id)
            self.speaker_embeddings.append(embedding)
            self.speaker_audio_buffers[speaker_id] = []
            return speaker_id
        
        # Find closest match
        similarities = [
            torch.cosine_similarity(embedding, known_emb, dim=1).item()
            for known_emb in self.speaker_embeddings
        ]
        
        max_sim = max(similarities)
        
        if max_sim > self.similarity_threshold:
            # Existing speaker
            speaker_id = self.known_speakers[similarities.index(max_sim)]
        else:
            # New speaker detected
            speaker_id = f"SPEAKER_{len(self.known_speakers)}"
            self.known_speakers.append(speaker_id)
            self.speaker_embeddings.append(embedding)
            self.speaker_audio_buffers[speaker_id] = []
        
        # Accumulate reference audio for this speaker
        self._accumulate_reference_audio(speaker_id, audio_chunk)
        
        return speaker_id
    
    def _accumulate_reference_audio(self, speaker_id, audio_chunk):
        """Build up a reference audio sample for voice cloning"""
        # Don't accumulate if we already have enough
        if speaker_id in self.speaker_reference_audio:
            return
        
        buffer = self.speaker_audio_buffers[speaker_id]
        buffer.append(audio_chunk.cpu().numpy())
        
        # Check if we have enough audio
        total_duration = sum(len(chunk) for chunk in buffer) / self.sample_rate
        
        if total_duration >= self.reference_duration_target:
            # Concatenate all chunks
            reference_audio = np.concatenate(buffer)
            
            # Take the best 6-8 seconds (middle portion, avoid start/end)
            target_samples = int(self.reference_duration_target * self.sample_rate)
            if len(reference_audio) > target_samples:
                # Take from middle
                start_idx = (len(reference_audio) - target_samples) // 2
                reference_audio = reference_audio[start_idx:start_idx + target_samples]
            
            # Save as WAV in memory for TTS
            self.speaker_reference_audio[speaker_id] = reference_audio
            
            # Clear buffer to save memory
            self.speaker_audio_buffers[speaker_id] = []
            
            print(f"✅ Voice profile captured for {speaker_id} ({total_duration:.1f}s)")
    
    def get_reference_audio(self, speaker_id):
        """Return reference audio for TTS voice cloning"""
        return self.speaker_reference_audio.get(speaker_id, None)
    
    def has_voice_profile(self, speaker_id):
        """Check if we've collected enough audio for this speaker"""
        return speaker_id in self.speaker_reference_audio


# Usage example
async def stream_translation():
    translator = RealtimeTranslator(source_lang="en", target_lang="hi")
    
    # Simulate real-time audio streaming
    import sounddevice as sd
    
    chunk_duration = 2.0  # seconds
    sample_rate = 16000
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        
        # Convert to torch tensor
        audio_chunk = torch.from_numpy(indata.copy()).squeeze()
        
        # Process chunk
        result = translator.process_audio_chunk(audio_chunk, sample_rate)
        
        if result:
            print(f"[{result['speaker']}]: {result['original']}")
            print(f"  -> {result['translated']}")
            # Play result['audio'] with minimal delay
    
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=audio_callback,
        blocksize=int(chunk_duration * sample_rate)
    ):
        print("🎙️ Streaming translation active...")
        await asyncio.sleep(3600)  # Run for 1 hour

if __name__ == "__main__":
    asyncio.run(stream_translation())