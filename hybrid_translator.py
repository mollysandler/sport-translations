# hybrid_translator.py
"""
Hybrid approach: Analyze first 30-60 seconds, then stream the rest
Perfect for live sports where commentators are usually consistent
"""

import asyncio
import torch
import numpy as np
from faster_whisper import WhisperModel
from TTS.api import TTS
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio

class HybridTranslator:
    def __init__(self, source_lang="en", target_lang="hi"):
        self.whisper = WhisperModel("medium.en", device="cpu", compute_type="int8")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Voice profiles discovered during pre-analysis
        self.speaker_voices = {}
        self.is_initialized = False
    
    async def initialize_from_stream(self, audio_stream, duration_seconds=30):
        """
        Analyze first 30-60 seconds to discover speakers and build voice profiles.
        This happens ONCE at the start of the broadcast.
        """
        print(f"🔍 Pre-analyzing first {duration_seconds}s of broadcast...")
        
        # Collect initial audio
        initial_audio = await self._collect_audio(audio_stream, duration_seconds)
        
        # Quick diarization on this segment only
        from pyannote.audio import Pipeline
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HUGGING_FACE_TOKEN")
        )
        
        # Run diarization
        diarization = diarization_pipeline({
            "waveform": initial_audio,
            "sample_rate": 16000
        })
        
        # Extract voice samples for each speaker
        for speaker_id in diarization.labels():
            # Get all segments for this speaker
            speaker_segments = [
                initial_audio[:, int(turn.start*16000):int(turn.end*16000)]
                for turn, _, label in diarization.itertracks(yield_label=True)
                if label == speaker_id
            ]
            
            if speaker_segments:
                # Concatenate up to 10 seconds
                speaker_audio = torch.cat(speaker_segments[:5], dim=1)
                target_length = min(speaker_audio.shape[1], 16000 * 10)
                speaker_audio = speaker_audio[:, :target_length]
                
                # Store reference audio
                self.speaker_voices[speaker_id] = speaker_audio.squeeze().numpy()
                print(f"✅ Voice profile ready for {speaker_id} ({target_length/16000:.1f}s)")
        
        self.is_initialized = True
        print(f"🎤 Discovered {len(self.speaker_voices)} speakers")
        print("🚀 Switching to real-time mode...\n")
    
    def process_chunk_streaming(self, audio_chunk):
        """
        Process audio chunks in real-time using pre-discovered voice profiles.
        Much faster because we already know the voices!
        """
        if not self.is_initialized:
            raise RuntimeError("Call initialize_from_stream() first!")
        
        # 1. Quick speaker ID (no voice collection needed!)
        speaker_id = self._identify_speaker_fast(audio_chunk)
        
        # 2. Transcribe
        segments, _ = self.whisper.transcribe(
            audio_chunk.numpy(),
            language=self.source_lang,
            beam_size=1,
            vad_filter=True
        )
        text = " ".join([seg.text for seg in segments])
        
        if not text.strip():
            return None
        
        # 3. Translate
        translated = self._translate(text)
        
        # 4. Synthesize with KNOWN voice
        reference_audio = self.speaker_voices[speaker_id]
        audio_output = self.tts.tts(
            text=translated,
            speaker_wav=reference_audio,
            language=self.target_lang
        )
        
        return {
            'speaker': speaker_id,
            'original': text,
            'translated': translated,
            'audio': audio_output
        }
    
    def _identify_speaker_fast(self, audio_chunk):
        """Identify speaker using pre-built embeddings"""
        # Extract embedding from chunk
        chunk_embedding = self.embedding_model.encode_batch(
            audio_chunk.unsqueeze(0)
        )
        
        # Compare to known speakers (we have them from pre-analysis!)
        best_match = None
        best_similarity = -1
        
        for speaker_id, reference_audio in self.speaker_voices.items():
            # Get embedding of reference
            ref_tensor = torch.from_numpy(reference_audio).unsqueeze(0)
            ref_embedding = self.embedding_model.encode_batch(ref_tensor)
            
            # Cosine similarity
            similarity = torch.cosine_similarity(
                chunk_embedding, ref_embedding, dim=1
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id
        
        return best_match
    
    def _translate(self, text):
        """Use local model for speed"""
        from transformers import pipeline
        translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-hi",
            device=0 if torch.cuda.is_available() else -1
        )
        return translator(text)[0]['translation_text']
    
    async def _collect_audio(self, stream, duration):
        """Helper to collect audio for pre-analysis"""
        chunks = []
        target_samples = int(duration * 16000)
        collected_samples = 0
        
        async for chunk in stream:
            chunks.append(chunk)
            collected_samples += chunk.shape[0]
            if collected_samples >= target_samples:
                break
        
        return torch.cat(chunks)[:target_samples]


# Usage Example
async def translate_live_broadcast():
    """
    Example: Translate a live sports broadcast
    """
    import sounddevice as sd
    
    translator = HybridTranslator(source_lang="en", target_lang="hi")
    
    # Step 1: Pre-analysis phase (30 seconds)
    print("=" * 60)
    print("PHASE 1: SPEAKER DISCOVERY (first 30 seconds)")
    print("=" * 60)
    
    # Simulate audio stream
    async def audio_stream():
        """Generator that yields audio chunks from microphone"""
        def callback(indata, frames, time, status):
            return torch.from_numpy(indata.copy()).squeeze()
        
        # In real implementation, this would be your actual audio source
        sample_rate = 16000
        duration = 0.5  # 500ms chunks
        
        with sd.InputStream(samplerate=sample_rate, channels=1) as stream:
            while True:
                audio, _ = stream.read(int(duration * sample_rate))
                yield torch.from_numpy(audio.squeeze())
                await asyncio.sleep(duration)
    
    # Initialize with first 30 seconds
    await translator.initialize_from_stream(audio_stream(), duration_seconds=30)
    
    # Step 2: Real-time streaming
    print("=" * 60)
    print("PHASE 2: REAL-TIME TRANSLATION")
    print("=" * 60)
    
    async for chunk in audio_stream():
        result = translator.process_chunk_streaming(chunk)
        
        if result:
            print(f"[{result['speaker']}]: {result['original']}")
            print(f"  → {result['translated']}\n")
            # Play result['audio']


# For pre-recorded videos (like your current use case)
def translate_video_hybrid(video_path: str, source_lang="en", target_lang="hi"):
    """
    Translate a video file using hybrid approach:
    1. Analyze first 30s to get speaker voices
    2. Process rest in pseudo-real-time chunks
    """
    translator = HybridTranslator(source_lang, target_lang)
    
    # Load full audio
    audio, sr = torchaudio.load(video_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    # Phase 1: Analyze first 30 seconds
    print("🔍 Analyzing speakers in first 30 seconds...")
    initial_audio = audio[:, :16000*30]  # First 30s
    
    # Use your existing diarizer on this segment
    from diarizer import SpeakerDiarizer
    diarizer = SpeakerDiarizer(os.getenv("HUGGING_FACE_TOKEN"))
    
    # Save temp file for diarizer
    temp_path = "temp_initial.wav"
    torchaudio.save(temp_path, initial_audio, 16000)
    segments = diarizer.diarize(temp_path)
    
    # Extract voice samples for each speaker
    for speaker_id in set(seg.speaker_id for seg in segments):
        speaker_segments = [
            audio[:, int(seg.start_sec*16000):int(seg.end_sec*16000)]
            for seg in segments if seg.speaker_id == speaker_id
        ]
        
        if speaker_segments:
            combined = torch.cat(speaker_segments[:5], dim=1)
            translator.speaker_voices[speaker_id] = combined.squeeze().numpy()
    
    translator.is_initialized = True
    print(f"✅ {len(translator.speaker_voices)} speakers identified\n")
    
    # Phase 2: Process remaining audio in chunks
    print("🚀 Processing video in real-time mode...")
    chunk_size = 16000 * 2  # 2-second chunks
    
    for i in range(30*16000, audio.shape[1], chunk_size):
        chunk = audio[:, i:i+chunk_size]
        result = translator.process_chunk_streaming(chunk)
        
        if result:
            print(f"[{result['speaker']}]: {result['translated']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        translate_video_hybrid(sys.argv[1])
    else:
        asyncio.run(translate_live_broadcast())