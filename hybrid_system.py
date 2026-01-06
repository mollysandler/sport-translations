# hybrid_system.py
"""
Complete hybrid translation system for sports commentary
Phase 1: Analyze first 30-60s to discover speakers
Phase 2: Stream remaining audio with known voice profiles
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pydub import AudioSegment
from pydub.playback import play
from faster_whisper import WhisperModel
from TTS.api import TTS
from scipy.io.wavfile import write as write_wav

# Import your existing diarizer
from diarizer import SpeakerDiarizer, SpeakerSegment


@dataclass
class TranslationSegment:
    speaker_id: str
    start_ms: int
    end_ms: int
    original_text: str
    translated_text: str
    audio_bytes: bytes
    duration_ms: int


class HybridSportsTranslator:
    """
    Hybrid translator optimized for sports commentary:
    - Analyzes first N seconds to identify speakers and collect voice samples
    - Processes remaining audio in streaming chunks with known voices
    """
    
    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        analysis_duration_sec: int = 30,
        chunk_duration_sec: float = 2.0
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.analysis_duration_sec = analysis_duration_sec
        self.chunk_duration_sec = chunk_duration_sec
        self.sample_rate = 16000
        
        # Speaker voice profiles (populated during analysis phase)
        self.speaker_voice_profiles: Dict[str, np.ndarray] = {}
        self.speaker_embeddings: Dict[str, torch.Tensor] = {}
        
        print("🔧 Initializing translation models...")
        self._initialize_models()
        print("✅ Models ready\n")
    
    def _initialize_models(self):
        """Load all required models"""
        # 1. Faster Whisper for transcription
        print("   📝 Loading Faster-Whisper...")
        model_name = "medium.en" if self.source_lang == "en" else "medium"
        self.whisper = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8"  # Quantization for speed
        )
        
        # 2. Translation
        print("   🌍 Loading translation model...")
        from transformers import MarianMTModel, MarianTokenizer
        
        # Map language codes to Helsinki models
        model_map = {
            ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
            ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
            ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
            ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
        }
        
        model_name = model_map.get((self.source_lang, self.target_lang))
        if model_name:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)
        else:
            # Fallback to Google Cloud
            from google.cloud import translate_v2 as translate
            self.translate_client = translate.Client()
            self.translation_model = None
            print("   ⚠️  Using Google Cloud Translate (slower)")
        
        # 3. TTS - Coqui for voice cloning
        print("   🎤 Loading Coqui TTS (this may take a minute)...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # 4. Speaker embedding model for fast identification
        print("   🔍 Loading speaker recognition...")
        from speechbrain.inference.speaker import EncoderClassifier
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec"
        )
    
    def translate_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> str:
        """
        Main entry point: translate entire video
        Returns path to output audio file
        """
        print("="*70)
        print("🎬 HYBRID TRANSLATION PIPELINE")
        print("="*70)
        
        # 1. Load audio from video
        audio_path = self._extract_audio(video_path)
        audio, sr = torchaudio.load(audio_path)
        
        # Ensure mono and correct sample rate
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        total_duration_sec = audio.shape[1] / self.sample_rate
        print(f"📊 Total duration: {total_duration_sec:.1f}s\n")
        
        # Adjust analysis duration for short videos
        actual_analysis_duration = min(
            self.analysis_duration_sec,
            max(5, total_duration_sec * 0.3)  # Use 30% of video or min 5s
        )
        
        if total_duration_sec < 10:
            print(f"⚠️  Short video detected ({total_duration_sec:.1f}s)")
            print(f"   Using {actual_analysis_duration:.1f}s for analysis\n")
        
        # 2. PHASE 1: Speaker Discovery
        print("="*70)
        print("🔍 PHASE 1: SPEAKER DISCOVERY & VOICE PROFILING")
        print("="*70)
        
        analysis_samples = int(actual_analysis_duration * self.sample_rate)
        analysis_audio = audio[:, :analysis_samples]
        
        self._analyze_speakers(analysis_audio, audio_path, hf_token)
        
        # 3. PHASE 2: Streaming Translation
        print("\n" + "="*70)
        print("🚀 PHASE 2: STREAMING TRANSLATION")
        print("="*70)
        
        segments = self._process_streaming(audio, start_sample=analysis_samples)
        
        # 4. PHASE 3: Audio Composition
        print("\n" + "="*70)
        print("🎵 PHASE 3: AUDIO COMPOSITION")
        print("="*70)
        
        final_audio = self._compose_audio(segments, total_duration_sec)
        
        # 5. Save and return
        if output_path is None:
            output_path = self._generate_output_path(video_path)
        
        final_audio.export(output_path, format="wav")
        print(f"\n✅ Translation complete!")
        print(f"💾 Saved to: {output_path}")
        
        return output_path
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        print(f"🎬 Extracting audio from '{Path(video_path).name}'...")
        
        from moviepy.editor import VideoFileClip
        
        temp_audio = "temp_extracted_audio.wav"
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            temp_audio,
            codec='pcm_s16le',
            fps=self.sample_rate,
            nbytes=2,
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        video.close()
        
        return temp_audio
    
    def _analyze_speakers(
        self,
        audio: torch.Tensor,
        audio_path: str,
        hf_token: Optional[str]
    ):
        """
        Phase 1: Analyze initial segment to discover speakers
        and collect voice samples
        """
        analysis_duration = audio.shape[1] / self.sample_rate
        print(f"⏱️  Analyzing first {analysis_duration:.1f}s...")
        
        # Save analysis segment to temp file for diarizer
        temp_analysis_path = "temp_analysis_segment.wav"
        torchaudio.save(temp_analysis_path, audio, self.sample_rate)
        
        # Run diarization on this segment
        if hf_token is None:
            hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        diarizer = SpeakerDiarizer(hf_token)
        segments = diarizer.diarize(temp_analysis_path)
        
        # Clean up temp file
        os.remove(temp_analysis_path)
        
        # Extract voice samples for each speaker
        print(f"\n🎤 Extracting voice profiles...")
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        
        for speaker_id in unique_speakers:
            # Get all audio segments for this speaker
            speaker_segments = [
                seg for seg in segments 
                if seg.speaker_id == speaker_id
            ]
            
            # Collect up to 10 seconds of clean audio
            collected_audio = []
            total_duration = 0
            target_duration = 10.0  # seconds
            
            for seg in speaker_segments:
                if total_duration >= target_duration:
                    break
                
                # Extract audio segment
                start_sample = int(seg.start_sec * self.sample_rate)
                end_sample = int(seg.end_sec * self.sample_rate)
                segment_audio = audio[:, start_sample:end_sample]
                
                # Only use segments longer than 1 second for quality
                seg_duration = (end_sample - start_sample) / self.sample_rate
                if seg_duration >= 1.0:
                    collected_audio.append(segment_audio)
                    total_duration += seg_duration
            
            if collected_audio:
                # Concatenate all segments
                voice_profile = torch.cat(collected_audio, dim=1)
                
                # Trim to target duration
                max_samples = int(target_duration * self.sample_rate)
                if voice_profile.shape[1] > max_samples:
                    voice_profile = voice_profile[:, :max_samples]
                
                # Store as numpy array
                self.speaker_voice_profiles[speaker_id] = voice_profile.squeeze().numpy()
                
                # Also compute and store embedding for fast lookup
                embedding = self.speaker_encoder.encode_batch(voice_profile)
                self.speaker_embeddings[speaker_id] = embedding
                
                profile_duration = voice_profile.shape[1] / self.sample_rate
                print(f"   ✅ {speaker_id}: {profile_duration:.1f}s voice profile captured")
        
        print(f"\n🎯 Discovered {len(self.speaker_voice_profiles)} speakers")
        print(f"   Speakers: {', '.join(sorted(self.speaker_voice_profiles.keys()))}")
    
    def _process_streaming(
        self,
        audio: torch.Tensor,
        start_sample: int
    ) -> List[TranslationSegment]:
        """
        Phase 2: Process audio in streaming chunks using known voice profiles
        """
        segments = []
        chunk_samples = int(self.chunk_duration_sec * self.sample_rate)
        total_samples = audio.shape[1]
        
        print(f"📦 Processing in {self.chunk_duration_sec}s chunks...")
        print(f"   Starting from {start_sample/self.sample_rate:.1f}s\n")
        
        current_sample = start_sample
        chunk_num = 0
        
        while current_sample < total_samples:
            chunk_num += 1
            
            # Extract chunk
            end_sample = min(current_sample + chunk_samples, total_samples)
            chunk = audio[:, current_sample:end_sample]
            
            # Process this chunk
            chunk_start_sec = current_sample / self.sample_rate
            chunk_end_sec = end_sample / self.sample_rate
            
            result = self._process_chunk(
                chunk,
                chunk_start_sec,
                chunk_end_sec,
                chunk_num
            )
            
            if result:
                segments.append(result)
            
            current_sample = end_sample
        
        print(f"\n✅ Processed {len(segments)} segments")
        return segments
    
    def _process_chunk(
        self,
        chunk: torch.Tensor,
        start_sec: float,
        end_sec: float,
        chunk_num: int
    ) -> Optional[TranslationSegment]:
        """Process a single audio chunk"""
        
        # 1. Check if there's speech (VAD)
        if self._is_silence(chunk):
            return None
        
        # 2. Identify speaker
        speaker_id = self._identify_speaker(chunk)
        
        # 3. Transcribe
        chunk_np = chunk.squeeze().numpy()
        segments, info = self.whisper.transcribe(
            chunk_np,
            language=self.source_lang,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Combine all text from this chunk
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        if not text_parts:
            return None
        
        original_text = " ".join(text_parts)
        
        # 4. Translate
        translated_text = self._translate_text(original_text)
        
        # 5. Synthesize with speaker's voice
        audio_bytes = self._synthesize_speech(translated_text, speaker_id)
        
        if not audio_bytes:
            return None
        
        # Calculate duration
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        duration_ms = len(audio_segment)
        
        # Progress indicator
        print(f"[Chunk {chunk_num}] {speaker_id}: \"{translated_text[:60]}...\"")
        
        return TranslationSegment(
            speaker_id=speaker_id,
            start_ms=int(start_sec * 1000),
            end_ms=int(end_sec * 1000),
            original_text=original_text,
            translated_text=translated_text,
            audio_bytes=audio_bytes,
            duration_ms=duration_ms
        )
    
    def _is_silence(self, chunk: torch.Tensor, threshold: float = 0.01) -> bool:
        """Check if audio chunk is mostly silence"""
        energy = torch.abs(chunk).mean().item()
        return energy < threshold
    
    def _identify_speaker(self, chunk: torch.Tensor) -> str:
        """Identify speaker using pre-computed embeddings"""
        # Ensure chunk has the right shape for speechbrain
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)  # Add batch dimension
        
        # Extract embedding from chunk
        chunk_embedding = self.speaker_encoder.encode_batch(chunk)
        
        # Compare to known speakers
        best_speaker = None
        best_similarity = -1
        
        for speaker_id, ref_embedding in self.speaker_embeddings.items():
            # Compute cosine similarity
            similarity = torch.cosine_similarity(
                chunk_embedding,
                ref_embedding,
                dim=-1  # Compute along feature dimension
            )
            
            # Get scalar value
            if similarity.numel() > 1:
                similarity = similarity.mean()  # Average if multiple values
            similarity_value = similarity.item()
            
            if similarity_value > best_similarity:
                best_similarity = similarity_value
                best_speaker = speaker_id
        
        # If no good match, default to first speaker
        if best_similarity < 0.5:
            best_speaker = list(self.speaker_embeddings.keys())[0]
        
        return best_speaker
    
    def _translate_text(self, text: str) -> str:
        """Translate text using loaded model"""
        if self.source_lang == self.target_lang:
            return text
        
        if self.translation_model:
            # Use local model
            tokens = self.tokenizer([text], return_tensors="pt", padding=True)
            translated = self.translation_model.generate(**tokens)
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            # Use Google Cloud
            result = self.translate_client.translate(
                text,
                source_language=self.source_lang,
                target_language=self.target_lang
            )
            return result['translatedText']
    
    def _synthesize_speech(self, text: str, speaker_id: str) -> Optional[bytes]:
        """Generate speech using speaker's voice profile"""
        reference_audio = self.speaker_voice_profiles.get(speaker_id)
        
        if reference_audio is None or len(reference_audio) == 0:
            print(f"   ⚠️  No voice profile for {speaker_id}, skipping")
            return None
        
        try:
            # Save reference audio to temp WAV for Coqui
            temp_ref_path = f"temp_ref_{speaker_id}.wav"
            write_wav(temp_ref_path, self.sample_rate, 
                     (reference_audio * 32767).astype(np.int16))
            
            # Generate audio
            audio_array = self.tts.tts(
                text=text,
                speaker_wav=temp_ref_path,
                language=self.target_lang
            )
            
            # Clean up temp file
            os.remove(temp_ref_path)
            
            # Convert to WAV bytes
            audio_int16 = (np.array(audio_array) * 32767).astype(np.int16)
            byte_io = io.BytesIO()
            write_wav(byte_io, 24000, audio_int16)  # Coqui uses 24kHz
            byte_io.seek(0)
            
            return byte_io.read()
            
        except Exception as e:
            print(f"   ❌ TTS error for {speaker_id}: {e}")
            return None
    
    def _compose_audio(
        self,
        segments: List[TranslationSegment],
        total_duration_sec: float
    ) -> AudioSegment:
        """
        Phase 3: Compose final audio with relative timing
        """
        print(f"🎼 Composing {len(segments)} segments...")
        
        # Create silent canvas
        canvas_ms = int(total_duration_sec * 1000) + 5000  # Extra padding
        final_audio = AudioSegment.silent(duration=canvas_ms)
        
        # Track timing
        prev_original_end_ms = 0
        prev_translated_end_ms = 0
        
        for i, seg in enumerate(segments):
            # Load segment audio
            segment_audio = AudioSegment.from_wav(io.BytesIO(seg.audio_bytes))
            
            # Calculate timing with relative sync
            original_gap_ms = seg.start_ms - prev_original_end_ms
            ideal_start_ms = prev_translated_end_ms + original_gap_ms
            actual_start_ms = max(seg.start_ms, ideal_start_ms)
            
            # Extend canvas if needed
            required_length = actual_start_ms + seg.duration_ms
            if len(final_audio) < required_length:
                padding = required_length - len(final_audio) + 2000
                final_audio += AudioSegment.silent(duration=padding)
            
            # Overlay
            final_audio = final_audio.overlay(segment_audio, position=actual_start_ms)
            
            # Update trackers
            prev_original_end_ms = seg.end_ms
            prev_translated_end_ms = actual_start_ms + seg.duration_ms
            
            if i % 10 == 0:
                print(f"   Progress: {i+1}/{len(segments)} segments composed")
        
        print("   ✅ Composition complete")
        return final_audio
    
    def _generate_output_path(self, video_path: str) -> str:
        """Generate output filename"""
        video_dir = Path(video_path).parent
        video_stem = Path(video_path).stem
        output_name = f"translated_{video_stem}_{self.target_lang}.wav"
        return str(video_dir / output_name)


def main():
    """Command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python hybrid_system.py <video_path> [source_lang] [target_lang]")
        print("\nExample: python hybrid_system.py video.mp4 en es")
        sys.exit(1)
    
    video_path = sys.argv[1]
    source_lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    target_lang = sys.argv[3] if len(sys.argv) > 3 else "es"
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    # Create translator
    translator = HybridSportsTranslator(
        source_lang=source_lang,
        target_lang=target_lang,
        analysis_duration_sec=45,  # Analyze first 45 seconds
        chunk_duration_sec=2.0      # Process in 2-second chunks
    )
    
    # Translate
    output_path = translator.translate_video(video_path)
    
    # Play result
    print("\n🔊 Playing translation...")
    audio = AudioSegment.from_wav(output_path)
    play(audio)


if __name__ == "__main__":
    main()