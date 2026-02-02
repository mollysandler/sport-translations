# elevenlabs_streaming.py
"""
Fast streaming translator using:
- Faster-Whisper (local, but faster than standard Whisper)
- Google Translate API (no auth needed, or local model)
- ElevenLabs TTS (premium quality, fast)

NO Google Cloud credentials needed!
"""

import os
import sys
import threading
import queue
import time
import io
import torch
from pathlib import Path
from typing import Optional, List
from pydub import AudioSegment
from pydub.playback import play
import torchaudio
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv  
load_dotenv()

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


class FastStreamingTranslator:
    """
    Fast streaming translator with ElevenLabs TTS
    """
    
    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        buffer_duration_sec: int = 300
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.buffer_duration_sec = buffer_duration_sec
        self.sample_rate = 16000
        
        # Initialize
        self._initialize_services()
        
        # Threading
        self.playback_queue = queue.Queue()
        self.all_segments = []
        self.segments_lock = threading.Lock()
        self.processing_complete = threading.Event()
        self.error_occurred = threading.Event()
        self.error_message = None
        
        # Voice mapping
        self.speaker_voice_ids = {}
    
    def _initialize_services(self):
        """Initialize services"""
        print("🔧 Initializing services...")
        
        # 1. Faster Whisper (faster than standard Whisper)
        print("   📝 Loading Faster-Whisper...")
        from faster_whisper import WhisperModel
        
        model_name = "medium.en" if self.source_lang == "en" else "medium"
        self.whisper = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8"
        )
        
        # 2. Translation
        print("   🌍 Loading translation...")
        try:
            # Try using free googletrans library (no API key needed!)
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
            self.use_local_translation = False
            print("   ✅ Using Google Translate (free API)")
        except ImportError:
            # Fallback to local model
            from transformers import MarianMTModel, MarianTokenizer
            model_map = {
                ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
                ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
                ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
            }
            model_name = model_map.get((self.source_lang, self.target_lang))
            if model_name:
                self.tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.translation_model = MarianMTModel.from_pretrained(model_name)
                self.use_local_translation = True
                print("   ✅ Using local translation model")
        
        # 3. ElevenLabs TTS
        print("   🎤 Loading ElevenLabs TTS...")
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found! Please set it.")
        
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        
        self.elevenlabs = ElevenLabs(api_key=api_key)
        print("   ✅ ElevenLabs ready")
        
        # 4. Speaker recognition
        print("   🔍 Loading speaker recognition...")
        from speechbrain.inference.speaker import EncoderClassifier
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec"
        )
        
        print("✅ All services ready\n")
    
    def translate_video_streaming(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> str:
        """Main streaming translation pipeline"""
        print("="*70)
        print("⚡ FAST STREAMING TRANSLATION")
        print("="*70)
        
        # 1. Extract audio
        audio_path = self._extract_audio(video_path)
        audio, sr = torchaudio.load(audio_path)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        total_duration_sec = audio.shape[1] / self.sample_rate
        print(f"📊 Total duration: {total_duration_sec:.1f}s")
        print(f"⏱️  Buffer: {self.buffer_duration_sec}s\n")
        
        # 2. Get speaker segments
        print("="*70)
        print("🔍 PHASE 1: SPEAKER DISCOVERY")
        print("="*70)
        
        speaker_segments = self._get_speaker_segments(audio, audio_path, hf_token)
        self._assign_voices(speaker_segments)
        
        # Calculate buffer
        buffer_duration_so_far = 0
        buffer_segment_count = 0
        for seg in speaker_segments:
            seg_duration = seg.end_sec - seg.start_sec
            if buffer_duration_so_far < self.buffer_duration_sec:
                buffer_segment_count += 1
                buffer_duration_so_far += seg_duration
            else:
                break
        
        print(f"\n📊 Found {len(speaker_segments)} segments")
        print(f"⏱️  Will buffer {buffer_segment_count} segments (~{buffer_duration_so_far:.1f}s)\n")
        
        # 3. Process with streaming
        print("="*70)
        print("🚀 PHASE 2: STREAMING TRANSLATION")
        print("="*70)
        
        processing_thread = threading.Thread(
            target=self._process_segments,
            args=(audio, speaker_segments),
            daemon=False
        )
        processing_thread.start()
        
        playback_thread = threading.Thread(
            target=self._playback_thread,
            args=(buffer_segment_count,),
            daemon=False
        )
        playback_thread.start()
        
        processing_thread.join()
        playback_thread.join()
        
        # 4. Check errors
        if self.error_occurred.is_set():
            print(f"\n❌ Error: {self.error_message}")
            sys.exit(1)
        
        # 5. Save output
        print("\n" + "="*70)
        print("💾 SAVING FINAL OUTPUT")
        print("="*70)
        
        if output_path is None:
            video_stem = Path(video_path).stem
            output_path = f"translated_{video_stem}_{self.target_lang}.wav"
        
        with self.segments_lock:
            if self.all_segments:
                print(f"📝 Composing {len(self.all_segments)} segments...")
                final_audio = self._compose_audio(self.all_segments, total_duration_sec)
                final_audio.export(output_path, format="wav")
                print(f"✅ Saved to: {output_path}")
        
        return output_path
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video"""
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
    
    def _get_speaker_segments(
        self,
        audio: torch.Tensor,
        audio_path: str,
        hf_token: Optional[str]
    ) -> List[SpeakerSegment]:
        """Get speaker segments via diarization"""
        total_duration = audio.shape[1] / self.sample_rate
        
        print(f"🎙️  Running diarization on full {total_duration:.1f}s...")
        
        if hf_token is None:
            hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        temp_path = "temp_full_audio.wav"
        torchaudio.save(temp_path, audio, self.sample_rate)
        
        diarizer = SpeakerDiarizer(hf_token)
        segments = diarizer.diarize(temp_path)
        
        os.remove(temp_path)
        
        # Show breakdown
        speaker_counts = {}
        for seg in segments:
            speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
        
        print(f"\n📊 Segment breakdown:")
        for speaker_id in sorted(speaker_counts.keys()):
            print(f"   {speaker_id}: {speaker_counts[speaker_id]} segments")
        
        return segments
    
    def _assign_voices(self, segments: List[SpeakerSegment]):
        """Assign ElevenLabs voices to speakers"""
        print(f"\n🎤 Assigning voices...")
        
        # Pre-made ElevenLabs voices (free tier compatible)
        available_voices = {
            "SPEAKER_00": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "SPEAKER_01": "AZnzlk1XvdvUeBnXmlld",  # Domi
            "SPEAKER_02": "EXAVITQu4vr4xnSDxMaL",  # Bella
            "SPEAKER_03": "ErXwobaYiN019PkySvjV",  # Antoni
            "SPEAKER_04": "MF3mGyEYCl7XYWbV9V6O",  # Elli
        }
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        
        for speaker_id in unique_speakers:
            voice_id = available_voices.get(speaker_id, "21m00Tcm4TlvDq8ikWAM")
            self.speaker_voice_ids[speaker_id] = voice_id
            print(f"   ✅ {speaker_id} → Voice {voice_id[:8]}...")
    
    def _process_segments(self, audio: torch.Tensor, segments: List[SpeakerSegment]):
        """Process segments"""
        try:
            start_time = time.time()
            
            for idx, seg in enumerate(segments, 1):
                # Extract audio
                start_sample = int(seg.start_sec * self.sample_rate)
                end_sample = int(seg.end_sec * self.sample_rate)
                segment_audio = audio[:, start_sample:end_sample]
                
                # 1. Transcribe
                transcribed_text = self._transcribe(segment_audio)
                
                if not transcribed_text:
                    print(f"[Seg {idx}/{len(segments)}] Skipped (no speech)")
                    continue
                
                # 2. Translate
                translated_text = self._translate(transcribed_text)
                
                # 3. Synthesize
                audio_bytes = self._synthesize(translated_text, seg.speaker_id)
                
                if not audio_bytes:
                    continue
                
                # Create segment
                audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                duration_ms = len(audio_segment)
                
                result = TranslationSegment(
                    speaker_id=seg.speaker_id,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    original_text=transcribed_text,
                    translated_text=translated_text,
                    audio_bytes=audio_bytes,
                    duration_ms=duration_ms
                )
                
                # Add to queues
                self.playback_queue.put(result)
                with self.segments_lock:
                    self.all_segments.append(result)
                
                # Progress
                elapsed = time.time() - start_time
                progress = (idx / len(segments)) * 100
                
                orig_preview = transcribed_text[:40] + "..." if len(transcribed_text) > 40 else transcribed_text
                trans_preview = translated_text[:40] + "..." if len(translated_text) > 40 else translated_text
                
                print(f"⚡ [{idx}/{len(segments)}] {seg.speaker_id}: \"{orig_preview}\" → \"{trans_preview}\"")
                print(f"   Progress: {progress:.0f}% | Queue: {self.playback_queue.qsize()} | {elapsed:.1f}s")
            
            self.processing_complete.set()
            print(f"\n✅ Processing complete! Total: {time.time() - start_time:.1f}s")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _transcribe(self, audio: torch.Tensor) -> str:
        """Transcribe using Faster-Whisper"""
        audio_np = audio.squeeze().numpy()
        
        segments, _ = self.whisper.transcribe(
            audio_np,
            language=self.source_lang,
            beam_size=1,
            vad_filter=True
        )
        
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(text_parts)
    
    def _translate(self, text: str) -> str:
        """Translate text"""
        if self.source_lang == self.target_lang:
            return text
        
        if not self.use_local_translation:
            # Use free Google Translate API
            return self.translator.translate(text)
        else:
            # Use local model
            tokens = self.tokenizer([text], return_tensors="pt", padding=True)
            translated = self.translation_model.generate(**tokens)
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def _synthesize(self, text: str, speaker_id: str) -> Optional[bytes]:
        """Synthesize with ElevenLabs"""
        try:
            voice_id = self.speaker_voice_ids.get(speaker_id)
            if not voice_id:
                print(f"   ⚠️  No voice for {speaker_id}")
                return None
            
            # Generate audio with ElevenLabs (using correct API)
            from elevenlabs import VoiceSettings
            
            audio_generator = self.elevenlabs.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            # Collect bytes
            audio_bytes = b"".join(audio_generator)
            
            # Convert MP3 to WAV
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            return wav_io.read()
            
        except Exception as e:
            print(f"   ⚠️  TTS error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _playback_thread(self, buffer_segment_count: int):
        """Playback thread"""
        try:
            print(f"⏳ Waiting for buffer ({buffer_segment_count} segments)...")
            
            while self.playback_queue.qsize() < buffer_segment_count:
                time.sleep(0.5)
                
                if self.error_occurred.is_set():
                    return
                
                if self.processing_complete.is_set() and self.playback_queue.qsize() > 0:
                    break
            
            print(f"\n📊 BUFFER READY! Starting playback...\n")
            print("="*70)
            
            segment_count = 0
            prev_end_ms = 0
            
            while not self.processing_complete.is_set() or not self.playback_queue.empty():
                try:
                    segment = self.playback_queue.get(timeout=2)
                    segment_count += 1
                    
                    audio_segment = AudioSegment.from_wav(io.BytesIO(segment.audio_bytes))
                    
                    # Add gap
                    gap_ms = segment.start_ms - prev_end_ms
                    if gap_ms > 100:
                        play(AudioSegment.silent(duration=int(gap_ms)))
                    
                    # Play
                    print(f"📊 [{segment.speaker_id}] {segment.translated_text[:60]}...")
                    play(audio_segment)
                    
                    prev_end_ms = segment.start_ms + segment.duration_ms
                    self.playback_queue.task_done()
                    
                except queue.Empty:
                    if not self.processing_complete.is_set():
                        time.sleep(0.5)
                    else:
                        break
            
            print(f"\n✅ Playback complete! Played {segment_count} segments")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Playback error: {e}")
    
    def _compose_audio(self, segments, total_duration_sec):
        """Compose final audio"""
        canvas_ms = int(total_duration_sec * 1000) + 5000
        final_audio = AudioSegment.silent(duration=canvas_ms)
        
        prev_original_end_ms = 0
        prev_translated_end_ms = 0
        
        for seg in segments:
            segment_audio = AudioSegment.from_wav(io.BytesIO(seg.audio_bytes))
            
            original_gap_ms = seg.start_ms - prev_original_end_ms
            
            if original_gap_ms < 200:
                actual_start_ms = prev_translated_end_ms + 100
            else:
                ideal_start_ms = prev_translated_end_ms + original_gap_ms
                actual_start_ms = max(seg.start_ms, ideal_start_ms)
            
            required_length = actual_start_ms + seg.duration_ms
            if len(final_audio) < required_length:
                final_audio += AudioSegment.silent(duration=required_length - len(final_audio) + 2000)
            
            final_audio = final_audio.overlay(segment_audio, position=actual_start_ms)
            
            prev_original_end_ms = seg.end_ms
            prev_translated_end_ms = actual_start_ms + seg.duration_ms
        
        return final_audio


def main():
    """CLI"""
    if len(sys.argv) < 2:
        print("Usage: python elevenlabs_streaming.py <video_path> [source] [target] [buffer_sec]")
        print("\nExample: python elevenlabs_streaming.py video.mp4 en es 10")
        print("\nRequired environment variables:")
        print("  - ELEVENLABS_API_KEY")
        print("  - HUGGING_FACE_TOKEN")
        print("\nOptional: pip install deep-translator (for free Google Translate)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    source_lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    target_lang = sys.argv[3] if len(sys.argv) > 3 else "es"
    buffer_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    translator = FastStreamingTranslator(
        source_lang=source_lang,
        target_lang=target_lang,
        buffer_duration_sec=buffer_sec
    )
    
    output_path = translator.translate_video_streaming(video_path)
    print(f"\n🎉 Complete! Output: {output_path}")


if __name__ == "__main__":
    main()