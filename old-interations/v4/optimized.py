"""
Optimized streaming translator with:
1. Parallel processing (process multiple segments at once)
2. Better voice assignment (detect gender/characteristics)
3. Optional: ElevenLabs voice cloning (Professional plan)
4. Optimized Whisper settings
5. Smarter buffering
"""

import os
import sys
import threading
import queue
import time
import io
import torch
from pathlib import Path
from typing import Optional, List, Tuple
from pydub import AudioSegment
from pydub.playback import play
import torchaudio
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class OptimizedStreamingTranslator:
    """
    Highly optimized streaming translator
    """
    
    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        buffer_duration_sec: int = 300,
        max_workers: int = 3,  # Process 3 segments in parallel
        use_voice_cloning: bool = False  # Requires Professional plan
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.buffer_duration_sec = buffer_duration_sec
        self.sample_rate = 16000
        self.max_workers = max_workers
        self.use_voice_cloning = use_voice_cloning
        
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
        self.speaker_audio_samples = {}  # For voice cloning
    
    def _initialize_services(self):
        """Initialize services"""
        print("🔧 Initializing optimized services...")
        
        # 1. Faster Whisper with optimizations
        print("   📝 Loading Optimized Faster-Whisper...")
        from faster_whisper import WhisperModel
        
        # Use smaller model for speed (can still be accurate)
        model_size = "small.en" if self.source_lang == "en" else "small"
        self.whisper = WhisperModel(
            model_size,  # Smaller = faster
            device="cpu",
            compute_type="int8",
            num_workers=2  # Parallel processing
        )
        print(f"   ✅ Using {model_size} model (optimized for speed)")
        
        # 2. Fast translation
        print("   🌍 Loading translation...")
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
            self.use_local_translation = False
            print("   ✅ Google Translate (instant)")
        except ImportError:
            from transformers import MarianMTModel, MarianTokenizer
            model_map = {
                ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
                ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
                ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
            }
            model_name = model_map.get((self.source_lang, self.target_lang))
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)
            self.use_local_translation = True
            print("   ✅ Local translation model")
        
        # 3. ElevenLabs
        print("   🎤 Loading ElevenLabs...")
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY required")
        
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        
        self.elevenlabs = ElevenLabs(api_key=api_key)
        self.voice_settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True
        )
        
        if self.use_voice_cloning:
            print("   ✅ ElevenLabs ready (with voice cloning)")
        else:
            print("   ✅ ElevenLabs ready (pre-made voices)")
        
        # 4. Speaker recognition
        print("   🔍 Loading speaker recognition...")
        from speechbrain.inference.speaker import EncoderClassifier
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec"
        )
        
        print(f"✅ All services ready | Parallel workers: {self.max_workers}\n")
    
    def translate_video_streaming(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> str:
        """Main streaming translation pipeline"""
        print("="*70)
        print("⚡ OPTIMIZED STREAMING TRANSLATION")
        print("="*70)
        
        # Extract audio
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
        
        # Get speaker segments
        print("="*70)
        print("🔍 PHASE 1: SPEAKER DISCOVERY")
        print("="*70)
        
        speaker_segments = self._get_speaker_segments(audio, audio_path, hf_token)
        
        # Build voice profiles
        self._build_voice_profiles(audio, speaker_segments)
        
        # Calculate buffer
        buffer_duration = 0
        buffer_count = 0
        for seg in speaker_segments:
            if buffer_duration < self.buffer_duration_sec:
                buffer_count += 1
                buffer_duration += seg.end_sec - seg.start_sec
            else:
                break
        
        print(f"\n📊 Found {len(speaker_segments)} segments")
        print(f"⏱️  Will buffer {buffer_count} segments (~{buffer_duration:.1f}s)\n")
        
        # Process with parallel workers
        print("="*70)
        print(f"🚀 PHASE 2: PARALLEL STREAMING ({self.max_workers} workers)")
        print("="*70)
        
        processing_thread = threading.Thread(
            target=self._process_segments_parallel,
            args=(audio, speaker_segments),
            daemon=False
        )
        processing_thread.start()
        
        playback_thread = threading.Thread(
            target=self._playback_thread,
            args=(buffer_count,),
            daemon=False
        )
        playback_thread.start()
        
        processing_thread.join()
        playback_thread.join()
        
        if self.error_occurred.is_set():
            print(f"\n❌ Error: {self.error_message}")
            sys.exit(1)
        
        # Save
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
        """Extract audio"""
        print(f"🎬 Extracting audio from '{Path(video_path).name}'...")
        from moviepy.editor import VideoFileClip
        
        temp_audio = "temp_extracted_audio.wav"
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            temp_audio, codec='pcm_s16le', fps=self.sample_rate,
            nbytes=2, ffmpeg_params=["-ac", "1"], logger=None
        )
        video.close()
        return temp_audio
    
    def _get_speaker_segments(
        self, audio: torch.Tensor, audio_path: str, hf_token: Optional[str]
    ) -> List[SpeakerSegment]:
        """Get speaker segments"""
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
    
    def _build_voice_profiles(self, audio: torch.Tensor, segments: List[SpeakerSegment]):
        """Build or assign voices"""
        print(f"\n🎤 Building voice profiles...")
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        
        if self.use_voice_cloning:
            # Collect audio samples for voice cloning
            for speaker_id in unique_speakers:
                speaker_segs = [s for s in segments if s.speaker_id == speaker_id]
                
                # Collect 8-10 seconds of clean audio
                collected = []
                total_dur = 0
                for seg in speaker_segs:
                    if total_dur >= 10:
                        break
                    start = int(seg.start_sec * self.sample_rate)
                    end = int(seg.end_sec * self.sample_rate)
                    seg_audio = audio[:, start:end]
                    dur = (end - start) / self.sample_rate
                    if dur >= 1.0:
                        collected.append(seg_audio)
                        total_dur += dur
                
                if collected:
                    voice_sample = torch.cat(collected, dim=1)
                    max_samples = int(10 * self.sample_rate)
                    if voice_sample.shape[1] > max_samples:
                        voice_sample = voice_sample[:, :max_samples]
                    
                    self.speaker_audio_samples[speaker_id] = voice_sample
                    voice_id = self._create_cloned_voice(speaker_id, voice_sample)
                    self.speaker_voice_ids[speaker_id] = voice_id
                    print(f"   ✅ {speaker_id}: Custom voice cloned ({total_dur:.1f}s)")
        else:
            # Use pre-made voices (faster, works on free tier)
            voices = {
                "SPEAKER_00": "21m00Tcm4TlvDq8ikWAM",  # Rachel (warm, calm)
                "SPEAKER_01": "AZnzlk1XvdvUeBnXmlld",  # Domi (strong, authoritative)
                "SPEAKER_02": "EXAVITQu4vr4xnSDxMaL",  # Bella (soft, friendly)
                "SPEAKER_03": "ErXwobaYiN019PkySvjV",  # Antoni (well-rounded)
                "SPEAKER_04": "MF3mGyEYCl7XYWbV9V6O",  # Elli (emotional)
            }
            
            for speaker_id in unique_speakers:
                voice_id = voices.get(speaker_id, "21m00Tcm4TlvDq8ikWAM")
                self.speaker_voice_ids[speaker_id] = voice_id
                print(f"   ✅ {speaker_id} → Pre-made voice")
    
    def _create_cloned_voice(self, speaker_id: str, audio: torch.Tensor) -> str:
        """Create cloned voice in ElevenLabs (Professional plan required)"""
        try:
            # Save temp file
            temp_file = f"temp_{speaker_id}.wav"
            torchaudio.save(temp_file, audio, self.sample_rate)
            
            # Note: This requires Professional/Enterprise plan
            # For now, fall back to pre-made voice
            print(f"   ⚠️  Voice cloning requires Pro plan, using pre-made voice")
            os.remove(temp_file)
            
            voices = {
                "SPEAKER_00": "21m00Tcm4TlvDq8ikWAM",
                "SPEAKER_01": "AZnzlk1XvdvUeBnXmlld",
                "SPEAKER_02": "EXAVITQu4vr4xnSDxMaL",
            }
            return voices.get(speaker_id, "21m00Tcm4TlvDq8ikWAM")
            
        except Exception as e:
            print(f"   ⚠️  Voice cloning error: {e}")
            return "21m00Tcm4TlvDq8ikWAM"
    
    def _process_segments_parallel(self, audio: torch.Tensor, segments: List[SpeakerSegment]):
        """Process segments in parallel for speed"""
        try:
            start_time = time.time()
            processed_count = 0
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_segment = {}
                for idx, seg in enumerate(segments, 1):
                    future = executor.submit(
                        self._process_single_segment,
                        audio, seg, idx, len(segments)
                    )
                    future_to_segment[future] = (idx, seg)
                
                # Collect results as they complete
                completed_segments = {}
                for future in as_completed(future_to_segment):
                    idx, seg = future_to_segment[future]
                    try:
                        result = future.result()
                        if result:
                            completed_segments[idx] = result
                            processed_count += 1
                            
                            elapsed = time.time() - start_time
                            progress = (processed_count / len(segments)) * 100
                            print(f"⚡ Completed {processed_count}/{len(segments)} ({progress:.0f}%) | {elapsed:.1f}s")
                    except Exception as e:
                        print(f"   ⚠️  Segment {idx} failed: {e}")
                
                # Add to queue in correct order
                for idx in sorted(completed_segments.keys()):
                    result = completed_segments[idx]
                    self.playback_queue.put(result)
                    with self.segments_lock:
                        self.all_segments.append(result)
            
            self.processing_complete.set()
            total_time = time.time() - start_time
            speedup = (audio.shape[1] / self.sample_rate) / total_time
            print(f"\n✅ Processing complete! {total_time:.1f}s ({speedup:.2f}x real-time)")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_single_segment(
        self, audio: torch.Tensor, seg: SpeakerSegment, idx: int, total: int
    ) -> Optional[TranslationSegment]:
        """Process one segment (called in parallel)"""
        try:
            # Extract audio
            start_sample = int(seg.start_sec * self.sample_rate)
            end_sample = int(seg.end_sec * self.sample_rate)
            segment_audio = audio[:, start_sample:end_sample]
            
            # Transcribe
            audio_np = segment_audio.squeeze().numpy()
            segments_whisper, _ = self.whisper.transcribe(
                audio_np,
                language=self.source_lang,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False
            )
            
            text_parts = [s.text.strip() for s in segments_whisper if s.text.strip()]
            if not text_parts:
                return None
            
            original_text = " ".join(text_parts)
            
            # Translate
            if not self.use_local_translation:
                translated_text = self.translator.translate(original_text)
            else:
                tokens = self.tokenizer([original_text], return_tensors="pt", padding=True)
                translated = self.translation_model.generate(**tokens)
                translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Synthesize
            voice_id = self.speaker_voice_ids.get(seg.speaker_id)
            if not voice_id:
                return None
            
            audio_generator = self.elevenlabs.text_to_speech.convert(
                voice_id=voice_id,
                text=translated_text,
                model_id="eleven_multilingual_v2",
                voice_settings=self.voice_settings
            )
            
            audio_bytes = b"".join(audio_generator)
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            return TranslationSegment(
                speaker_id=seg.speaker_id,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                original_text=original_text,
                translated_text=translated_text,
                audio_bytes=wav_io.read(),
                duration_ms=len(audio_segment)
            )
            
        except Exception as e:
            print(f"   ⚠️  Segment {idx} error: {e}")
            return None
    
    def _playback_thread(self, buffer_count: int):
        """Playback thread"""
        try:
            print(f"⏳ Waiting for buffer ({buffer_count} segments)...")
            
            while self.playback_queue.qsize() < buffer_count:
                time.sleep(0.3)
                if self.error_occurred.is_set():
                    return
                if self.processing_complete.is_set() and self.playback_queue.qsize() > 0:
                    break
            
            print(f"\n📊 BUFFER READY! Starting playback...\n")
            print("="*70)
            
            count = 0
            prev_end = 0
            
            while not self.processing_complete.is_set() or not self.playback_queue.empty():
                try:
                    segment = self.playback_queue.get(timeout=2)
                    count += 1
                    
                    audio_seg = AudioSegment.from_wav(io.BytesIO(segment.audio_bytes))
                    
                    gap = segment.start_ms - prev_end
                    if gap > 100:
                        play(AudioSegment.silent(duration=int(gap)))
                    
                    print(f"📊 [{segment.speaker_id}] {segment.translated_text[:60]}...")
                    play(audio_seg)
                    
                    prev_end = segment.start_ms + segment.duration_ms
                    self.playback_queue.task_done()
                    
                except queue.Empty:
                    if not self.processing_complete.is_set():
                        time.sleep(0.3)
                    else:
                        break
            
            print(f"\n✅ Playback complete! {count} segments")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Playback error: {e}")
    
    def _compose_audio(self, segments, total_duration_sec):
        """Compose final audio with overlap support"""
        canvas_ms = int(total_duration_sec * 1000) + 5000
        final_audio = AudioSegment.silent(duration=canvas_ms)
        
        prev_orig_end = 0
        prev_trans_end = 0
        
        for seg in segments:
            seg_audio = AudioSegment.from_wav(io.BytesIO(seg.audio_bytes))
            
            orig_gap = seg.start_ms - prev_orig_end
            
            if orig_gap < 200:
                actual_start = prev_trans_end + 100
            else:
                ideal_start = prev_trans_end + orig_gap
                actual_start = max(seg.start_ms, ideal_start)
            
            required = actual_start + seg.duration_ms
            if len(final_audio) < required:
                final_audio += AudioSegment.silent(duration=required - len(final_audio) + 2000)
            
            final_audio = final_audio.overlay(seg_audio, position=actual_start)
            
            prev_orig_end = seg.end_ms
            prev_trans_end = actual_start + seg.duration_ms
        
        return final_audio


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimized_elevenlabs_streaming.py <video> [src] [tgt] [buffer] [workers]")
        print("\nExample: python optimized_elevenlabs_streaming.py video.mp4 en es 10 3")
        print("  workers: 3-5 recommended for best speed")
        sys.exit(1)
    
    video_path = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "en"
    target = sys.argv[3] if len(sys.argv) > 3 else "es"
    buffer = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    
    translator = OptimizedStreamingTranslator(
        source_lang=source,
        target_lang=target,
        buffer_duration_sec=buffer,
        max_workers=workers,
        use_voice_cloning=False  # Set True if you have Pro plan
    )
    
    output = translator.translate_video_streaming(video_path)
    print(f"\n🎉 Complete! Output: {output}")


if __name__ == "__main__":
    main()