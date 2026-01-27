# streaming_hybrid_v2.py
"""
FIXED v2: Streaming translation with proper segment processing
- Phase 1: Analyze speakers AND get ALL diarization segments from entire video
- Phase 2: Process segments in order with buffered playback
- This ensures we don't miss any speech!
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

from hybrid_system import HybridSportsTranslator, TranslationSegment
from diarizer import SpeakerDiarizer, SpeakerSegment


class StreamingTranslatorV2(HybridSportsTranslator):
    """
    Improved streaming translator that processes actual speaker segments
    """
    
    def __init__(self, *args, buffer_duration_sec: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_duration_sec = buffer_duration_sec
        self.playback_queue = queue.Queue()
        self.all_segments = []
        self.segments_lock = threading.Lock()
        self.processing_complete = threading.Event()
        self.error_occurred = threading.Event()
        self.error_message = None
        
        # Optional: Parallel TTS processing
        from concurrent.futures import ThreadPoolExecutor
        self.tts_executor = ThreadPoolExecutor(max_workers=2)  # Process 2 TTS in parallel
    
    def translate_video_streaming(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> str:
        """
        Translate video with streaming playback
        Returns path to final output file
        """
        print("="*70)
        print("🎬 STREAMING TRANSLATION PIPELINE V2")
        print("="*70)
        
        # 1. Load audio
        audio_path = self._extract_audio(video_path)
        audio, sr = torchaudio.load(audio_path)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        total_duration_sec = audio.shape[1] / self.sample_rate
        print(f"📊 Total duration: {total_duration_sec:.1f}s")
        print(f"⏱️  Buffer: {self.buffer_duration_sec}s")
        print()
        
        # 2. PHASE 1: Complete Analysis (speakers + ALL segments)
        print("="*70)
        print("🔍 PHASE 1: SPEAKER DISCOVERY & SEGMENTATION")
        print("="*70)
        
        speaker_segments = self._full_analysis(audio, audio_path, hf_token)
        
        print(f"\n📊 Found {len(speaker_segments)} speech segments to translate")
        
        # Calculate buffer in terms of segments
        buffer_duration_so_far = 0
        buffer_segment_count = 0
        for seg in speaker_segments:
            seg_duration = (seg.end_sec - seg.start_sec)
            if buffer_duration_so_far < self.buffer_duration_sec:
                buffer_segment_count += 1
                buffer_duration_so_far += seg_duration
            else:
                break
        
        print(f"⏱️  Will buffer first {buffer_segment_count} segments (~{buffer_duration_so_far:.1f}s)\n")
        
        # 3. PHASE 2: Process segments with streaming playback
        print("="*70)
        print("🚀 PHASE 2: STREAMING TRANSLATION")
        print("="*70)
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=self._process_segments_background,
            args=(audio, speaker_segments),
            daemon=False
        )
        processing_thread.start()
        
        # Start playback thread
        playback_thread = threading.Thread(
            target=self._playback_thread,
            args=(buffer_segment_count,),
            daemon=False
        )
        playback_thread.start()
        
        # Wait for completion
        processing_thread.join()
        playback_thread.join()
        
        # 4. Check for errors
        if self.error_occurred.is_set():
            print(f"\n❌ Error occurred: {self.error_message}")
            sys.exit(1)
        
        # 5. Save final composition
        print("\n" + "="*70)
        print("💾 SAVING FINAL OUTPUT")
        print("="*70)
        
        if output_path is None:
            output_path = self._generate_output_path(video_path)
        
        with self.segments_lock:
            if self.all_segments:
                print(f"📝 Composing {len(self.all_segments)} segments...")
                final_audio = self._compose_audio(self.all_segments, total_duration_sec)
                final_audio.export(output_path, format="wav")
                print(f"✅ Saved to: {output_path}")
            else:
                print("⚠️  No segments to save")
        
        return output_path
    
    def _full_analysis(
        self,
        audio: torch.Tensor,
        audio_path: str,
        hf_token: Optional[str]
    ) -> List[SpeakerSegment]:
        """
        Phase 1: Analyze ENTIRE video for speakers and segments
        """
        total_duration = audio.shape[1] / self.sample_rate
        
        # For short videos, analyze everything
        # For long videos, analyze first portion for speaker discovery
        analysis_duration = min(
            self.analysis_duration_sec,
            max(10, total_duration * 0.3)
        )
        
        print(f"🔍 Step 1: Analyzing first {analysis_duration:.1f}s for speaker profiles...")
        
        analysis_samples = int(analysis_duration * self.sample_rate)
        analysis_audio = audio[:, :analysis_samples]
        
        # Save analysis segment
        temp_analysis_path = "temp_analysis_segment.wav"
        torchaudio.save(temp_analysis_path, analysis_audio, self.sample_rate)
        
        # Get speaker profiles from analysis segment
        if hf_token is None:
            hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        diarizer = SpeakerDiarizer(hf_token)
        analysis_segments = diarizer.diarize(temp_analysis_path)
        os.remove(temp_analysis_path)
        
        # Build voice profiles from analysis
        self._build_voice_profiles(audio, analysis_segments)
        
        # Now run diarization on FULL audio to get ALL segments
        print(f"\n🔍 Step 2: Finding ALL speech segments in full {total_duration:.1f}s video...")
        temp_full_path = "temp_full_audio.wav"
        torchaudio.save(temp_full_path, audio, self.sample_rate)
        
        all_segments = diarizer.diarize(temp_full_path)
        os.remove(temp_full_path)
        
        print(f"✅ Found {len(all_segments)} speech segments across entire video")
        
        # Show segment breakdown
        speaker_counts = {}
        for seg in all_segments:
            speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
        
        print(f"\n📊 Segment breakdown:")
        for speaker_id in sorted(speaker_counts.keys()):
            count = speaker_counts[speaker_id]
            print(f"   {speaker_id}: {count} segments")
        
        return all_segments
    
    def _build_voice_profiles(self, audio: torch.Tensor, segments: List[SpeakerSegment]):
        """Build voice profiles from segments"""
        print(f"\n🎤 Extracting voice profiles...")
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        
        for speaker_id in unique_speakers:
            speaker_segments = [
                seg for seg in segments 
                if seg.speaker_id == speaker_id
            ]
            
            collected_audio = []
            total_duration = 0
            target_duration = 10.0
            
            for seg in speaker_segments:
                if total_duration >= target_duration:
                    break
                
                start_sample = int(seg.start_sec * self.sample_rate)
                end_sample = int(seg.end_sec * self.sample_rate)
                segment_audio = audio[:, start_sample:end_sample]
                
                seg_duration = (end_sample - start_sample) / self.sample_rate
                if seg_duration >= 1.0:
                    collected_audio.append(segment_audio)
                    total_duration += seg_duration
            
            if collected_audio:
                voice_profile = torch.cat(collected_audio, dim=1)
                max_samples = int(target_duration * self.sample_rate)
                if voice_profile.shape[1] > max_samples:
                    voice_profile = voice_profile[:, :max_samples]
                
                self.speaker_voice_profiles[speaker_id] = voice_profile.squeeze().numpy()
                
                embedding = self.speaker_encoder.encode_batch(voice_profile)
                self.speaker_embeddings[speaker_id] = embedding
                
                profile_duration = voice_profile.shape[1] / self.sample_rate
                print(f"   ✅ {speaker_id}: {profile_duration:.1f}s voice profile")
        
        print(f"\n🎯 Discovered {len(self.speaker_voice_profiles)} speakers")
        print(f"   Speakers: {', '.join(sorted(self.speaker_voice_profiles.keys()))}")
    
    def _process_segments_background(
        self,
        audio: torch.Tensor,
        speaker_segments: List[SpeakerSegment]
    ):
        """Process each diarization segment"""
        try:
            start_time = time.time()
            
            for idx, seg in enumerate(speaker_segments, 1):
                # Extract audio for this segment
                start_sample = int(seg.start_sec * self.sample_rate)
                end_sample = int(seg.end_sec * self.sample_rate)
                segment_audio = audio[:, start_sample:end_sample]
                
                # Process segment
                result = self._process_segment(
                    segment_audio,
                    seg,
                    idx,
                    len(speaker_segments)
                )
                
                if result:
                    self.playback_queue.put(result)
                    with self.segments_lock:
                        self.all_segments.append(result)
                    
                    elapsed = time.time() - start_time
                    progress = (idx / len(speaker_segments)) * 100
                    queue_size = self.playback_queue.qsize()
                    
                    print(f"⚙️  Processing: {progress:.1f}% ({idx}/{len(speaker_segments)}) | "
                          f"Queue: {queue_size} | Elapsed: {elapsed:.0f}s")
            
            self.processing_complete.set()
            print(f"\n✅ Processing complete! Total time: {time.time() - start_time:.1f}s")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_segment(
        self,
        audio_chunk: torch.Tensor,
        speaker_seg: SpeakerSegment,
        seg_num: int,
        total_segs: int
    ) -> Optional[TranslationSegment]:
        """Process a single speaker segment"""
        
        # Skip very short segments
        duration_sec = (speaker_seg.end_sec - speaker_seg.start_sec)
        if duration_sec < 0.3:
            print(f"[Segment {seg_num}/{total_segs}] Skipped (too short: {duration_sec:.2f}s)")
            return None
        
        # Get speaker from diarization (we already know it!)
        speaker_id = speaker_seg.speaker_id
        
        # Transcribe
        chunk_np = audio_chunk.squeeze().numpy()
        segments, info = self.whisper.transcribe(
            chunk_np,
            language=self.source_lang,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False
        )
        
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        if not text_parts:
            print(f"[Segment {seg_num}/{total_segs}] Skipped (no transcription)")
            return None
        
        original_text = " ".join(text_parts)
        
        # Translate
        translated_text = self._translate_text(original_text)
        
        # Synthesize
        audio_bytes = self._synthesize_speech(translated_text, speaker_id)
        
        if not audio_bytes:
            return None
        
        # Calculate duration
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        duration_ms = len(audio_segment)
        
        # Show progress
        original_preview = original_text[:50] + "..." if len(original_text) > 50 else original_text
        translated_preview = translated_text[:50] + "..." if len(translated_text) > 50 else translated_text
        print(f"[Seg {seg_num}/{total_segs}] {speaker_id}: \"{original_preview}\" → \"{translated_preview}\"")
        
        return TranslationSegment(
            speaker_id=speaker_id,
            start_ms=speaker_seg.start_ms,
            end_ms=speaker_seg.end_ms,
            original_text=original_text,
            translated_text=translated_text,
            audio_bytes=audio_bytes,
            duration_ms=duration_ms
        )
    
    def _playback_thread(self, buffer_segment_count: int):
        """Playback with segment-based buffering"""
        try:
            print(f"⏳ Waiting for buffer ({buffer_segment_count} segments)...")
            
            segments_ready = 0
            last_update = 0
            
            while segments_ready < buffer_segment_count:
                time.sleep(0.5)
                segments_ready = self.playback_queue.qsize()
                
                if int(time.time()) > last_update:
                    last_update = int(time.time())
                    progress = min(100, (segments_ready / buffer_segment_count) * 100)
                    print(f"   Buffer: {segments_ready}/{buffer_segment_count} ({progress:.0f}%)")
                
                if self.error_occurred.is_set():
                    return
                
                if self.processing_complete.is_set():
                    if self.playback_queue.empty():
                        print("\n⚠️  Processing finished before buffer filled")
                        return
                    else:
                        print(f"\n📊 Short video: Starting playback with {segments_ready} segments")
                        break
            
            print(f"\n📊 BUFFER READY! Starting playback...\n")
            print("="*70)
            
            prev_end_ms = 0
            segment_count = 0
            
            while not self.processing_complete.is_set() or not self.playback_queue.empty():
                try:
                    segment = self.playback_queue.get(timeout=2)
                    segment_count += 1
                    
                    audio_segment = AudioSegment.from_wav(
                        io.BytesIO(segment.audio_bytes)
                    )
                    
                    # Add gap if needed
                    gap_ms = segment.start_ms - prev_end_ms
                    if gap_ms > 100:
                        silence = AudioSegment.silent(duration=int(gap_ms))
                        play(silence)
                    
                    # Play segment
                    text_preview = segment.translated_text[:60]
                    if len(segment.translated_text) > 60:
                        text_preview += "..."
                    
                    print(f"📊 [{segment.speaker_id}] {text_preview}")
                    play(audio_segment)
                    
                    prev_end_ms = segment.start_ms + segment.duration_ms
                    self.playback_queue.task_done()
                    
                except queue.Empty:
                    if not self.processing_complete.is_set():
                        print("   ⏳ Waiting for next segment...")
                        time.sleep(0.5)
                    else:
                        break
                except Exception as e:
                    print(f"⚠️  Playback error: {e}")
                    continue
            
            print(f"\n✅ Playback complete! Played {segment_count} segments")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Playback error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Command-line interface for streaming translation v2"""
    if len(sys.argv) < 2:
        print("Usage: python streaming_hybrid_v2.py <video_path> [source_lang] [target_lang] [buffer_sec]")
        print("\nExample: python streaming_hybrid_v2.py video.mp4 en es 300")
        print("         (300 seconds = 5 minute buffer)")
        print("\nFor shorter videos:")
        print("         python streaming_hybrid_v2.py short_clip.mp4 en es 10")
        sys.exit(1)
    
    video_path = sys.argv[1]
    source_lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    target_lang = sys.argv[3] if len(sys.argv) > 3 else "es"
    buffer_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    # Create streaming translator
    translator = StreamingTranslatorV2(
        source_lang=source_lang,
        target_lang=target_lang,
        analysis_duration_sec=30,
        chunk_duration_sec=2.0,  # Not used in v2, kept for compatibility
        buffer_duration_sec=buffer_sec
    )
    
    # Start streaming translation
    output_path = translator.translate_video_streaming(video_path)
    
    print(f"\n🎉 Complete! Final output: {output_path}")


if __name__ == "__main__":
    main()