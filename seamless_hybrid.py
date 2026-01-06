# streaming_hybrid.py
"""
Streaming translation with buffer playback
- Processes chunks in background
- Starts playing after initial buffer (5 minutes)
- Continues playing seamlessly while processing
"""

import os
import sys
import threading
import queue
import time
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
from pydub.playback import play
import torchaudio

from hybrid_system import HybridSportsTranslator, TranslationSegment


class StreamingTranslator(HybridSportsTranslator):
    """
    Extends HybridSportsTranslator with streaming playback capability
    """
    
    def __init__(self, *args, buffer_duration_sec: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_duration_sec = buffer_duration_sec  # 5 minutes default
        self.playback_queue = queue.Queue()
        self.processing_complete = threading.Event()
        self.error_occurred = threading.Event()
        self.error_message = None
    
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
        print("🎬 STREAMING TRANSLATION PIPELINE")
        print("="*70)
        
        # 1. Load and setup (same as before)
        audio_path = self._extract_audio(video_path)
        audio, sr = torchaudio.load(audio_path)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        total_duration_sec = audio.shape[1] / self.sample_rate
        print(f"📊 Total duration: {total_duration_sec:.1f}s")
        print(f"⏱️  Buffer: {self.buffer_duration_sec}s (~{self.buffer_duration_sec//13} chunks)")
        print()
        
        # 2. PHASE 1: Speaker Discovery
        print("="*70)
        print("🔍 PHASE 1: SPEAKER DISCOVERY")
        print("="*70)
        
        actual_analysis_duration = min(
            self.analysis_duration_sec,
            max(5, total_duration_sec * 0.3)
        )
        
        analysis_samples = int(actual_analysis_duration * self.sample_rate)
        analysis_audio = audio[:, :analysis_samples]
        
        self._analyze_speakers(analysis_audio, audio_path, hf_token)
        
        # 3. PHASE 2: Start Processing + Playback Threads
        print("\n" + "="*70)
        print("🚀 PHASE 2: STREAMING TRANSLATION")
        print("="*70)
        print(f"⚙️  Starting background processing...")
        print(f"🔊 Will begin playback after {self.buffer_duration_sec}s buffer\n")
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=self._process_in_background,
            args=(audio, analysis_samples, total_duration_sec)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start playback thread (waits for buffer)
        playback_thread = threading.Thread(
            target=self._playback_thread,
            args=(total_duration_sec,)
        )
        playback_thread.daemon = True
        playback_thread.start()
        
        # Wait for both to complete
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
        
        # Compose all segments
        all_segments = []
        while not self.playback_queue.empty():
            try:
                seg = self.playback_queue.get_nowait()
                all_segments.append(seg)
            except queue.Empty:
                break
        
        if all_segments:
            final_audio = self._compose_audio(all_segments, total_duration_sec)
            final_audio.export(output_path, format="wav")
            print(f"✅ Saved to: {output_path}")
        else:
            print("⚠️  No segments to save")
        
        return output_path
    
    def _process_in_background(self, audio, start_sample, total_duration_sec):
        """Background thread that processes chunks and adds to queue"""
        try:
            chunk_samples = int(self.chunk_duration_sec * self.sample_rate)
            total_samples = audio.shape[1]
            
            current_sample = start_sample
            chunk_num = 0
            
            start_time = time.time()
            
            while current_sample < total_samples:
                chunk_num += 1
                
                # Extract chunk
                end_sample = min(current_sample + chunk_samples, total_samples)
                chunk = audio[:, current_sample:end_sample]
                
                chunk_start_sec = current_sample / self.sample_rate
                chunk_end_sec = end_sample / self.sample_rate
                
                # Process chunk
                result = self._process_chunk(
                    chunk,
                    chunk_start_sec,
                    chunk_end_sec,
                    chunk_num
                )
                
                if result:
                    # Add to playback queue
                    self.playback_queue.put(result)
                    
                    # Progress indicator
                    elapsed = time.time() - start_time
                    progress = (current_sample - start_sample) / (total_samples - start_sample) * 100
                    queue_size = self.playback_queue.qsize()
                    
                    print(f"⚙️  Processing: {progress:.1f}% | "
                          f"Queue: {queue_size} chunks | "
                          f"Elapsed: {elapsed:.0f}s")
                
                current_sample = end_sample
            
            # Signal completion
            self.processing_complete.set()
            print(f"\n✅ Processing complete! Total time: {time.time() - start_time:.1f}s")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _playback_thread(self, total_duration_sec):
        """Background thread that plays audio as it becomes available"""
        try:
            # Calculate buffer requirements
            buffer_chunks_needed = int(self.buffer_duration_sec / self.chunk_duration_sec)
            
            print(f"🕐 Waiting for buffer ({buffer_chunks_needed} chunks)...")
            
            # Wait for buffer to fill
            chunks_ready = 0
            while chunks_ready < buffer_chunks_needed:
                time.sleep(1)
                chunks_ready = self.playback_queue.qsize()
                
                if chunks_ready % 5 == 0 and chunks_ready > 0:
                    progress = (chunks_ready / buffer_chunks_needed) * 100
                    print(f"   Buffer: {chunks_ready}/{buffer_chunks_needed} "
                          f"({progress:.0f}%) - {self.buffer_duration_sec - (chunks_ready * self.chunk_duration_sec):.0f}s remaining")
                
                # Check if processing failed
                if self.error_occurred.is_set():
                    return
                
                # Check if processing finished early
                if self.processing_complete.is_set() and self.playback_queue.empty():
                    print("\n⚠️  Processing finished before buffer filled")
                    break
            
            print(f"\n🔊 BUFFER READY! Starting playback...\n")
            print("="*70)
            
            # Start playing
            played_segments = []
            prev_end_ms = 0
            
            while not self.processing_complete.is_set() or not self.playback_queue.empty():
                try:
                    # Get next segment (wait up to 1 second)
                    segment = self.playback_queue.get(timeout=1)
                    
                    # Play audio
                    audio_segment = AudioSegment.from_wav(
                        io.BytesIO(segment.audio_bytes)
                    )
                    
                    # Calculate any gap from previous segment
                    gap_ms = segment.start_ms - prev_end_ms
                    if gap_ms > 100:  # More than 100ms gap
                        silence = AudioSegment.silent(duration=gap_ms)
                        play(silence)
                    
                    # Play this segment
                    print(f"🔊 [{segment.speaker_id}] {segment.translated_text[:60]}...")
                    play(audio_segment)
                    
                    played_segments.append(segment)
                    prev_end_ms = segment.start_ms + segment.duration_ms
                    
                except queue.Empty:
                    # Queue empty but processing not done - wait a bit
                    if not self.processing_complete.is_set():
                        print("   ⏳ Waiting for next chunk...")
                        time.sleep(0.5)
                    else:
                        break
            
            print("\n✅ Playback complete!")
            
            # Put segments back in queue for final save
            for seg in played_segments:
                self.playback_queue.put(seg)
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Playback error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Command-line interface for streaming translation"""
    if len(sys.argv) < 2:
        print("Usage: python streaming_hybrid.py <video_path> [source_lang] [target_lang] [buffer_sec]")
        print("\nExample: python streaming_hybrid.py video.mp4 en es 300")
        print("         (300 seconds = 5 minute buffer)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    source_lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    target_lang = sys.argv[3] if len(sys.argv) > 3 else "es"
    buffer_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 300  # 5 minutes
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    # Create streaming translator
    translator = StreamingTranslator(
        source_lang=source_lang,
        target_lang=target_lang,
        analysis_duration_sec=30,
        chunk_duration_sec=2.0,
        buffer_duration_sec=buffer_sec
    )
    
    # Start streaming translation
    output_path = translator.translate_video_streaming(video_path)
    
    print(f"\n🎉 Complete! Final output: {output_path}")


if __name__ == "__main__":
    import io
    import torch
    main()