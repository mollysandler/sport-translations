# main.py
import os
import sys
import tempfile
import shutil
import io
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from moviepy.editor import VideoFileClip

from hybrid_config import DEFAULT_SOURCE, DEFAULT_TARGET, SAMPLE_RATE
from v3.translator import AccurateTranslator
from diarizer import SpeakerDiarizer

load_dotenv()

def extract_audio_from_video(video_path: str, output_path: str) -> bool:
    print(f"🎬 Extracting audio from '{video_path}'...")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            output_path,
            codec='pcm_s16le',
            fps=SAMPLE_RATE,
            nbytes=2,
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        return True
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False

def get_speaker_for_segment(whisper_start, whisper_end, diarization_segments):
    """
    Find which speaker talked the most during this text segment.
    """
    max_overlap = 0
    best_speaker = "SPEAKER_00" 
    
    for diag_seg in diarization_segments:
        start_ov = max(whisper_start, diag_seg.start_sec)
        end_ov = min(whisper_end, diag_seg.end_sec)
        overlap = max(0, end_ov - start_ov)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = diag_seg.speaker_id
            
    return best_speaker

def translate_video(video_path: str, source_lang: str, target_lang: str):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("❌ HUGGING_FACE_TOKEN missing")
        sys.exit(1)

    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Setup
        audio_path = os.path.join(temp_dir, "extracted.wav")
        if not extract_audio_from_video(video_path, audio_path):
            return
            
        translator = AccurateTranslator(source_lang, target_lang)
        diarizer = SpeakerDiarizer(hf_token)
        
        # 2. Run Processes
        print(f"\n{'='*60}")
        print("🚀 PHASE 1: Transcription")
        text_segments = translator.transcribe_full_audio(audio_path)
        
        print("\n🚀 PHASE 2: Diarization")
        speaker_segments = diarizer.diarize(audio_path)
        
        # 3. Alignment & Synthesis
        print("\n🚀 PHASE 3: Relative Sync & Composition")
        print(f"{'='*60}")
        
        original_audio = AudioSegment.from_wav(audio_path)
        final_composition = AudioSegment.silent(duration=len(original_audio))
        
        # --- TIMING VARIABLES ---
        # Track when the previous sentence ended in both timelines
        prev_original_end_ms = 0
        prev_translated_end_ms = 0
        # ------------------------

        for i, segment in enumerate(text_segments):
            # 1. Get timestamps
            current_original_start_ms = int(segment['start'] * 1000)
            current_original_end_ms = int(segment['end'] * 1000)
            
            original_text = segment['text'].strip()
            if not original_text:
                continue
                
            # 2. Identify Speaker & Translate
            speaker_id = get_speaker_for_segment(segment['start'], segment['end'], speaker_segments)
            translated_text = translator.translate_text(original_text)
            
            # 3. Synthesize
            audio_bytes = translator.synthesize_speech(translated_text, speaker_id)
            
            if audio_bytes:
                segment_audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                duration_ms = len(segment_audio)
                
                # --- RELATIVE SYNC LOGIC ---
                
                # A. Calculate the gap in the ORIGINAL video. 
                # Positive = Silence. Negative = Interruption/Overlap.
                original_gap_ms = current_original_start_ms - prev_original_end_ms
                
                # B. Apply that gap to the NEW timeline.
                # If the gap was -1000ms (interruption), we start 1000ms before the previous audio ends.
                ideal_start_ms = prev_translated_end_ms + original_gap_ms
                
                # C. Safety Check: Don't travel back in time before the video starts.
                # If translation was faster than original, wait for the video to catch up.
                actual_start_ms = max(current_original_start_ms, ideal_start_ms)
                
                # Debug Info
                drift = actual_start_ms - current_original_start_ms
                print(f"[{i+1}] {speaker_id}: (Gap: {original_gap_ms}ms | Drift: {drift/1000:.1f}s) \"{translated_text}\"")

                # ---------------------------

                # Extend canvas if needed
                required_length = actual_start_ms + duration_ms
                if len(final_composition) < required_length:
                    padding = required_length - len(final_composition) + 2000 
                    final_composition += AudioSegment.silent(duration=padding)

                # Overlay
                final_composition = final_composition.overlay(segment_audio, position=actual_start_ms)
                
                # Update trackers
                prev_original_end_ms = current_original_end_ms
                
                # IMPORTANT: For the next loop, the "end" of this segment depends on 
                # whether we just created an overlap or a sequence.
                # We typically track where this audio file PHYSICALLY ends.
                prev_translated_end_ms = actual_start_ms + duration_ms
            
        # 4. Save
        output_path = os.path.join(
            os.path.dirname(video_path),
            f"translated_{Path(video_path).stem}_{target_lang}.wav"
        )
        final_composition.export(output_path, format="wav")
        print(f"\n💾 Saved to: {output_path}")
        print("🔊 Playing result...")
        play(final_composition)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video> [source] [target]")
        sys.exit(1)
    
    source = sys.argv[2] if len(sys.argv) > 2 else "en"
    target = sys.argv[3] if len(sys.argv) > 3 else "en"
    translate_video(sys.argv[1], source, target)

if __name__ == "__main__":
    main()