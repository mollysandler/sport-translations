# v3_main.py
# Main application for video translation with speaker diarization

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

from config import (
    SEAMLESS_LANGUAGE_MAPPING,
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    SAMPLE_RATE
)
from translator import SeamlessSpeechTranslator
from diarizer import SpeakerDiarizer

# Load environment variables
load_dotenv()


def extract_audio_from_video(video_path: str, output_path: str) -> bool:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save extracted audio (WAV format)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🎬 Extracting audio from '{video_path}'...")
    
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            output_path,
            codec='pcm_s16le',
            fps=SAMPLE_RATE,
            nbytes=2,
            ffmpeg_params=["-ac", "1"],  # Convert to mono
            logger=None  # Suppress moviepy output
        )
        print(f"✅ Audio extracted to '{output_path}'")
        return True
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False


def translate_video(
    video_path: str,
    source_lang: str = DEFAULT_SOURCE,
    target_lang: str = DEFAULT_TARGET,
    play_output: bool = True
):
    """
    Translate a video's audio from source language to target language.
    Preserves speaker diarization and prosody.
    
    Args:
        video_path: Path to input video file
        source_lang: Source language code (e.g., "eng", "spa")
        target_lang: Target language code (e.g., "hin", "fra")
        play_output: Whether to play the translated audio when done
    """
    # Validate file exists
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    # Get HuggingFace token
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("❌ Error: HUGGING_FACE_TOKEN not found in .env file")
        sys.exit(1)
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # Step 1: Extract audio from video
        extracted_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        if not extract_audio_from_video(video_path, extracted_audio_path):
            return
        
        # Step 2: Perform speaker diarization
        diarizer = SpeakerDiarizer(hf_token)
        segments = diarizer.diarize(extracted_audio_path)
        
        if not segments:
            print("❌ No speech segments found in audio")
            return
        
        # Step 3: Initialize translator
        translator = SeamlessSpeechTranslator(source_lang, target_lang)
        
        # Step 4: Load full audio for segment extraction
        full_audio = AudioSegment.from_wav(extracted_audio_path)
        
        # Step 5: Create silent base for composition
        final_composition = AudioSegment.silent(duration=len(full_audio))
        
        print(f"\n{'='*60}")
        print(f"🔄 Translating {len(segments)} segments...")
        print(f"{'='*60}\n")
        
        # Step 6: Process each segment
        successful_translations = 0
        for i, segment in enumerate(segments, 1):
            print(f"[{i}/{len(segments)}] {segment.speaker_id}: "
                  f"{segment.start_sec:.2f}s - {segment.end_sec:.2f}s "
                  f"({segment.duration_ms}ms)")
            
            # Extract audio segment
            segment_audio = full_audio[segment.start_ms:segment.end_ms]
            segment_wav_bytes = segment_audio.export(format="wav").read()
            
            # Translate segment
            result = translator.translate_segment(
                audio_chunk_data=segment_wav_bytes,
                speaker_id=segment.speaker_id
            )
            
            if result and result.get('audio_data'):
                # Load translated audio
                translated_audio = AudioSegment.from_wav(
                    io.BytesIO(result['audio_data'])
                )
                
                # Overlay at original position
                final_composition = final_composition.overlay(
                    translated_audio,
                    position=segment.start_ms
                )
                
                # Show translation text
                if result.get('translated_text'):
                    print(f"   📝 \"{result['translated_text']}\"")
                
                successful_translations += 1
            else:
                print(f"   ⚠️ Translation failed for this segment")
            
            print()  # Blank line for readability
        
        # Step 7: Summary
        print(f"{'='*60}")
        print(f"✅ Translation complete!")
        print(f"   Speakers: {translator.get_speaker_count()}")
        print(f"   Segments: {successful_translations}/{len(segments)} successful")
        print(f"{'='*60}\n")
        
        # Step 8: Save output
        output_path = os.path.join(
            os.path.dirname(video_path),
            f"translated_{Path(video_path).stem}_{target_lang}.wav"
        )
        final_composition.export(output_path, format="wav")
        print(f"💾 Saved translated audio to: {output_path}")
        
        # Step 9: Play if requested
        if play_output:
            print("🔊 Playing translated audio...")
            play(final_composition)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Translation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("👋 Done!\n")


def main():
    """Command-line interface for the translator."""
    
    if len(sys.argv) < 2:
        print("Usage: python v3_main.py <video_file> [source_lang] [target_lang]")
        print("\nExamples:")
        print("  python v3_main.py commentary.mp4")
        print("  python v3_main.py commentary.mp4 eng spa")
        print("  python v3_main.py commentary.mp4 eng hin")
        print("\nSupported languages:")
        for code, seamless_code in sorted(SEAMLESS_LANGUAGE_MAPPING.items()):
            print(f"  {code:3s} -> {seamless_code}")
        sys.exit(1)
    
    video_path = sys.argv[1]
    source_lang = sys.argv[2] if len(sys.argv) > 2 else "eng"
    target_lang = sys.argv[3] if len(sys.argv) > 3 else "hin"
    
    # Map short codes to SeamlessM4T codes if needed
    source_lang = SEAMLESS_LANGUAGE_MAPPING.get(source_lang, source_lang)
    target_lang = SEAMLESS_LANGUAGE_MAPPING.get(target_lang, target_lang)
    
    print(f"\n{'='*60}")
    print(f"🌐 Video Translation System v3")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Translation: {source_lang} → {target_lang}")
    print(f"{'='*60}\n")
    
    translate_video(video_path, source_lang, target_lang)


if __name__ == "__main__":
    main()