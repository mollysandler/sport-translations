# main.py

import os
import sys
import tempfile
import shutil
import io
from pathlib import Path
import torch
from dotenv import load_dotenv

from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.playback import play
from pyannote.audio import Pipeline

from translator import SpeechTranslator
from config import TARGET_LANGUAGE, STT_SAMPLE_RATE

# Load .env file for credentials
load_dotenv()

def extract_audio(video_path: str, output_path: str):
    print(f"🎬 Extracting audio from '{video_path}'...")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            output_path, 
            codec='pcm_s16le', 
            fps=STT_SAMPLE_RATE,
            nbytes=2,
            ffmpeg_params=["-ac", "1"]
        )
        print(f"🔊 Audio extracted successfully to '{output_path}'")
        return True
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_video_file>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        extracted_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        if not extract_audio(video_path, extracted_audio_path):
            return

        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if hf_token is None:
            print("❌ Error: HUGGING_FACE_TOKEN not found in .env file.")
            sys.exit(1)
            
        print("LOADING: Speaker diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pipeline.to(device)
        print("DIARIZING: Identifying speakers in the audio...")
        diarization = pipeline(extracted_audio_path)
        print("✅ Diarization complete.")

        translator = SpeechTranslator(target_language=TARGET_LANGUAGE)
        full_original_audio = AudioSegment.from_wav(extracted_audio_path)
        final_composition = AudioSegment.silent(duration=len(full_original_audio))

        print("\n--- Starting translation of speaker segments ---")
        
        # --- THE FINAL FIX: Iterating over the correct attribute ---
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            
            print(f"\nProcessing segment for {speaker} from {start_ms}ms to {end_ms}ms")
            
            segment_audio = full_original_audio[start_ms:end_ms]
            segment_wav_data = segment_audio.export(format="wav").read()
            
            translated_audio_data = translator.translate_audio_chunk(
                audio_chunk_data=segment_wav_data,
                speaker_id=speaker
            )

            if translated_audio_data:
                translated_segment = AudioSegment.from_file(io.BytesIO(translated_audio_data), format="mp3")
                final_composition = final_composition.overlay(translated_segment, position=start_ms)

        print("\n✅ All segments processed. Playing final composed translation...")
        play(final_composition)

    except Exception as e:
        import traceback
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()
        
    finally:
        print("🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("👋 Application finished.")

if __name__ == "__main__":
    main()