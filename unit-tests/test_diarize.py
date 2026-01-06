# test_2_diarize.py
import os
from dotenv import load_dotenv
from diarizer import SpeakerDiarizer

load_dotenv()

def test():
    token = os.getenv("HUGGING_FACE_TOKEN")
    diarizer = SpeakerDiarizer(token)
    audio_path = "extracted_audio.wav"
    
    print("running diarization...")
    segments = diarizer.diarize(audio_path)
    
    print("\n--- SPEAKERS ---")
    for s in segments:
        print(f"[{s.start_sec:.2f} - {s.end_sec:.2f}] {s.speaker_id}")

if __name__ == "__main__":
    test()