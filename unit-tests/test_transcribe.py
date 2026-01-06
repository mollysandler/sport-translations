from dotenv import load_dotenv
load_dotenv()
from v3.translator import AccurateTranslator


def test():
    translator = AccurateTranslator("en", "en")
    audio_path = "extracted_audio.wav" 
    
    print("running whisper...")
    segments = translator.transcribe_full_audio(audio_path)
    
    print("\n--- TRANSCRIPT ---")
    for s in segments:
        print(f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}")

if __name__ == "__main__":
    test()