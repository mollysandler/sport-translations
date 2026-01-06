# test_4_tts.py
from v3.translator import AccurateTranslator
from pydub import AudioSegment
from pydub.playback import play
import io

def test():
    translator = AccurateTranslator("en", "es") # Test translation too
    
    # Hardcoded test case
    text = "The Philadelphia Eagles beat the Vikings 28 to 22."
    speaker = "SPEAKER_00"
    
    print(f"Generating: {text}")
    audio_bytes = translator.synthesize_speech(text, speaker)
    
    song = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    play(song)

if __name__ == "__main__":
    test()