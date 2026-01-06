# test_tts_playback.py (Revised)
import io
from dotenv import load_dotenv
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play

# --- NEW: Import language settings from your config file ---
from hybrid_config import TARGET_LANGUAGE

# --- NEW: Load environment variables from .env file ---
# This line looks for a .env file and loads the variables from it,
# including GOOGLE_APPLICATION_CREDENTIALS.
load_dotenv()

print("--- TTS and Playback Test ---")
print(f"This test will synthesize a test sentence in the target language: '{TARGET_LANGUAGE}'")

try:
    # 1. Test Google Cloud Authentication and TTS
    print("Connecting to Google Cloud TTS...")
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text="This is a test of the text to speech system.")
    
    # --- UPDATED: Use the language from config.py ---
    voice = texttospeech.VoiceSelectionParams(
        language_code=TARGET_LANGUAGE, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    print("✅ Speech synthesized successfully.")

    # 2. Test Pydub and FFmpeg for playback
    print("Loading audio data for playback...")
    audio_segment = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
    
    print("Playing audio...")
    play(audio_segment)
    print("✅ Audio playback finished.")
    print("\n--- Test Passed! ---")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("--- Test Failed! ---")
    print("\nTroubleshooting:")
    print("- If the error still mentions 'credentials', double-check that the path in your .env file is correct and the file exists.")
    print("- Ensure the GOOGLE_APPLICATION_CREDENTIALS variable is spelled correctly in your .env file.")
    print("- If the error mentions 'ffmpeg', ensure it is installed (run 'brew install ffmpeg').")