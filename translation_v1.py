import os
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from deepgram import DeepgramClient, PrerecordedOptions
from google.cloud import texttospeech

load_dotenv()
AUDIO_FILE_PATH = "live.m4a"
OUTPUT_AUDIO_FILENAME = "output_speech.mp3"

# --- !! LANGUAGE CONFIGURATION !! ---
# all suppported languages: https://cloud.google.com/text-to-speech/docs/list-voices-and-types#list_of_all_supported_languages
SOURCE_LANGUAGE = "en-US" 
TARGET_LANGUAGE = "hi-IN" 

def translate_text_with_mymemory(text, source_lang, target_lang):
    print(f"\n2. Translating from '{source_lang}' to '{target_lang}'...")
    try:
        api_url = "https://api.mymemory.translated.net/get"
        params = {"q": text, "langpair": f"{source_lang}|{target_lang}"}
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        translated_text = data["responseData"]["translatedText"]
        print(f"   -> Translation successful: '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"   -> Error during translation: {e}")
        return None

def generate_audio_with_google(text, language_code, output_file):
    """Generates speech from text using Google Cloud TTS."""
    try:
        print(f"\n3. Converting text to speech in '{language_code}' using Google TTS...")
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"   -> Audio saved to {output_file}. Now playing...")
        audio = AudioSegment.from_mp3(output_file)
        play(audio)
        return True
    except Exception as e:
        print(f"   -> An error occurred with Google TTS: {e}")
        return False

def main():
    """Transcribes, translates, and speaks, using configurable languages."""
    try:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        deepgram = DeepgramClient(api_key)

        with open(AUDIO_FILE_PATH, "rb") as audio_file:
            buffer_data = audio_file.read()
        payload = {"buffer": buffer_data}

        options = PrerecordedOptions(model="nova-2", smart_format=True, language=SOURCE_LANGUAGE)
        print(f"1. Transcribing audio (assuming source is '{SOURCE_LANGUAGE}')...")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        original_transcript = response.results.channels[0].alternatives[0].transcript
        
        if original_transcript:
            print(f"   -> Transcription successful: '{original_transcript}'")
            
            source_lang_short = SOURCE_LANGUAGE.split('-')[0]
            translated_transcript = translate_text_with_mymemory(
                original_transcript, 
                source_lang=source_lang_short, 
                target_lang=TARGET_LANGUAGE
            )

            if translated_transcript:
                generate_audio_with_google(
                    text=translated_transcript, 
                    language_code=TARGET_LANGUAGE, 
                    output_file=OUTPUT_AUDIO_FILENAME
                )
            
            print("\n--- Process Complete ---")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()