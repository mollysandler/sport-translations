# translator.py

import io
from google.cloud import speech, texttospeech, translate
from google.api_core.exceptions import GoogleAPICallError
from hybrid_config import GOOGLE_PROJECT_ID

# --- OPTIMIZATION: Initialize Clients Globally ---
# Creating these clients is expensive, so we do it once and reuse them.
_speech_client = speech.SpeechClient()
_translate_client = translate.TranslationServiceClient()
_tts_client = texttospeech.TextToSpeechClient()

class SpeechTranslator:
    def __init__(self, source_language: str, target_language: str):
        # Store the languages for this specific request
        self.source_language = source_language
        self.target_language = target_language
        
        # Use the global clients
        self.speech_client = _speech_client
        self.translate_client = _translate_client
        self.tts_client = _tts_client
        
        self.voice_map = {} 

    def _get_speaker_voice(self, speaker_id: str) -> str:
        if speaker_id in self.voice_map:
            return self.voice_map[speaker_id]

        try:
            # Use self.target_language instead of hardcoded value
            voices_response = self.tts_client.list_voices(language_code=self.target_language)
            wavenet_voices = [v for v in voices_response.voices if "Wavenet" in v.name]
            if not wavenet_voices:
                wavenet_voices = voices_response.voices
            
            num_seen_speakers = len(self.voice_map)
            voice_name = wavenet_voices[num_seen_speakers % len(wavenet_voices)].name
            self.voice_map[speaker_id] = voice_name
            print(f"   » Assigning voice '{voice_name}' to {speaker_id}")
            return voice_name
        except Exception as e:
            print(f"   » Could not fetch TTS voices: {e}. Using default.")
            # Fallback logic based on language could go here
            return "en-US-Wavenet-D" 

    def translate_audio_chunk(self, audio_chunk_data: bytes, speaker_id: str):
        # 1. Speech-to-Text
        try:
            audio = speech.RecognitionAudio(content=audio_chunk_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                # Use DYNAMIC source language
                language_code=self.source_language,
            )
            stt_response = self.speech_client.recognize(config=config, audio=audio)
        except Exception as e:
            print(f"   » [{speaker_id}] STT Error: {e}")
            return None

        if not stt_response.results or not stt_response.results[0].alternatives:
            return None
        
        transcript = stt_response.results[0].alternatives[0].transcript
        # print(f"   » [{speaker_id}] Transcription ({self.source_language}): {transcript}") # Optional logging
        
        if not transcript.strip():
            return None

        # 2. Translation
        try:
            # Extract base language codes (e.g., "en-US" -> "en")
            source_lang_code = self.source_language.split('-')[0]
            target_lang_code = self.target_language.split('-')[0]
            
            parent = f"projects/{GOOGLE_PROJECT_ID}/locations/global"

            translation_response = self.translate_client.translate_text(
                parent=parent,
                contents=[transcript],
                target_language_code=target_lang_code,
                source_language_code=source_lang_code,
            )
            translated_text = translation_response.translations[0].translated_text
            # print(f"   » [{speaker_id}] Translation ({self.target_language}): {translated_text}") # Optional logging
        except Exception as e:
            print(f"   » [{speaker_id}] Translation Error: {e}")
            return None

        # 3. Text-to-Speech
        try:
            voice_name = self._get_speaker_voice(speaker_id)
            synthesis_input = texttospeech.SynthesisInput(text=translated_text)
            
            # Use DYNAMIC target language
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.target_language, name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            tts_response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
        except Exception as e:
            print(f"   » [{speaker_id}] TTS Error: {e}")
            return None

        return {
            "original_text": transcript,
            "translated_text": translated_text,
            "audio_data": tts_response.audio_content
        }