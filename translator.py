# translator.py

import io
from pathlib import Path
from google.cloud import speech, texttospeech, translate
from config import SOURCE_LANGUAGE, GOOGLE_PROJECT_ID
from google.api_core.exceptions import GoogleAPICallError

class SpeechTranslator:
    """
    A simple, reusable class for the STT -> Translate -> TTS pipeline.
    Uses the modern, compatible versions of all Google Cloud client libraries.
    """
    def __init__(self, target_language: str):
        self.target_language = target_language
        self.speech_client = speech.SpeechClient()
        self.translate_client = translate.TranslationServiceClient()
        self.tts_client = texttospeech.TextToSpeechClient()
        self.voice_map = {} # Cache for assigning unique voices to speakers

    def _get_speaker_voice(self, speaker_id: str) -> str:
        """Selects a unique, high-quality voice for a given speaker ID."""
        if speaker_id in self.voice_map:
            return self.voice_map[speaker_id]

        try:
            voices_response = self.tts_client.list_voices(language_code=self.target_language)
            wavenet_voices = [v for v in voices_response.voices if "Wavenet" in v.name]
            if not wavenet_voices:
                wavenet_voices = voices_response.voices # Fallback to any voice
            
            # Cycle through available voices based on the number of speakers we've already seen
            num_seen_speakers = len(self.voice_map)
            voice_name = wavenet_voices[num_seen_speakers % len(wavenet_voices)].name
            self.voice_map[speaker_id] = voice_name
            print(f"   » Assigning voice '{voice_name}' to {speaker_id}")
            return voice_name
        except Exception as e:
            print(f"   » Could not fetch TTS voices: {e}. Using a default.")
            return "en-US-Wavenet-D" # A fallback voice

    def translate_audio_chunk(self, audio_chunk_data: bytes, speaker_id: str):
        """
        Processes a single audio chunk through the full translation pipeline.
        Returns a dictionary or None if processing fails.
        """
        # 1. Speech-to-Text
        try:
            audio = speech.RecognitionAudio(content=audio_chunk_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=SOURCE_LANGUAGE,
            )
            
            # --- WRAPPED IN TRY/EXCEPT ---
            stt_response = self.speech_client.recognize(config=config, audio=audio)
            
        except Exception as e:
            print(f"   » [{speaker_id}] STT Error (Skipping chunk): {e}")
            return None
        # -----------------------------

        if not stt_response.results or not stt_response.results[0].alternatives:
            return None
        
        transcript = stt_response.results[0].alternatives[0].transcript
        print(f"   » [{speaker_id}] Transcription ({SOURCE_LANGUAGE}): {transcript}")
        
        if not transcript.strip():
            return None

        # 2. Translation 
        try:
            source_lang = SOURCE_LANGUAGE.split('-')[0]
            target_lang = self.target_language.split('-')[0]
            parent = f"projects/{GOOGLE_PROJECT_ID}/locations/global"

            translation_response = self.translate_client.translate_text(
                parent=parent,
                contents=[transcript],
                target_language_code=target_lang,
                source_language_code=source_lang,
            )
            translated_text = translation_response.translations[0].translated_text
            print(f"   » [{speaker_id}] Translation ({self.target_language}): {translated_text}")
        except Exception as e:
            print(f"   » [{speaker_id}] Translation Error: {e}")
            return None

        # 3. Text-to-Speech
        try:
            voice_name = self._get_speaker_voice(speaker_id)
            synthesis_input = texttospeech.SynthesisInput(text=translated_text)
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