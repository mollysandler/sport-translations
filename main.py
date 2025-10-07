import queue
import threading
import io
import os
import time

import sounddevice as sd
import numpy as np

from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from google.cloud import speech, texttospeech, translate_v2 as translate

from config import SOURCE_LANGUAGE, TARGET_LANGUAGE, RATE, CHUNK

load_dotenv()

class RealTimeTranslator:
    """
    A class that handles the real-time translation process in background threads.
    It can be started and stopped gracefully.
    """
    def __init__(self, target_language):
        self.target_language = target_language  # Store the target language for this session
        self.audio_input_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.threads = []

    def _microphone_thread(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(status, flush=True)
            self.audio_input_queue.put(bytes(indata))

        print("🎤 Microphone is now live. Start speaking!", flush=True)
        try:
            with sd.RawInputStream(samplerate=RATE,
                                   blocksize=CHUNK,
                                   dtype='int16',
                                   channels=1,
                                   callback=audio_callback):
                self.shutdown_event.wait()
        except Exception as e:
            print(f"An error occurred in the microphone thread: {e}", flush=True)

    def _google_stt_thread(self):
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=SOURCE_LANGUAGE,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=False
        )

        def audio_generator():
            while not self.shutdown_event.is_set():
                chunk = self.audio_input_queue.get()
                if chunk is None: return
                yield chunk

        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator()
        )
        
        print("🔊 Speech-to-Text is running...", flush=True)
        try:
            responses = client.streaming_recognize(streaming_config, requests)
            for response in responses:
                if self.shutdown_event.is_set(): break
                if not response.results: continue
                
                result = response.results[0]
                if not result.alternatives or not result.is_final: continue
                
                transcript = result.alternatives[0].transcript.strip()
                if transcript:
                    print(f"   » Original ({SOURCE_LANGUAGE}): {transcript}", flush=True)
                    self.transcript_queue.put(transcript)
        except Exception as e:
            if not self.shutdown_event.is_set():
                 print(f"An error occurred in the STT thread: {e}", flush=True)

    def _translation_thread(self):
        client = translate.Client()
        source_lang_short = SOURCE_LANGUAGE.split('-')[0]
        target_lang_short = self.target_language.split('-')[0]
        
        print("🌐 Translation service is running...", flush=True)
        while not self.shutdown_event.is_set():
            text_to_translate = self.transcript_queue.get()
            if text_to_translate is None: continue
            
            result = client.translate(
                text_to_translate,
                target_language=target_lang_short,
                source_language=source_lang_short
            )
            translated_text = result["translatedText"]
            print(f"   » Translated ({self.target_language}): {translated_text}", flush=True)
            self.translation_queue.put(translated_text)

    def _google_tts_thread(self):
        client = texttospeech.TextToSpeechClient()
        voice = texttospeech.VoiceSelectionParams(language_code=self.target_language)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        print("🗣️ Text-to-Speech service is running...", flush=True)
        while not self.shutdown_event.is_set():
            text_to_synthesize = self.translation_queue.get()
            if text_to_synthesize is None: continue

            synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            self.audio_output_queue.put(response.audio_content)

    def _playback_thread(self):
        print("🎧 Playback service is running...", flush=True)
        while not self.shutdown_event.is_set():
            audio_data = self.audio_output_queue.get()
            if audio_data is None: continue
            
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                play(audio_segment)
            except Exception as e:
                print(f"Error during playback: {e}", flush=True)
    
    def start(self):
        """Starts all the translator threads."""
        self.shutdown_event.clear()
        self.threads = [
            threading.Thread(target=self._microphone_thread),
            threading.Thread(target=self._google_stt_thread),
            threading.Thread(target=self._translation_thread),
            threading.Thread(target=self._google_tts_thread),
            threading.Thread(target=self._playback_thread),
        ]
        for thread in self.threads:
            thread.daemon = True
            thread.start()

    def stop(self):
        """Gracefully stops all translator threads."""
        print("\nStopping translator...", flush=True)
        self.shutdown_event.set()
        sd.stop()
        self.audio_input_queue.put(None)
        self.transcript_queue.put(None)
        self.translation_queue.put(None)
        self.audio_output_queue.put(None)
        for thread in self.threads:
            thread.join()
        print("Translator stopped.", flush=True)


if __name__ == "__main__":
    translator = None
    
    print("--- Real-Time Translator ---")
    print("Commands: 'start', 'stop', 'exit'")
    
    while True:
        command = input("> ").lower().strip()
        
        if command == "start":
            if translator:
                print("Translator is already running.", flush=True)
            else:
                prompt = f"Enter target language code (e.g., hi-IN, fr-FR) [default: {TARGET_LANGUAGE}]: "
                target_lang_input = input(prompt).strip()
                
                # If the user presses enter with no input, use the default from config.py
                if not target_lang_input:
                    target_lang_input = TARGET_LANGUAGE
                    print(f"Using default target language: {target_lang_input}", flush=True)

                print(f"Starting translator to '{target_lang_input}'...", flush=True)
                translator = RealTimeTranslator(target_language=target_lang_input)
                translator.start()

        elif command == "stop":
            if translator:
                translator.stop()
                translator = None
            else:
                print("Translator is not running.", flush=True)
        
        elif command == "exit":
            if translator:
                translator.stop()
                translator = None
            print("Exiting application.", flush=True)
            break
            
        else:
            print(f"Unknown command: '{command}'", flush=True)