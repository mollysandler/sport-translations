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

# SPEAKER_VOICES = {
#     1: "es-ES-Wavenet-B",  # A male voice
#     2: "es-ES-Wavenet-C",  # A female voice
#     3: "es-ES-Wavenet-E",  # A male voice
#     4: "es-ES-Wavenet-F",  # A female voice
# }

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
        self.processed_word_count = 0

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

    def _file_streaming_thread(self, filepath):
        """Reads an audio file and streams it to the input queue."""
        print(f"📁 Streaming audio from file: {filepath}", flush=True)
        try:
            # Load the audio file using pydub
            audio = AudioSegment.from_file(filepath)

            # Ensure audio format matches the required settings
            audio = audio.set_frame_rate(RATE).set_channels(1).set_sample_width(2) # 16-bit

            # Get raw audio data
            raw_data = audio.raw_data

            # Stream data in chunks
            for i in range(0, len(raw_data), CHUNK):
                if self.shutdown_event.is_set():
                    break
                chunk = raw_data[i:i+CHUNK]
                self.audio_input_queue.put(chunk)
                # Simulate real-time streaming by waiting
                time.sleep(float(CHUNK) / (RATE * 2)) # sample_width is 2 bytes

            print("✅ Finished streaming file.", flush=True)
            self.audio_input_queue.put(None) # Signal the end of the stream

        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.", flush=True)
        except Exception as e:
            print(f"An error occurred while processing the audio file: {e}", flush=True)


    def _google_stt_thread(self):
        client = speech.SpeechClient()
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=4,
            max_speaker_count=4,
        )
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=SOURCE_LANGUAGE,
            diarization_config=diarization_config,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=False
        )

        def audio_generator():
            while not self.shutdown_event.is_set():
                chunk = self.audio_input_queue.get()
                if chunk is None: return
                yield chunk

        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator())
        
        print("🔊 Speech-to-Text is running...", flush=True)
        try:
            responses = client.streaming_recognize(streaming_config, requests)
            for response in responses:
                if self.shutdown_event.is_set(): break
                if not response.results: continue
                
                result = response.results[0]
                if not result.alternatives or not result.is_final: continue
                
                all_words = result.alternatives[0].words
                new_words = all_words[self.processed_word_count:]

                if not new_words: continue

                speaker_chunks = []
                current_speaker = new_words[0].speaker_tag
                current_chunk = []

                for word_info in new_words:
                    if word_info.speaker_tag == current_speaker:
                        current_chunk.append(word_info.word)
                    else:
                        speaker_chunks.append({'speaker': current_speaker, 'text': ' '.join(current_chunk)})
                        current_speaker = word_info.speaker_tag
                        current_chunk = [word_info.word]
                
                speaker_chunks.append({'speaker': current_speaker, 'text': ' '.join(current_chunk)})

                for chunk in speaker_chunks:
                    print(f"   » Speaker {chunk['speaker']} ({SOURCE_LANGUAGE}): {chunk['text']}", flush=True)
                    self.transcript_queue.put(chunk)

                self.processed_word_count = len(all_words)

        except Exception as e:
            if not self.shutdown_event.is_set():
                    print(f"An error occurred in the STT thread: {e}", flush=True)

    def _translation_thread(self):
        client = translate.Client()
        source_lang_short = SOURCE_LANGUAGE.split('-')[0]
        target_lang_short = self.target_language.split('-')[0]
        
        print("🌐 Translation service is running...", flush=True)
        while not self.shutdown_event.is_set():
            chunk_to_translate = self.transcript_queue.get()
            if chunk_to_translate is None: continue
            
            text_to_translate = chunk_to_translate['text']
            speaker = chunk_to_translate['speaker']

            result = client.translate(
                text_to_translate,
                target_language=target_lang_short,
                source_language=source_lang_short
            )
            translated_text = result["translatedText"]

            translated_chunk = {'speaker': speaker, 'text': translated_text}
            print(f"   » Translated for Speaker {speaker} ({self.target_language}): {translated_text}", flush=True)
            self.translation_queue.put(translated_chunk)

    def _google_tts_thread(self):
        client = texttospeech.TextToSpeechClient()
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        # --- NEW DYNAMIC VOICE SELECTION LOGIC ---
        print("🗣️ Selecting voices for language: {}".format(self.target_language), flush=True)
        
        # 1. Get all available voices for the target language
        try:
            voices_response = client.list_voices(language_code=self.target_language)
            voices = voices_response.voices
        except Exception as e:
            print(f"Error fetching voices: {e}. The TTS thread will not run.", flush=True)
            return

        # 2. Filter for high-quality WaveNet voices first
        wavenet_voices = [v for v in voices if "Wavenet" in v.name]
        
        # 3. Select one male and one female voice
        male_voice = None
        female_voice = None
        
        # Try to find a WaveNet male/female pair
        for voice in wavenet_voices:
            if voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE and not male_voice:
                male_voice = voice.name
            elif voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE and not female_voice:
                female_voice = voice.name
            if male_voice and female_voice:
                break

        # 4. Fallback logic if a perfect pair isn't found
        if not male_voice or not female_voice:
            print("   » Could not find a Male/Female WaveNet pair. Using fallback voices.", flush=True)
            # Try to find any voices if the WaveNet search failed
            all_male = [v.name for v in voices if v.ssml_gender == texttospeech.SsmlVoiceGender.MALE]
            all_female = [v.name for v in voices if v.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE]
            
            if not male_voice: male_voice = all_male[0] if all_male else None
            if not female_voice: female_voice = all_female[0] if all_female else None

            # Ultimate fallback: if one gender is missing, use another voice of the other gender
            if not male_voice and all_female and len(all_female) > 1:
                male_voice = all_female[1]
            elif not female_voice and all_male and len(all_male) > 1:
                female_voice = all_male[1]

        # Check if we have at least one voice
        if not male_voice and not female_voice:
            print(f"   » CRITICAL: No voices found for language '{self.target_language}'. TTS will not work.", flush=True)
            return # Stop the thread if no voices are available

        # If one is still missing, make them the same voice
        if not male_voice: male_voice = female_voice
        if not female_voice: female_voice = male_voice

        dynamic_speaker_voices = {1: male_voice, 2: female_voice}
        print(f"   » Speaker 1 (Male preference): {dynamic_speaker_voices[1]}", flush=True)
        print(f"   » Speaker 2 (Female preference): {dynamic_speaker_voices[2]}", flush=True)
        # --- END OF DYNAMIC VOICE SELECTION ---
        
        print("🗣️ Text-to-Speech service is running...", flush=True)
        while not self.shutdown_event.is_set():
            chunk_to_synthesize = self.translation_queue.get()
            if chunk_to_synthesize is None:
                break

            text_to_synthesize = chunk_to_synthesize['text']
            speaker = chunk_to_synthesize['speaker']

            # Use the dynamically selected voice
            voice_name = dynamic_speaker_voices.get(speaker, dynamic_speaker_voices[1])
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.target_language, name=voice_name
            )
            
            synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            self.audio_output_queue.put(response.audio_content)

        self.audio_output_queue.put(None)

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
    
    def start(self, filepath=None):
        """Starts all the translator threads."""
        self.shutdown_event.clear()

        # Determine which audio input thread to start
        if filepath:
            input_thread = threading.Thread(target=self._file_streaming_thread, args=(filepath,))
        else:
            input_thread = threading.Thread(target=self._microphone_thread)
        
        self.threads = [
            input_thread, # Use the selected input thread
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
        # Give threads a moment to finish processing queues
        for thread in self.threads:
            thread.join(timeout=1.0)
        print("Translator stopped.", flush=True)


if __name__ == "__main__":
    translator = None
    
    print("--- Real-Time Translator ---")
    print("Commands: 'start', 'stop', 'testfile', 'exit'")
    print("Press Ctrl+C at any time to exit gracefully.")
    
    try:
        while True:
            command = input("> ").lower().strip()
            
            if command == "start":
                if translator:
                    print("Translator is already running.", flush=True)
                else:
                    prompt = f"Enter target language code (e.g., hi-IN, fr-FR) [default: {TARGET_LANGUAGE}]: "
                    target_lang_input = input(prompt).strip() or TARGET_LANGUAGE
                    if target_lang_input == TARGET_LANGUAGE:
                        print(f"Using default target language: {target_lang_input}", flush=True)
                    print(f"Starting translator to '{target_lang_input}'...", flush=True)
                    translator = RealTimeTranslator(target_language=target_lang_input)
                    translator.start()
            
            elif command == "testfile":
                if translator:
                    print("Translator is already running. Please stop it first.", flush=True)
                else:
                    filepath = input("Enter the path to your audio file: ").strip()
                    if not os.path.exists(filepath):
                        print(f"File not found: {filepath}", flush=True)
                        continue

                    prompt = f"Enter target language code [default: {TARGET_LANGUAGE}]: "
                    target_lang_input = input(prompt).strip() or TARGET_LANGUAGE
                    if target_lang_input == TARGET_LANGUAGE:
                        print(f"Using default target language: {target_lang_input}", flush=True)
                    
                    print(f"Starting file test for '{filepath}'...", flush=True)
                    translator = RealTimeTranslator(target_language=target_lang_input)
                    translator.start(filepath=filepath)

                    # --- NEW: Wait for processing to finish automatically ---
                    print("Processing file... Press Ctrl+C to interrupt.", flush=True)
                    try:
                        # The playback thread is the last in the chain. We wait for it to finish.
                        playback_thread = translator.threads[-1]
                        playback_thread.join() # This will block until the thread is done
                        print("All processing is complete.", flush=True)
                    except KeyboardInterrupt:
                        print("\nUser interrupted file processing.", flush=True)
                    finally:
                        if translator:
                            translator.stop()
                            translator = None
                    # --------------------------------------------------------

            elif command == "stop":
                if translator:
                    translator.stop()
                    translator = None
                else:
                    print("Translator is not running.", flush=True)
            
            elif command == "exit":
                if translator:
                    translator.stop()
                print("Exiting application.", flush=True)
                break
                
            else:
                print(f"Unknown command: '{command}'", flush=True)

    except KeyboardInterrupt:
        if translator:
            translator.stop()
        print("\nExiting application due to user interrupt.", flush=True)