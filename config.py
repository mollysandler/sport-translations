
# # --- LANGUAGE CONFIGURATION ---
# # Set the source and target languages for the translation.
# # A full list of supported language codes can be found here:
# # Speech-to-Text: https://cloud.google.com/speech-to-text/docs/languages
# # Text-to-Speech: https://cloud.google.com/text-to-speech/docs/voices
# # Translation: https://cloud.google.com/translate/docs/languages

# SOURCE_LANGUAGE = "en-US"
# TARGET_LANGUAGE = "es-ES"  # Example: Spanish (Spain)

# GOOGLE_PROJECT_ID = "expanded-curve-473800-b0"

# # --- AUDIO CONFIGURATION ---
# # These settings are optimized for Google's streaming STT API.
# # Do not change unless you know what you are doing.

# # # Sampling rate of the audio stream.
# # RATE = 16000

# # # Size of each audio chunk to process.
# # CHUNK = int(RATE / 10)  # 100ms

# # --- Audio Configuration ---
# # Sample rate required by Google Speech-to-Text. Do not change.
# STT_SAMPLE_RATE = 16000

# # Sample rate for the final mixed audio output. 24000 is a good standard for TTS.
# OUTPUT_SAMPLE_RATE = 24000

# # The size of audio chunks to process at a time.
# CHUNK = 4096

# # --- Source Separation Configuration ---
# # The Demucs model to use. 'htdemucs_6s' is a high-quality model for speech.
# # Other options: 'htdemucs', 'htdemucs_ft', 'mdx_extra'
# # This model will be downloaded automatically on the first run.
# DEMUCS_MODEL = "htdemucs_6s"

# config.py

# --- Language Configuration ---
# The language of the original sports commentary (e.g., "en-US", "es-ES")
SOURCE_LANGUAGE = "en-US"

# The default language to translate into (e.g., "es-ES", "fr-FR", "hi-IN")
TARGET_LANGUAGE = "es-ES"


# --- Audio Configuration ---
# Sample rate required by Google STT and Pyannote. Do not change.
STT_SAMPLE_RATE = 16000


# --- Google Cloud Configuration ---
# Find this on your Google Cloud Console dashboard.
# It is required for the new version of the Translate API.
GOOGLE_PROJECT_ID = "expanded-curve-473800-b0"