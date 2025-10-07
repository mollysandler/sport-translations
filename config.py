
# --- LANGUAGE CONFIGURATION ---
# Set the source and target languages for the translation.
# A full list of supported language codes can be found here:
# Speech-to-Text: https://cloud.google.com/speech-to-text/docs/languages
# Text-to-Speech: https://cloud.google.com/text-to-speech/docs/voices
# Translation: https://cloud.google.com/translate/docs/languages

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "es-ES"  # Example: Spanish (Spain)


# --- AUDIO CONFIGURATION ---
# These settings are optimized for Google's streaming STT API.
# Do not change unless you know what you are doing.

# Sampling rate of the audio stream.
RATE = 16000

# Size of each audio chunk to process.
CHUNK = int(RATE / 10)  # 100ms