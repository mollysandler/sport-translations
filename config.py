
# # --- Language Configuration ---
# # Set the source and target languages for the translation.
# # A full list of supported language codes can be found here:
# # Speech-to-Text: https://cloud.google.com/speech-to-text/docs/languages
# # Text-to-Speech: https://cloud.google.com/text-to-speech/docs/voices
# # Translation: https://cloud.google.com/translate/docs/languages

# The language of the original sports commentary (e.g., "en-US", "es-ES")
SOURCE_LANGUAGE = "en-US"

# The default language to translate into (e.g., "es-ES", "fr-FR", "hi-IN")
TARGET_LANGUAGE = "hi-IN"


# --- Audio Configuration ---
# Sample rate required by Google STT and Pyannote. Do not change.
STT_SAMPLE_RATE = 16000


# --- Google Cloud Configuration ---
# Find this on your Google Cloud Console dashboard.
# It is required for the new version of the Translate API.
GOOGLE_PROJECT_ID = "expanded-curve-473800-b0"