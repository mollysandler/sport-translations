# Configuration for SeamlessM4T-based translation system

# --- Language Mapping ---
# Maps frontend short codes to SeamlessM4T language codes
# Full list: https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md
SEAMLESS_LANGUAGE_MAPPING = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ja": "jpn",
    "ko": "kor",
    "zh": "cmn",
    "ar": "arb",
    "ru": "rus",
}

# Default languages
DEFAULT_SOURCE = "eng"
DEFAULT_TARGET = "hin"

# --- Audio Configuration ---
SAMPLE_RATE = 16000  # Required by both Pyannote and SeamlessM4T

# --- Diarization Configuration ---
MIN_SPEAKERS = 1
MAX_SPEAKERS = 3  # Good default for interviews + commentary
# We can be sensitive again because this ONLY affects speaker labeling,
# not text accuracy.
MIN_SEGMENT_DURATION_MS = 300  # Catch "Why not?"
SPEAKER_MERGE_GAP_SECONDS = 0.3 # Reasonable balance

# --- Model Configuration ---
SEAMLESS_MODEL = "facebook/seamless-m4t-v2-large"
DEVICE = "mps"  # M1 acceleration (use "cpu" for Intel Macs or "cuda" for NVIDIA)
