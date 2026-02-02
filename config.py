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

# --- Model Configuration ---
SEAMLESS_MODEL = "facebook/seamless-m4t-v2-large"
DEVICE = "mps"  # M1 acceleration (use "cpu" for Intel Macs or "cuda" for NVIDIA)
