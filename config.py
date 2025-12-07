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
    "tr": "tur",
    "pl": "pol",
    "nl": "nld",
    "sv": "swe",
    "da": "dan",
    "no": "nob",
    "fi": "fin",
    "cs": "ces",
}

# Default languages
DEFAULT_SOURCE = "eng"
DEFAULT_TARGET = "hin"

# --- Audio Configuration ---
SAMPLE_RATE = 16000  # Required by both Pyannote and SeamlessM4T

# --- Diarization Configuration ---
MIN_SPEAKERS = 2  # Minimum speakers (for sports commentary)
MAX_SPEAKERS = 4  # Maximum speakers
MIN_SEGMENT_DURATION_MS = 500  # Skip segments shorter than this
SPEAKER_MERGE_GAP_SECONDS = 0.3  # Merge segments from same speaker within this gap

# --- Model Configuration ---
SEAMLESS_MODEL = "facebook/seamless-m4t-v2-large"
DEVICE = "mps"  # M1 acceleration (use "cpu" for Intel Macs or "cuda" for NVIDIA)
