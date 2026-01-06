# hybrid_config.py
"""
Configuration for hybrid translation system
"""

# --- Language Configuration ---
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi", 
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian"
}

DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "hi"

# --- Translation Model Mapping ---
# Maps (source, target) pairs to Helsinki-NLP models
TRANSLATION_MODELS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-roa",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    # Add more as needed
}

# --- Hybrid Pipeline Configuration ---

# Phase 1: Speaker Discovery
ANALYSIS_DURATION_SEC = 45  # Analyze first N seconds to find speakers
MIN_VOICE_PROFILE_DURATION = 5.0  # Minimum seconds needed per speaker
TARGET_VOICE_PROFILE_DURATION = 10.0  # Target duration for voice profiles

# Phase 2: Streaming Translation
CHUNK_DURATION_SEC = 2.0  # Process audio in N-second chunks
VAD_SILENCE_THRESHOLD = 0.01  # Energy threshold for silence detection

# Phase 3: Audio Composition
RELATIVE_TIMING = True  # Preserve timing gaps from original
MAX_DRIFT_MS = 3000  # Maximum allowed drift from original timing

# --- Model Configuration ---

# Whisper
WHISPER_MODEL = "medium.en"  # Use .en for English-only (faster)
WHISPER_COMPUTE_TYPE = "int8"  # int8 quantization for speed

# Speaker Recognition
SPEAKER_SIMILARITY_THRESHOLD = 0.50  # Minimum similarity to match speaker
SPEAKER_ENCODER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# TTS
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_SAMPLE_RATE = 24000

# Diarization (Phase 1 only)
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
MIN_SPEAKERS = 1
MAX_SPEAKERS = 5  # Adjust based on your use case

# --- Audio Configuration ---
SAMPLE_RATE = 16000  # Standard for speech processing

# --- Performance Tuning ---
USE_GPU = False  # Set to True if you have CUDA
DEVICE = "cuda" if USE_GPU else "cpu"

# Transcription settings
WHISPER_BEAM_SIZE = 1  # Lower = faster (use 5 for better accuracy)
WHISPER_VAD_FILTER = True  # Skip non-speech segments

# --- Output Configuration ---
OUTPUT_FORMAT = "wav"
OUTPUT_SAMPLE_RATE = 24000

# --- Environment Variables (optional) ---
# These can be set in .env file
ENV_VARS = {
    "HUGGING_FACE_TOKEN": None,  # Required for pyannote
    "GOOGLE_APPLICATION_CREDENTIALS": None,  # Optional for Google Translate
}