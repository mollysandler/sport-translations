# translator.py
import io
import torch
import whisper
import numpy as np
import nltk
from scipy.io.wavfile import write as write_wav
from google.cloud import translate_v2 as translate
from bark import SAMPLE_RATE as BARK_SAMPLE_RATE, generate_audio, preload_models
from typing import List, Dict, Any

# Ensure nltk is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- PYTORCH PATCH (Keep this from before) ---
_original_torch_load = torch.load
def _safe_legacy_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_legacy_load
# ---------------------------------------------

class AccurateTranslator:
    def __init__(self, source_language: str, target_language: str):
        self.source_language = source_language
        self.target_language = target_language
        
        print("   📝 Loading Whisper (Medium.en)...")
        model_name = "medium.en" if source_language == "en" else "medium"
        self.whisper_model = whisper.load_model(model_name)
        
        print("   🌐 Initializing Google Cloud Translate...")
        self.translate_client = translate.Client()
        
        print("   🔊 Loading Bark...")
        preload_models()
        
        # Consistent voice map
        self.speaker_voices = {}
        self.voice_presets = [
            "v2/en_speaker_6", # Male Professional
            "v2/en_speaker_9", # Female Professional
            "v2/en_speaker_3", # Male Casual
            "v2/en_speaker_1", # Female Casual
        ]

    def transcribe_full_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes full audio and returns segments with timestamps.
        Does NOT chop audio.
        """

        result = self.whisper_model.transcribe(
            audio_path,
            language=self.source_language,
            word_timestamps=False,
            condition_on_previous_text=False,
            
            # 1. Beam Search (Accuracy)
            beam_size=5,
            best_of=5,
            patience=2.0,
            temperature=0.0,
            
            # 2. Anti-Hallucination & Clean-up (NEW)
            # This prompt forces the model to ignore non-verbal sounds
            initial_prompt="The following is a clean transcript of a conversation. Do not include filler words like um, uh, ah. Do not transcribe background noise. The speakers talk over each other, so ensure you are catching ALL spoken words even if there are two speakers at once.",
            
            # 3. Confidence Thresholds
            # If Whisper isn't sure what was said (like a mumbled "um"), ignore it.
            logprob_threshold=-1.0,         
            no_speech_threshold=0.6         
        )

        return result['segments']
    
    def translate_text(self, text: str) -> str:
        if self.source_language == self.target_language:
            return text
        
        result = self.translate_client.translate(
            text,
            source_language=self.source_language,
            target_language=self.target_language
        )
        return result['translatedText']

    def synthesize_speech(self, text: str, speaker_id: str) -> bytes:
        """Generate audio for a specific text and speaker."""
        # Get consistent voice
        if speaker_id not in self.speaker_voices:
            idx = len(self.speaker_voices) % len(self.voice_presets)
            self.speaker_voices[speaker_id] = self.voice_presets[idx]
            print(f"      🎤 Assigned {self.speaker_voices[speaker_id]} to {speaker_id}")
        
        voice_preset = self.speaker_voices[speaker_id]
        
        try:
            # Generate audio
            audio_array = generate_audio(
                text,
                history_prompt=voice_preset,
                silent=True
            )
            
            # Convert to int16 WAV bytes
            audio_int16 = (audio_array * 32767).astype(np.int16)
            byte_io = io.BytesIO()
            write_wav(byte_io, BARK_SAMPLE_RATE, audio_int16)
            byte_io.seek(0)
            return byte_io.read()
        except Exception as e:
            print(f"      ❌ TTS Error: {e}")
            return None