# dynamic_voice_optimizer.py
"""
Smart voice matching system that:
1. Handles unlimited number of speakers
2. Analyzes speaker characteristics (gender, pitch, age)
3. Matches to best ElevenLabs voice automatically
4. Falls back to voice cloning if needed
"""

import os
import sys
import threading
import queue
import time
import librosa
import io
import tempfile
from scipy.io.wavfile import write as write_wav
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from pydub import AudioSegment
from pydub.playback import play
import torchaudio
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from diarizer import SpeakerDiarizer, SpeakerSegment
from utils import estimate_pitch_yin, gender_from_pitch, TTSConfig, SpeakerMergeConfig

def _get_hf_token_from_env() -> Optional[str]:    
    return (
        os.getenv("HUGGING_FACE_TOKEN")
    )

@dataclass
class TranslationSegment:
    speaker_id: str
    start_ms: int
    end_ms: int
    original_text: str
    translated_text: str
    audio_bytes: bytes
    duration_ms: int


@dataclass
class VoiceProfile:
    """Analyzed characteristics of a speaker's voice"""
    speaker_id: str
    gender: str  # 'male' or 'female'
    avg_pitch: float  # Average fundamental frequency
    pitch_range: float  # Variation in pitch
    speaking_rate: float  # Words per second estimate
    voice_id: str  # Assigned ElevenLabs voice ID

class SmartVoiceManager:
    """
    Intelligent voice assignment system
    """
    
    def __init__(
        self,
        use_voice_cloning: bool = False,
        tts_config: Optional[TTSConfig] = None,
        speaker_merge: Optional[SpeakerMergeConfig] = None,
    ):

        # Configs (formerly env vars)
        self.tts_config = tts_config or TTSConfig()
        self.speaker_merge = speaker_merge or SpeakerMergeConfig()

        # Comprehensive ElevenLabs voice library with characteristics
        self.available_voices = {
            # Female voices
            "21m00Tcm4TlvDq8ikWAM": {  # Rachel
                "gender": "female",
                "pitch": "medium",
                "style": "warm_calm",
                "age": "young_adult"
            },
            "AZnzlk1XvdvUeBnXmlld": {  # Domi
                "gender": "female", 
                "pitch": "medium_low",
                "style": "strong_confident",
                "age": "adult"
            },
            "EXAVITQu4vr4xnSDxMaL": {  # Bella
                "gender": "female",
                "pitch": "medium_high",
                "style": "soft_friendly",
                "age": "young_adult"
            },
            "MF3mGyEYCl7XYWbV9V6O": {  # Elli
                "gender": "female",
                "pitch": "medium",
                "style": "emotional_expressive",
                "age": "young_adult"
            },
            # Male voices
            "ErXwobaYiN019PkySvjV": {  # Antoni
                "gender": "male",
                "pitch": "medium",
                "style": "well_rounded",
                "age": "adult"
            },
            "TxGEqnHWrfWFTfGW9XjX": {  # Josh
                "gender": "male",
                "pitch": "medium_low",
                "style": "deep_authoritative",
                "age": "adult"
            },
            "VR6AewLTigWG4xSOukaG": {  # Arnold
                "gender": "male",
                "pitch": "low",
                "style": "crisp_strong",
                "age": "mature"
            },
            "pNInz6obpgDQGcFmaJgB": {  # Adam
                "gender": "male",
                "pitch": "medium_low",
                "style": "deep_calm",
                "age": "adult"
            },
            "yoZ06aMxZJJ28mfd3POQ": {  # Sam
                "gender": "male",
                "pitch": "medium",
                "style": "energetic_young",
                "age": "young_adult"
            },
            "IKne3meq5aSn9XLyUdCD": {  # Charlie
                "gender": "male",
                "pitch": "medium_high",
                "style": "casual_conversational",
                "age": "young_adult"
            },
        }
        
        self.assigned_voices: Dict[str, VoiceProfile] = {}
        self.used_voice_ids = set()
    
    def analyze_speaker_characteristics(
        self, 
        audio_sample: torch.Tensor, 
        speaker_id: str,
        sample_rate: int = 16000
    ) -> VoiceProfile:
        """
        Analyze speaker's voice characteristics to match appropriate voice
        """        
        audio_np = audio_sample.squeeze().detach().cpu().numpy().astype(np.float32)

        # Optional: light high-pass to reduce rumble that confuses F0
        try:
            audio_np = librosa.effects.preemphasis(audio_np)
        except Exception:
            pass

        # 1) Pitch estimate (choose ONE)
        # Fast:
        avg_pitch = estimate_pitch_yin(audio_np, sample_rate)

        # Or more stable (slower): pYIN
        # avg_pitch = estimate_pitch_pyin(audio_np, sample_rate)

        if avg_pitch is None:
            avg_pitch = 150.0
            pitch_range = 20.0
        else:
            # compute a pitch range robustly too
            f0 = librosa.yin(audio_np, fmin=70, fmax=300, sr=sample_rate)
            f0 = f0[np.isfinite(f0)]
            pitch_range = float(np.std(f0)) if len(f0) else 20.0

        # 2) Gender (use your helper with an “unknown” band)
        gender = gender_from_pitch(avg_pitch, pitch_range)
        
        # 3. Estimate speaking rate (simple)
        # Rough estimate: detect speech segments
        energy = np.abs(audio_np)
        speech_ratio = np.sum(energy > 0.01) / len(energy)
        speaking_rate = speech_ratio * 2  # Rough words per second
        
        print(f"   🔍 {speaker_id} Analysis:")
        print(f"      Gender: {gender} | Pitch: {avg_pitch:.0f}Hz | Rate: {speaking_rate:.1f}w/s")
        
        # 4. Match to best available voice
        matched_voice_id = self._match_best_voice(gender, avg_pitch, pitch_range)
        
        return VoiceProfile(
            speaker_id=speaker_id,
            gender=gender,
            avg_pitch=avg_pitch,
            pitch_range=pitch_range,
            speaking_rate=speaking_rate,
            voice_id=matched_voice_id
        )
    
    def _match_best_voice(
        self, 
        gender: str, 
        avg_pitch: float, 
        pitch_range: float
    ) -> str:
        """
        Find the best matching voice from available voices
        """
        # Filter by gender first
        gender_matches = {
            vid: props for vid, props in self.available_voices.items()
            if props["gender"] == gender and vid not in self.used_voice_ids
        }
        
        # If no matches of same gender available, use opposite gender
        if not gender_matches:
            print(f"   ⚠️  No unused {gender} voices, using opposite gender")
            gender_matches = {
                vid: props for vid, props in self.available_voices.items()
                if vid not in self.used_voice_ids
            }
        
        # If ALL voices used, allow reuse but prefer unused
        if not gender_matches:
            print(f"   ⚠️  All voices used, reusing voices")
            gender_matches = {
                vid: props for vid, props in self.available_voices.items()
                if props["gender"] == gender
            }
        
        # Match pitch characteristics
        best_match = None
        best_score = -1
        
        for voice_id, props in gender_matches.items():
            score = 0
            
            # Pitch matching
            if avg_pitch < 140 and props["pitch"] == "low":
                score += 3
            elif 140 <= avg_pitch < 180 and props["pitch"] in ["medium_low", "medium"]:
                score += 3
            elif avg_pitch >= 180 and props["pitch"] in ["medium_high", "medium"]:
                score += 3
            
            # Style preferences for sports commentary
            if props["style"] in ["strong_confident", "deep_authoritative", "energetic_young"]:
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = voice_id
        
        # Default fallback
        if best_match is None:
            best_match = list(gender_matches.keys())[0]
        
        self.used_voice_ids.add(best_match)
        
        print(f"      ✅ Matched to: {best_match[:8]}... ({self.available_voices[best_match]['style']})")
        
        return best_match
    
    def assign_voices(
        self, 
        audio: torch.Tensor, 
        segments: List[SpeakerSegment],
        sample_rate: int = 16000
    ) -> Dict[str, str]:
        """
        Analyze all speakers and assign best matching voices
        Returns dict of {speaker_id: voice_id}
        """
        unique_speakers = sorted(set(seg.speaker_id for seg in segments))
        
        print(f"\n🎤 Analyzing {len(unique_speakers)} speakers for voice matching...")
        
        for speaker_id in unique_speakers:
            # Collect audio samples for this speaker
            speaker_segments = [s for s in segments if s.speaker_id == speaker_id]
            
            # Use first few segments (up to 10 seconds) for analysis
            collected_audio = []
            total_duration = 0
            target_duration = 10.0
            
            for seg in speaker_segments:
                if total_duration >= target_duration:
                    break
                
                start_sample = int(seg.start_sec * sample_rate)
                end_sample = int(seg.end_sec * sample_rate)
                seg_audio = audio[:, start_sample:end_sample]
                
                seg_duration = (end_sample - start_sample) / sample_rate
                if seg_duration >= 1.0:  # Only use segments longer than 1s
                    collected_audio.append(seg_audio)
                    total_duration += seg_duration
            
            if collected_audio:
                # Combine audio samples
                combined = torch.cat(collected_audio, dim=1)
                
                # Limit to target duration
                max_samples = int(target_duration * sample_rate)
                if combined.shape[1] > max_samples:
                    combined = combined[:, :max_samples]
                
                # Analyze and assign voice
                profile = self.analyze_speaker_characteristics(
                    combined, 
                    speaker_id, 
                    sample_rate
                )
                
                self.assigned_voices[speaker_id] = profile
            else:
                # Fallback: assign random voice
                print(f"   ⚠️  {speaker_id}: Insufficient audio, using default")
                fallback_voice = list(self.available_voices.keys())[
                    len(self.assigned_voices) % len(self.available_voices)
                ]
                self.assigned_voices[speaker_id] = VoiceProfile(
                    speaker_id=speaker_id,
                    gender="male",
                    avg_pitch=150,
                    pitch_range=20,
                    speaking_rate=2.0,
                    voice_id=fallback_voice
                )
        
        # Summary
        print(f"\n📊 Voice Assignment Summary:")
        for speaker_id, profile in sorted(self.assigned_voices.items()):
            voice_style = self.available_voices[profile.voice_id]['style']
            print(f"   {speaker_id}: {profile.gender} | {profile.avg_pitch:.0f}Hz | {voice_style}")
        
        return {sid: profile.voice_id for sid, profile in self.assigned_voices.items()}




# === Voice cloning backends ===
QWEN_LANG_NAME = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

def _wav_bytes_from_audio_np(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Convert float/np audio array to 16-bit PCM WAV bytes."""
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    if audio_np.dtype != np.int16:
        max_val = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if max_val > 0:
            audio_np = audio_np / max_val
        audio_i16 = (audio_np * 32767).astype(np.int16)
    else:
        audio_i16 = audio_np
    bio = io.BytesIO()
    write_wav(bio, sample_rate, audio_i16)
    bio.seek(0)
    return bio.read()

def _audio_np_from_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes into (float32 mono numpy, sample_rate)."""
    if not wav_bytes:
        return np.zeros(0, dtype=np.float32), 0
    with io.BytesIO(wav_bytes) as bio:
        wav, sr = torchaudio.load(bio)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    return wav.squeeze(0).cpu().numpy().astype(np.float32), int(sr)



class QwenLocalVoiceCloner:
    """Optional local Qwen3-TTS voice cloning via `qwen-tts` package."""
    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device: str = "cpu"):
        self.available = False
        self.model = None
        self.model_id = model_id
        self.device = device
        self._qwen_synth_sem = threading.Semaphore(1)

        try:
            from qwen_tts import Qwen3TTSModel
            # Qwen examples use device_map like 'cuda:0'. On macOS, 'cpu' is safest.
            device_map = device
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device_map,
            )
            self.available = True
        except Exception as e:
            print(f"   ⚠️  Qwen local voice cloning unavailable: {e}")
            self.available = False

    def supports_lang(self, target_lang_iso: str) -> bool:
        return target_lang_iso in QWEN_LANG_NAME

    def create_prompt(self, ref_audio_np: np.ndarray, ref_sr: int, ref_text: str):
        if not self.available:
            return None
        try:
            return self.model.create_voice_clone_prompt(
                ref_audio=(ref_audio_np, ref_sr),
                ref_text=ref_text,
                x_vector_only_mode=not bool(ref_text.strip()),
            )
        except Exception as e:
            print(f"   ⚠️  Qwen create prompt failed: {e}")
            return None

    def synthesize(self, text: str, target_lang_iso: str, prompt_items):
        if not self.available or prompt_items is None:
            return None
        try:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=QWEN_LANG_NAME[target_lang_iso],
                voice_clone_prompt=prompt_items,
            )
            return _wav_bytes_from_audio_np(np.array(wavs[0]), int(sr))
        except Exception as e:
            print(f"   ⚠️  Qwen synth failed: {e}")
            return None
        

class XTTSLocalVoiceCloner:
    """
    Free/local voice cloning using Coqui XTTS v2 (via `TTS` package).
    Uses speaker reference WAV + language to synthesize speech.
    """
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = "cpu"):
        self.available = False
        self.model_name = model_name
        self.device = device
        self.tts = None
        self.output_sr = 24000

        try:
            from TTS.api import TTS  # pip install TTS
            self.tts = TTS(model_name).to(device)
            # Best-effort: discover model sample rate
            try:
                self.output_sr = int(getattr(self.tts.synthesizer, "output_sample_rate", 24000))
            except Exception:
                self.output_sr = 24000
            self.available = True
        except Exception as e:
            print(f"   ⚠️  XTTS voice cloning unavailable: {e}")
            self.available = False

    def supports_lang(self, target_lang_iso: str) -> bool:
        # XTTS usually supports a set of languages; fall back to allowing common ones.
        try:
            langs = getattr(self.tts, "languages", None)
            if langs:
                return target_lang_iso in langs
        except Exception:
            pass
        return target_lang_iso in {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"}

    def synthesize(self, text: str, target_lang_iso: str, ref_wav_bytes: bytes) -> Optional[bytes]:
        if not self.available or not ref_wav_bytes:
            return None

        try:
            # XTTS expects a path to a speaker wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                f.write(ref_wav_bytes)
                f.flush()

                # Returns a waveform (list/np array)
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=f.name,
                    language=target_lang_iso
                )

            return _wav_bytes_from_audio_np(np.array(wav), int(self.output_sr))
        except Exception as e:
            print(f"   ⚠️  XTTS synth failed: {e}")
            return None

class DynamicSpeakerTranslator:
    """
    Translator with dynamic speaker support
    """
    
    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "en",
        buffer_duration_sec: int = 300,
        max_workers: int = 3,  # Reduced default to avoid ElevenLabs rate limits
        use_voice_cloning: bool = False,
        tts_config: Optional[TTSConfig] = None,
        speaker_merge: Optional[SpeakerMergeConfig] = None,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.buffer_duration_sec = buffer_duration_sec
        self.sample_rate = 16000
        self.max_workers = max_workers
        self.use_voice_cloning = use_voice_cloning
        self.tts_config = tts_config or TTSConfig()
        self.speaker_merge = speaker_merge or SpeakerMergeConfig()

        # Speaker -> cloned voice ids / prompts (persist for the full run)
        self.speaker_clone_voice_ids: Dict[str, str] = {}
        self.speaker_ref_wav: Dict[str, bytes] = {}
        self.speaker_ref_sec: Dict[str, float] = {}
        self.speaker_ref_text: Dict[str, str] = {}
        self.speaker_qwen_prompt: Dict[str, object] = {}
        self.xtts_cloner = None
        # Lazy-load guards for heavy TTS backends
        self._tts_load_lock = threading.Lock()
        self._qwen_load_attempted = False
        self._xtts_load_attempted = False
        self._qwen_prompt_lock = threading.Lock()

        
        # Initialize services
        self._initialize_services()
        
        # Voice manager
        self.voice_manager = SmartVoiceManager(
            use_voice_cloning,
            tts_config=self.tts_config,
            speaker_merge=self.speaker_merge,
        )

        # Threading
        self.playback_queue = queue.Queue()
        self.all_segments = []
        self.segments_lock = threading.Lock()
        self.processing_complete = threading.Event()
        self.error_occurred = threading.Event()
        self.error_message = None
        
        # Voice mapping
        self.speaker_voice_ids = {}
    
    def _initialize_services(self):
        """Initialize all required services"""
        print("🔧 Initializing optimized services...")
        
        # 1. Whisper
        print("   📝 Loading Faster-Whisper...")
        from faster_whisper import WhisperModel
        
        model_size = "small.en" if self.source_lang == "en" else "small"
        self.whisper = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=2
        )
        print(f"   ✅ Using {model_size} model")
        
        # 2. Translation
        print("   🌍 Loading translation...")
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
            self.use_local_translation = False
            print("   ✅ Google Translate (instant)")
        except ImportError:
            from transformers import MarianMTModel, MarianTokenizer
            model_map = {
                ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
                ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
                ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
            }
            model_name = model_map.get((self.source_lang, self.target_lang))
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)
            self.use_local_translation = True
            print("   ✅ Local translation model")
        
        # 3. ElevenLabs
        print("   🎤 Loading ElevenLabs...")
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY required")
        
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        
        self.elevenlabs = ElevenLabs(api_key=api_key)
        self.voice_settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True
        )
        print("   ✅ ElevenLabs ready")

        # 4/5. Local voice cloning backends (Qwen / XTTS)
        # These can take a long time to load, so we lazy-load them on first use.
        self.qwen_cloner = None
        self.xtts_cloner = None
        self._qwen_load_attempted = False
        self._xtts_load_attempted = False

        print(f"✅ All services ready | Parallel workers: {self.max_workers}\n")
    

    # ----------------------------
    # Lazy-loaded local TTS backends
    # ----------------------------
    def _ensure_qwen_loaded(self) -> bool:
        """Load Qwen local voice cloning backend on-demand."""
        if not self.tts_config.qwen_enable:
            return False
        if getattr(self, "qwen_cloner", None) is not None:
            return bool(self.qwen_cloner.available)
        with self._tts_load_lock:
            if getattr(self, "qwen_cloner", None) is not None:
                return bool(self.qwen_cloner.available)
            if getattr(self, "_qwen_load_attempted", False):
                return False
            self._qwen_load_attempted = True
            print("   🗣️  Lazy-loading Qwen3-TTS (local voice cloning)...")
            try:
                self.qwen_cloner = QwenLocalVoiceCloner(
                    model_id=self.tts_config.qwen_model_id,
                    device=self.tts_config.qwen_device,
                )
                if self.qwen_cloner.available:
                    print("   ✅ Qwen3-TTS ready (local)")
                    return True
            except Exception as e:
                print(f"   ⚠️  Qwen lazy-load failed: {e}")
            self.qwen_cloner = None
            return False

    def _ensure_xtts_loaded(self) -> bool:
        """Load XTTS local voice cloning backend on-demand."""
        if not self.tts_config.xtts_enable:
            return False
        if getattr(self, "xtts_cloner", None) is not None:
            return bool(self.xtts_cloner.available)
        with self._tts_load_lock:
            if getattr(self, "xtts_cloner", None) is not None:
                return bool(self.xtts_cloner.available)
            if getattr(self, "_xtts_load_attempted", False):
                return False
            self._xtts_load_attempted = True
            print("   🧬 Lazy-loading XTTS (local voice cloning backup)...")
            try:
                self.xtts_cloner = XTTSLocalVoiceCloner(
                    model_name=self.tts_config.xtts_model_id,
                    device=self.tts_config.xtts_device,
                )
                if self.xtts_cloner.available:
                    print("   ✅ XTTS ready (local)")
                    return True
            except Exception as e:
                print(f"   ⚠️  XTTS lazy-load failed: {e}")
            self.xtts_cloner = None
            return False

    def translate_video_streaming(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        ) -> str:
            """Main translation pipeline with dynamic speaker support"""
            print("="*70)
            print("⚡ DYNAMIC SPEAKER TRANSLATION")
            print("="*70)
            
            # Extract audio
            audio_path = self._extract_audio(video_path)
            audio, sr = torchaudio.load(audio_path)
            
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            total_duration_sec = audio.shape[1] / self.sample_rate
            print(f"📊 Total duration: {total_duration_sec:.1f}s")
            print(f"⏱️  Buffer: {self.buffer_duration_sec}s\n")
            
            # Get speaker segments
            print("="*70)
            print("🔍 PHASE 1: SPEAKER DISCOVERY & VOICE MATCHING")
            print("="*70)
            
            speaker_segments = self._get_speaker_segments(audio, audio_path)
            
            # Speaker -> voice mapping
            if self.use_voice_cloning:
                # Create true voice clones so each diarized speaker sounds like themselves.
                self.speaker_voice_ids = self._create_speaker_voice_clones(
                    audio, speaker_segments, self.sample_rate
                )
            else:
                # Legacy behavior: pick a pleasant stock voice per speaker.
                self.speaker_voice_ids = self.voice_manager.assign_voices(
                    audio, 
                    speaker_segments, 
                    self.sample_rate
                )
            
            # Calculate buffer
            buffer_count = 0
            if speaker_segments:
                t0 = speaker_segments[0].start_ms
                target_ms = int(self.buffer_duration_sec * 1000)

                for seg in speaker_segments:
                    buffer_count += 1
                    if (seg.end_ms - t0) >= target_ms:
                        break

            buffer_count = max(1, min(buffer_count, len(speaker_segments)))

            print(
                f"\n📊 Ready to process {len(speaker_segments)} segments\n"
                f"⏱️  Buffer: {buffer_count} segments "
                f"(~{(speaker_segments[buffer_count-1].end_ms - speaker_segments[0].start_ms)/1000:.1f}s timeline)\n"
            )

            # Process
            print("="*70)
            print(f"🚀 PHASE 2: PARALLEL STREAMING ({self.max_workers} workers)")
            print("="*70)
            
            processing_thread = threading.Thread(
                target=self._process_segments_parallel,
                args=(audio, speaker_segments),
                daemon=False
            )
            processing_thread.start()
            
            playback_thread = threading.Thread(
                target=self._playback_thread,
                args=(buffer_count,),
                daemon=False
            )
            playback_thread.start()
            
            processing_thread.join()
            playback_thread.join()
            
            if self.error_occurred.is_set():
                print(f"\n❌ Error: {self.error_message}")
                sys.exit(1)
            
            # Save
            print("\n" + "="*70)
            print("💾 SAVING FINAL OUTPUT")
            print("="*70)
            
            if output_path is None:
                video_stem = Path(video_path).stem
                output_path = f"translated_{video_stem}_{self.target_lang}.wav"
            
            with self.segments_lock:
                if self.all_segments:
                    print(f"📝 Composing {len(self.all_segments)} segments...")
                    final_audio = self._compose_audio(self.all_segments, total_duration_sec)
                    final_audio.export(output_path, format="wav")
                    print(f"✅ Saved to: {output_path}")

            print(f"🧪 Input duration:  {total_duration_sec*1000:.0f} ms")
            print(f"🧪 Output duration: {len(final_audio):.0f} ms")
            print(f"🧪 Delta: {(len(final_audio) - total_duration_sec*1000):.0f} ms")
            
            return output_path
        
    def _extract_audio(self, video_path: str) -> str:
            """Extract audio"""
            print(f"🎬 Extracting audio from '{Path(video_path).name}'...")
            try:
                # MoviePy v2+
                from moviepy import VideoFileClip
            except ImportError:
                # MoviePy v1
                from moviepy.editor import VideoFileClip

            
            temp_audio = "temp_extracted_audio.wav"
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(
                temp_audio, codec='pcm_s16le', fps=self.sample_rate,
                nbytes=2, ffmpeg_params=["-ac", "1"], logger=None
            )
            video.close()
            return temp_audio
        
    def _create_speaker_voice_clones(
            self,
            audio: torch.Tensor,
            segments: List[SpeakerSegment],
            sample_rate: int
        ) -> Dict[str, str]:
            """
            Prepare per-speaker local cloning materials (Qwen prompt, XTTS ref wav),
            and return per-speaker ElevenLabs STOCK voice_ids as absolute fallback.

            Returns: {speaker_id: elevenlabs_stock_voice_id}
            """
            unique_speakers = sorted(set(seg.speaker_id for seg in segments))
            print(f"\n🧬 Preparing local voice cloning for {len(unique_speakers)} speakers...")

            # Optional: embedding-based “clean ref” selection
            use_clean_ref = os.getenv("CLONE_REF_EMB_SELECT", "1") == "1"
            ref_min_score = float(os.getenv("CLONE_REF_MIN_SCORE", "0.03"))
            ref_min_ms = int(os.getenv("CLONE_REF_MIN_MS", "800"))
            ref_max_seg_sec = float(os.getenv("CLONE_REF_MAX_SEG_SEC", "3.0"))
            ref_debug = os.getenv("CLONE_REF_DEBUG", "0") == "1"
            save_ref = os.getenv("CLONE_REF_SAVE", "0") == "1"

            spkrec = None
            spkrec_device = torch.device("cpu")
            try:
                d = getattr(self, "_last_diarizer", None)
                if d is not None:
                    d._load_spkrec()
                    spkrec = d._spkrec
                    spkrec_device = d._spkrec_device
            except Exception:
                spkrec = None

            def _cos(a: np.ndarray, b: np.ndarray) -> float:
                a = a.astype(np.float32); b = b.astype(np.float32)
                na = float(np.linalg.norm(a) + 1e-9)
                nb = float(np.linalg.norm(b) + 1e-9)
                return float(np.dot(a, b) / (na * nb))

            def _embed_chunk(wav: torch.Tensor) -> Optional[np.ndarray]:
                if spkrec is None:
                    return None
                wav = wav.to(spkrec_device).float()
                with torch.no_grad():
                    emb = spkrec.encode_batch(wav).squeeze(0).squeeze(0).detach().cpu().numpy()
                return emb
            
            centroids = {}
            if use_clean_ref and spkrec is not None:
                by_spk = {sid: [s for s in segments if s.speaker_id == sid] for sid in unique_speakers}

                # Build centroid from the longest segments first (more stable)
                for sid, segs in by_spk.items():
                    segs_sorted = sorted(segs, key=lambda s: (s.duration_ms), reverse=True)
                    total = 0.0
                    chunks = []
                    for s in segs_sorted:
                        if total >= float(self.speaker_merge.merge_ref_sec):
                            break
                        if s.duration_ms < ref_min_ms:
                            continue
                        a = int(s.start_sec * sample_rate)
                        b = int(s.end_sec * sample_rate)
                        chunk = audio[:, a:b]
                        # cap per-segment chunk length to avoid boundary contamination
                        max_samples = int(ref_max_seg_sec * sample_rate)
                        if chunk.shape[1] > max_samples:
                            mid = chunk.shape[1] // 2
                            half = max_samples // 2
                            x0 = max(0, mid - half)
                            x1 = min(chunk.shape[1], x0 + max_samples)
                            chunk = chunk[:, x0:x1]

                        chunks.append(chunk)
                        total += chunk.shape[1] / sample_rate

                    if chunks:
                        ref = torch.cat(chunks, dim=1)
                        emb = _embed_chunk(ref)
                        if emb is not None:
                            centroids[sid] = emb

            # Collect reference audio per speaker (used by both Qwen and XTTS)
            for speaker_id in unique_speakers:
                speaker_segments = [s for s in segments if s.speaker_id == speaker_id]

                # Default fallback behavior (if embeddings aren’t available)
                selected = []

                collected = []
                total = 0.0
                target = float(os.getenv("CLONE_REF_SECONDS", str(self.tts_config.clone_ref_target_sec)))
                min_required = float(os.getenv("CLONE_MIN_SECONDS", str(self.tts_config.clone_ref_min_sec)))

                for seg in speaker_segments:
                    if total >= target:
                        break
                    start_sample = int(seg.start_sec * sample_rate)
                    end_sample = int(seg.end_sec * sample_rate)
                    seg_audio = audio[:, start_sample:end_sample]
                    seg_dur = (end_sample - start_sample) / sample_rate
                    if seg_dur >= 1.0:
                        collected.append(seg_audio)
                        total += seg_dur

                if not collected:
                    print(f"   ⚠️  {speaker_id}: insufficient audio for cloning; will rely on fallback voices.")
                    continue

                combined = torch.cat(collected, dim=1)
                max_samples = int(target * sample_rate)
                if combined.shape[1] > max_samples:
                    combined = combined[:, :max_samples]

                ref_np = combined.squeeze().cpu().numpy().astype(np.float32)
                ref_wav_bytes = _wav_bytes_from_audio_np(ref_np, sample_rate)
                self.speaker_ref_wav[speaker_id] = ref_wav_bytes
                self.speaker_ref_sec[speaker_id] = float(total)
                if float(total) < min_required:
                    print(f"   ⚠️  {speaker_id}: only {total:.1f}s ref audio (<{min_required:.1f}s). Skipping local cloning; will use ElevenLabs.")
                    continue

                # Reference transcript (helps Qwen prompt quality; optional)
                ref_text = ""
                try:
                    segs, _ = self.whisper.transcribe(
                        ref_np,
                        language=self.source_lang,
                        beam_size=1,
                        vad_filter=False,
                        condition_on_previous_text=False
                    )
                    parts = [s.text.strip() for s in segs if s.text and s.text.strip()]
                    ref_text = " ".join(parts)
                except Exception as e:
                    print(f"   ⚠️  {speaker_id}: reference transcription failed: {e}")
                self.speaker_ref_text[speaker_id] = ref_text
                # Local cloning backends are lazy-loaded during synthesis.

            # Absolute backup: assign ElevenLabs stock voices to EVERY speaker
            fallback_map = self.voice_manager.assign_voices(audio, segments, sample_rate)

            for speaker_id in unique_speakers:
                # Ensure we always have a stock voice id for last resort
                self.speaker_clone_voice_ids[speaker_id] = fallback_map[speaker_id]

                has_ref = (speaker_id in self.speaker_ref_wav) and (self.speaker_ref_sec.get(speaker_id, 0.0) >= float(os.getenv("CLONE_MIN_SECONDS", str(self.tts_config.clone_ref_min_sec))))
                has_qwen = bool(self.tts_config.qwen_enable) and has_ref
                has_xtts = bool(self.tts_config.xtts_enable) and has_ref

                if has_qwen:
                    print(f"   ✅ {speaker_id}: using Qwen voice clone (backup stock voice {fallback_map[speaker_id][:8]}...)")
                elif has_xtts:
                    print(f"   ✅ {speaker_id}: using XTTS voice clone (backup stock voice {fallback_map[speaker_id][:8]}...)")
                else:
                    print(f"   ✅ {speaker_id}: using fallback stock voice ({fallback_map[speaker_id][:8]}...)")

            return self.speaker_clone_voice_ids

    def _print_segments(self, label, segs):
            print(f"\n--- {label} ({len(segs)}) ---")
            for s in segs:
                print(f"{s.speaker_id:10s} {s.start_ms:6d}-{s.end_ms:6d}  ({(s.end_ms-s.start_ms)}ms)")            
                # overlap check
            overlaps = 0
            for a, b in zip(segs, segs[1:]):
                if a.end_ms > b.start_ms:                    
                    overlaps += 1
            print("overlaps:", overlaps)

    def _get_speaker_segments(
            self, audio: torch.Tensor, audio_path: str
        ) -> List[SpeakerSegment]:
            """Get speaker segments and remove overlaps"""
            total_duration = audio.shape[1] / self.sample_rate
            print(f"🎙️  Running diarization on full {total_duration:.1f}s...")
            
            hf_token = _get_hf_token_from_env()
            diarizer = SpeakerDiarizer(hf_token, merge_config=self.speaker_merge)
            self._last_diarizer = diarizer
            raw_segments = diarizer.diarize(audio, self.sample_rate)

            segments = self._make_exclusive_turns(
                raw_segments,
                solo_keep_ms=500,
                ignore_interruptions_ms=900,
                min_turn_ms=550,
                merge_gap_ms=120,
            )
            segments = self._merge_adjacent_turns(segments, max_gap_ms=400, min_keep_ms=250)
            # call it:
            self._print_segments("RAW", raw_segments)
            self._print_segments("EXCLUSIVE", segments)
            
            speaker_counts = {}
            for seg in segments:
                speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
            
            print(f"\n📊 Detected {len(speaker_counts)} unique speakers:")
            for speaker_id in sorted(speaker_counts.keys()):
                print(f"   {speaker_id}: {speaker_counts[speaker_id]} segments")
            
            return segments
        
    def _merge_adjacent_turns(
            self,
            segments: List[SpeakerSegment],
            max_gap_ms: int = 400,
            min_keep_ms: int = 250,
        ) -> List[SpeakerSegment]:
            """
            Merge adjacent segments from the same speaker if separated by <= max_gap_ms.
            Also drop extremely tiny turns (min_keep_ms).
            """
            if not segments:
                return []

            segs = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
            merged: List[SpeakerSegment] = []

            cur = segs[0]
            for nxt in segs[1:]:
                same = (nxt.speaker_id == cur.speaker_id)
                gap = nxt.start_ms - cur.end_ms

                if same and gap <= max_gap_ms:
                    # extend current
                    cur = SpeakerSegment(
                        cur.speaker_id,
                        cur.start_ms,
                        max(cur.end_ms, nxt.end_ms),
                        cur.start_ms / 1000.0,
                        max(cur.end_ms, nxt.end_ms) / 1000.0,
                    )
                else:
                    if (cur.end_ms - cur.start_ms) >= min_keep_ms:
                        merged.append(cur)
                    cur = nxt

            if (cur.end_ms - cur.start_ms) >= min_keep_ms:
                merged.append(cur)

            return merged

        
    def _make_exclusive_turns(
        self,
        segments: List[SpeakerSegment],
        solo_keep_ms: int = 250,         
        ignore_interruptions_ms: int = 900,
        min_turn_ms: int = 350,
        merge_gap_ms: int = 120,
    ) -> List[SpeakerSegment]:
            """
            Convert overlap-capable diarization into exclusive turns.

            Key policy for sports:
            - An "interruption" is only kept as a real turn if it has >= solo_keep_ms of SOLO speech.
            - Overlap-only spurts (never solo) are treated as crosstalk and usually absorbed.
            """
            if not segments:
                return []

            segs = sorted(segments, key=lambda s: (s.start_ms, s.end_ms, s.speaker_id))

            time_to_starts = {}
            time_to_ends = {}
            for seg in segs:
                time_to_starts.setdefault(seg.start_ms, []).append((seg.speaker_id, seg.start_ms))
                time_to_ends.setdefault(seg.end_ms, []).append(seg.speaker_id)

            times = sorted(set(time_to_starts.keys()) | set(time_to_ends.keys()))
            if len(times) < 2:
                return []

            active = {}  # speaker_id -> start_ms
            intervals = []  # (speaker_id, start, end, is_solo)

            for i, t in enumerate(times[:-1]):
                # end first (half-open [start, end))
                for spk in time_to_ends.get(t, []):
                    active.pop(spk, None)

                for spk, start_ms in time_to_starts.get(t, []):
                    active[spk] = start_ms

                t_next = times[i + 1]
                if t_next <= t or not active:
                    continue

                # chosen = max(active.items(), key=lambda kv: kv[1])[0]  # interrupter wins
                chosen = min(active.items(), key=lambda kv: kv[1])[0]  # floor-holder wins
                is_solo = (len(active) == 1)
                intervals.append((chosen, t, t_next, is_solo))

            if not intervals:
                return []

            # Merge adjacent intervals with same speaker, accumulating solo duration
            merged = []
            spk, s, e, is_solo = intervals[0]
            solo = (e - s) if is_solo else 0
            merged.append([spk, s, e, solo])  # [spk, start, end, solo_ms]

            for spk, s, e, is_solo in intervals[1:]:
                p_spk, p_s, p_e, p_solo = merged[-1]
                add_solo = (e - s) if is_solo else 0

                if spk == p_spk and s <= p_e + merge_gap_ms:
                    merged[-1] = [p_spk, p_s, max(p_e, e), p_solo + add_solo]
                else:
                    merged.append([spk, s, e, add_solo])

            # Debounce: absorb overlap-only (or near overlap-only) interruptions
            debounced = []
            for spk, s, e, solo_ms in merged:
                dur = e - s

                # If this segment does NOT have enough SOLO floor time,
                # treat it as crosstalk unless it's truly long.
                should_absorb = (solo_ms < solo_keep_ms) and (dur < ignore_interruptions_ms)

                if debounced and should_absorb:
                    p_spk, p_s, p_e, p_solo = debounced[-1]
                    debounced[-1] = [p_spk, p_s, e, p_solo]  # extend previous speaker
                else:
                    debounced.append([spk, s, e, solo_ms])

            # Merge again after debouncing (gaps)
            merged2 = [debounced[0]]
            for spk, s, e, solo_ms in debounced[1:]:
                p_spk, p_s, p_e, p_solo = merged2[-1]
                if spk == p_spk and s <= p_e + merge_gap_ms:
                    merged2[-1] = [p_spk, p_s, max(p_e, e), p_solo + solo_ms]
                else:
                    merged2.append([spk, s, e, solo_ms])

            # Filter tiny turns
            out = []
            for spk, s, e, _solo_ms in merged2:
                if (e - s) < min_turn_ms:
                    continue
                PAD_MS = int(os.getenv("TURN_PAD_MS", "120"))  # 80–200 works well
                out.append(SpeakerSegment(spk, s, e + PAD_MS, s/1000.0, (e + PAD_MS)/1000.0))

            out.sort(key=lambda x: x.start_ms)
            return out
        
    def _process_segments_parallel(self, audio: torch.Tensor, segments: List[SpeakerSegment]):
            """Process segments in parallel (skip-safe ordered emission)."""
            try:
                start_time = time.time()
                processed_count = 0

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_segment = {}
                    for idx, seg in enumerate(segments, 1):
                        future = executor.submit(self._process_single_segment, idx, seg, audio)
                        future_to_segment[future] = (idx, seg)

                    completed_segments: Dict[int, TranslationSegment] = {}
                    skipped = set()
                    next_to_emit = 1

                    for future in as_completed(future_to_segment):
                        idx, _seg = future_to_segment[future]
                        try:
                            result = future.result()

                            if not result:
                                skipped.add(idx)
                                while next_to_emit in skipped:
                                    next_to_emit += 1
                                continue

                            completed_segments[idx] = result
                            processed_count += 1

                            elapsed = time.time() - start_time
                            progress = (processed_count / len(segments)) * 100
                            print(f"⚡ Completed {processed_count}/{len(segments)} ({progress:.0f}%) | {elapsed:.1f}s")

                            # Emit contiguous ready segments in order, skipping failures
                            while True:
                                if next_to_emit in skipped:
                                    next_to_emit += 1
                                    continue
                                if next_to_emit in completed_segments:
                                    ready = completed_segments.pop(next_to_emit)
                                    self.playback_queue.put(ready)
                                    with self.segments_lock:
                                        self.all_segments.append(ready)
                                    next_to_emit += 1
                                    continue
                                break

                        except Exception as e:
                            print(f"   ⚠️  Segment {idx} failed: {e}")
                            skipped.add(idx)
                            while next_to_emit in skipped:
                                next_to_emit += 1

                self.processing_complete.set()
                total_time = time.time() - start_time
                speedup = (audio.shape[1] / self.sample_rate) / total_time if total_time > 0 else 0.0
                print(f"\n✅ Processing complete! {total_time:.1f}s ({speedup:.2f}x real-time)")

            except Exception as e:
                self.error_message = str(e)
                self.error_occurred.set()
                print(f"\n❌ Error: {e}")

    def _tts_bytes_to_wav(self, audio_bytes: bytes, output_sr: int = 24000) -> Tuple[bytes, int]:
            """
            Convert arbitrary TTS bytes (often MP3 from ElevenLabs) into normalized WAV bytes.
            Returns (wav_bytes, duration_ms).
            """
            if not audio_bytes:
                raise ValueError("TTS returned empty audio bytes")

            bio = io.BytesIO(audio_bytes)

            audio_segment = None
            # ElevenLabs commonly returns MP3. Try MP3 first, then fall back.
            for fmt in ("mp3", "wav", "m4a", "ogg"):
                try:
                    bio.seek(0)
                    audio_segment = AudioSegment.from_file(bio, format=fmt)
                    break
                except Exception:
                    continue

            if audio_segment is None:
                # Last resort: let ffmpeg auto-detect (can work if bytes have headers)
                bio.seek(0)
                audio_segment = AudioSegment.from_file(bio)

            audio_segment = audio_segment.set_frame_rate(output_sr).set_channels(1)
            duration_ms = len(audio_segment)

            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()

            # Validate WAV header
            if not wav_bytes.startswith(b"RIFF") or b"WAVE" not in wav_bytes[:32]:
                raise ValueError(f"Invalid WAV header; first 16 bytes: {wav_bytes[:16]!r}")

            return wav_bytes, duration_ms

    def _process_single_segment(
        self,
        idx: int,
        seg: SpeakerSegment,
        audio: torch.Tensor,
    ) -> Optional[TranslationSegment]:
        """
        Process one diarized segment: ASR -> translate -> TTS -> TranslationSegment.

        Fix (1): ONLY apply duration/trim logic AFTER wav_bytes + duration_ms exist.
        This prevents: "'>' not supported between instances of 'NoneType' and 'int'".
        """
        try:
            # ----------------------------
            # 1) Slice waveform for ASR
            # ----------------------------
            start_sample = int(seg.start_sec * self.sample_rate)
            end_sample = int(seg.end_sec * self.sample_rate)
            if end_sample <= start_sample:
                return None

            segment_audio = audio[:, start_sample:end_sample]
            audio_np = segment_audio.squeeze().numpy()

            # ----------------------------
            # 2) ASR (Faster-Whisper)
            # ----------------------------
            segments_whisper, _ = self.whisper.transcribe(
                audio_np,
                language=self.source_lang,
                beam_size=5,
                vad_filter=True,
                vad_parameters=Dict(min_silence_duration_ms=250),
                condition_on_previous_text=False,
            )

            text_parts = [s.text.strip() for s in segments_whisper if s.text and s.text.strip()]
            if not text_parts:
                print(f"   ⚠️  No ASR text for {seg.speaker_id} (segment {idx}, {seg.end_ms - seg.start_ms}ms) — skipping")
                return None

            original_text = " ".join(text_parts)

            # ----------------------------
            # 3) Translate
            # ----------------------------
            if not self.use_local_translation:
                translated_text = self.translator.translate(original_text)
            else:
                tokens = self.tokenizer([original_text], return_tensors="pt", padding=True)
                translated = self.translation_model.generate(**tokens)
                translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)

            # ----------------------------
            # Helpers
            # ----------------------------
            def _chunk_text_for_tts(text: str, max_chars: int) -> List[str]:
                text = " ".join(text.split())
                if len(text) <= max_chars:
                    return [text]

                chunks: List[str] = []
                s = text
                preferred_breaks = [". ", "! ", "? ", "; ", ": ", ", ", " — ", " - "]

                while len(s) > max_chars:
                    window = s[: max_chars + 1]
                    cut = -1

                    for br in preferred_breaks:
                        pos = window.rfind(br)
                        if pos != -1 and pos >= int(max_chars * 0.55):
                            cut = pos + len(br)
                            break

                    if cut == -1:
                        pos = window.rfind(" ")
                        if pos != -1 and pos >= int(max_chars * 0.55):
                            cut = pos + 1

                    if cut == -1:
                        cut = max_chars

                    chunk = s[:cut].strip()
                    if chunk:
                        chunks.append(chunk)
                    s = s[cut:].strip()

                if s:
                    chunks.append(s)

                return chunks

            def _concat_wav_bytes(wav_list: List[bytes], silence_ms: int = 50) -> bytes:
                if not wav_list:
                    raise ValueError("No wav chunks to concatenate")

                out = AudioSegment.silent(duration=0, frame_rate=24000)
                gap = AudioSegment.silent(duration=int(silence_ms), frame_rate=24000)

                for i, wb in enumerate(wav_list):
                    seg_audio = AudioSegment.from_wav(io.BytesIO(wb))
                    seg_audio = seg_audio.set_frame_rate(24000).set_channels(1)
                    seg_audio = seg_audio.fade_in(10).fade_out(10)
                    if i > 0 and silence_ms > 0:
                        out += gap
                    out += seg_audio

                bio = io.BytesIO()
                out.export(bio, format="wav")
                return bio.getvalue()

            def _fit_tts_to_original(wav_bytes: bytes, tts_ms: int, orig_ms: int) -> Tuple[bytes, int]:
                """
                Fit TTS audio into the diarized segment duration *without cutting words* when possible.

                Strategy:
                - If close enough: keep as-is
                - If too long: time-stretch slightly faster (preferred)
                - If too short: pad with silence at end
                - If stretch would be extreme: truncate as last resort (with fade)
                """
                # Safety
                if not wav_bytes or tts_ms is None or orig_ms is None or orig_ms <= 0:
                    return wav_bytes, tts_ms

                # Allow a little natural drift (sports cadence varies)
                # If within 5%, don't touch it.
                if abs(tts_ms - orig_ms) <= int(orig_ms * 0.05):
                    return wav_bytes, tts_ms

                # Hard bounds on how much we’ll time-stretch.
                # rate > 1.0 = faster (shorter); rate < 1.0 = slower (longer)
                MAX_RATE_UP = float(os.getenv("TTS_MAX_RATE_UP", "1.25"))   # up to 25% faster
                MAX_RATE_DOWN = float(os.getenv("TTS_MAX_RATE_DOWN", "0.90"))  # allow ~10% slower if too short
                FADE_MS = int(os.getenv("TTS_FADE_MS", "25"))

                # Decode WAV bytes to float32 numpy
                try:
                    y, sr = torchaudio.load(io.BytesIO(wav_bytes))  # y: [ch, time]
                    if y.shape[0] > 1:
                        y = torch.mean(y, dim=0, keepdim=True)
                    y = y.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    sr = int(sr)
                except Exception:
                    # If decode fails, fall back to truncate/pad using pydub
                    a = AudioSegment.from_wav(io.BytesIO(wav_bytes))
                    if len(a) > orig_ms:
                        a = a[:orig_ms].fade_out(FADE_MS)
                    elif len(a) < orig_ms:
                        a = a + AudioSegment.silent(duration=(orig_ms - len(a)), frame_rate=a.frame_rate)
                    bio = io.BytesIO()
                    a.export(bio, format="wav")
                    return bio.getvalue(), orig_ms

                # Avoid stretching tiny clips (can sound awful)
                if orig_ms < 400 or tts_ms < 400:
                    a = AudioSegment.from_wav(io.BytesIO(wav_bytes))
                    if len(a) > orig_ms:
                        a = a[:orig_ms].fade_out(FADE_MS)
                    elif len(a) < orig_ms:
                        a = a + AudioSegment.silent(duration=(orig_ms - len(a)), frame_rate=a.frame_rate)
                    bio = io.BytesIO()
                    a.export(bio, format="wav")
                    return bio.getvalue(), orig_ms

                # If TTS is longer, speed it up (time-stretch) to fit
                if tts_ms > orig_ms:
                    # We need duration shrink => rate > 1
                    desired_rate = tts_ms / orig_ms  # e.g., 1200/1000 = 1.2 (20% faster)
                    rate = min(desired_rate, MAX_RATE_UP)

                    # If even max speed-up still leaves it too long, we’ll stretch then truncate tail lightly.
                    try:
                        # librosa expects float32 [-1, 1] generally; your audio is already float32
                        y2 = librosa.effects.time_stretch(y, rate=rate)

                        # Convert back to WAV bytes at same SR
                        wav2 = _wav_bytes_from_audio_np(y2, sr)

                        # Now check final ms and trim/pad precisely with pydub (easy + consistent)
                        a = AudioSegment.from_wav(io.BytesIO(wav2)).set_frame_rate(24000).set_channels(1)
                        if len(a) > orig_ms:
                            # last resort micro-trim to exact slot
                            a = a[:orig_ms].fade_out(FADE_MS)
                        elif len(a) < orig_ms:
                            a = a + AudioSegment.silent(duration=(orig_ms - len(a)), frame_rate=a.frame_rate)

                        a = a.fade_in(min(FADE_MS, 10)).fade_out(FADE_MS)
                        bio = io.BytesIO()
                        a.export(bio, format="wav")
                        return bio.getvalue(), orig_ms

                    except Exception:
                        # fallback: truncate
                        a = AudioSegment.from_wav(io.BytesIO(wav_bytes))
                        a = a[:orig_ms].fade_out(FADE_MS)
                        bio = io.BytesIO()
                        a.export(bio, format="wav")
                        return bio.getvalue(), orig_ms

                # If TTS is shorter, pad (optionally slow down slightly, but padding is safer)
                else:
                    a = AudioSegment.from_wav(io.BytesIO(wav_bytes)).set_frame_rate(24000).set_channels(1)

                    # Optional: if it's moderately short, you *can* slow down a bit (rate < 1.0),
                    # but slowing down often hurts intelligibility more than padding.
                    # So we mostly pad.
                    if len(a) < orig_ms:
                        a = a + AudioSegment.silent(duration=(orig_ms - len(a)), frame_rate=a.frame_rate)

                    a = a.fade_in(min(FADE_MS, 10)).fade_out(FADE_MS)
                    bio = io.BytesIO()
                    a.export(bio, format="wav")
                    return bio.getvalue(), orig_ms


            # TTS chunking knobs
            lang_default_limits = {
                "es": 220,
                "en": 260,
                "fr": 220,
                "de": 220,
                "it": 220,
                "pt": 220,
            }
            xtts_max_chars = int(os.getenv("XTTS_MAX_CHARS", str(lang_default_limits.get(self.target_lang, 220))))
            xtts_silence_ms = int(os.getenv("XTTS_CHUNK_SILENCE_MS", "50"))

            # ----------------------------
            # 4) TTS (Qwen -> XTTS -> Eleven)
            # ----------------------------
            wav_bytes: Optional[bytes] = None
            duration_ms: Optional[int] = None
            orig_ms = int(seg.end_ms - seg.start_ms)

            # 4A) Qwen local clone
            if self.use_voice_cloning:
                ref_seconds = float(self.speaker_ref_sec.get(seg.speaker_id, 0.0))
                min_required = float(os.getenv("CLONE_MIN_SECONDS", str(self.tts_config.clone_ref_min_sec)))
                has_ref_audio = (seg.speaker_id in self.speaker_ref_wav) and (ref_seconds >= min_required)

                if has_ref_audio:
                    qwen_lang_ok = False
                    if self.tts_config.qwen_enable and self._ensure_qwen_loaded():
                        qwen_lang_ok = self.qwen_cloner.supports_lang(self.target_lang)

                    if qwen_lang_ok:
                        if seg.speaker_id not in self.speaker_qwen_prompt:
                            with self._qwen_prompt_lock:
                                if seg.speaker_id not in self.speaker_qwen_prompt:
                                    try:
                                        ref_wav = self.speaker_ref_wav.get(seg.speaker_id)
                                        ref_np, ref_sr = _audio_np_from_wav_bytes(ref_wav) if ref_wav else (None, None)

                                        prompt = None
                                        if ref_np is not None and len(ref_np) > 0:
                                            prompt = self.qwen_cloner.create_prompt(
                                                ref_np,
                                                int(ref_sr or self.sample_rate),
                                                self.speaker_ref_text.get(seg.speaker_id, "") or "",
                                            )

                                        if prompt is not None:
                                            self.speaker_qwen_prompt[seg.speaker_id] = prompt
                                            print(f"   ✅ {seg.speaker_id}: Qwen prompt created (lazy)")
                                    except Exception as e:
                                        print(f"   ⚠️  Qwen prompt creation failed for {seg.speaker_id}: {e}")

                        prompt = self.speaker_qwen_prompt.get(seg.speaker_id)
                        if prompt is not None:
                            try:
                                qwen_bytes = self.qwen_cloner.synthesize(translated_text, self.target_lang, prompt)
                                if qwen_bytes:
                                    wav_bytes, duration_ms = self._tts_bytes_to_wav(qwen_bytes, output_sr=24000)
                                    wav_bytes, duration_ms = _fit_tts_to_original(wav_bytes, duration_ms, orig_ms)
                            except Exception as e:
                                print(f"   ⚠️  Qwen TTS failed for {seg.speaker_id} (segment {idx}): {e}")

                    # 4B) XTTS backup clone
                    if wav_bytes is None and self.tts_config.xtts_enable:
                        if self._ensure_xtts_loaded() and self.xtts_cloner.supports_lang(self.target_lang):
                            try:
                                ref_wav = self.speaker_ref_wav.get(seg.speaker_id)
                                if not ref_wav:
                                    raise ValueError("missing speaker reference wav")

                                chunks = _chunk_text_for_tts(translated_text, xtts_max_chars)
                                xtts_wavs: List[bytes] = []
                                for c in chunks:
                                    xtts_bytes = self.xtts_cloner.synthesize(c, self.target_lang, ref_wav)
                                    if not xtts_bytes:
                                        raise ValueError("XTTS returned empty bytes")
                                    w, _dur = self._tts_bytes_to_wav(xtts_bytes, output_sr=24000)
                                    xtts_wavs.append(w)

                                wav_bytes = _concat_wav_bytes(xtts_wavs, silence_ms=xtts_silence_ms)
                                wav_bytes, duration_ms = self._tts_bytes_to_wav(wav_bytes, output_sr=24000)
                                wav_bytes, duration_ms = _fit_tts_to_original(wav_bytes, duration_ms, orig_ms)
                            except Exception as e:
                                print(f"   ⚠️  XTTS failed for {seg.speaker_id} (segment {idx}): {e}")
                                wav_bytes = None
                                duration_ms = None

            # 4C) ElevenLabs last resort
            if wav_bytes is None:
                voice_id = self.speaker_voice_ids.get(seg.speaker_id)
                if not voice_id:
                    print(f"   ⚠️  No voice_id for {seg.speaker_id} (segment {idx}) — skipping")
                    return None

                audio_generator = self.elevenlabs.text_to_speech.convert(
                    voice_id=voice_id,
                    text=translated_text,
                    model_id="eleven_multilingual_v2",
                    voice_settings=self.voice_settings,
                )
                eleven_bytes = b"".join(audio_generator)
                wav_bytes, duration_ms = self._tts_bytes_to_wav(eleven_bytes, output_sr=24000)
                wav_bytes, duration_ms = _fit_tts_to_original(wav_bytes, duration_ms, orig_ms)

            assert wav_bytes is not None and duration_ms is not None

            return TranslationSegment(
                speaker_id=seg.speaker_id,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                original_text=original_text,
                translated_text=translated_text,
                audio_bytes=wav_bytes,
                duration_ms=duration_ms,
            )

        except Exception as e:
            print(f"   ⚠️  Segment {idx} error: {e}")
            return None


    def _playback_thread(self, buffer_count: int):
            """Playback thread (matches concat + clipped-gaps compose policy)."""
            try:
                print(f"⏳ Waiting for buffer ({buffer_count} segments)...")

                while self.playback_queue.qsize() < buffer_count:
                    time.sleep(0.3)
                    if self.error_occurred.is_set():
                        return
                    if self.processing_complete.is_set() and self.playback_queue.qsize() > 0:
                        break

                print(f"\n📊 BUFFER READY! Starting playback...\n")
                print("=" * 70)

                count = 0
                prev_orig_end: Optional[int] = None
                MAX_PAUSE_MS = 5000

                while not self.processing_complete.is_set() or not self.playback_queue.empty():
                    try:
                        segment = self.playback_queue.get(timeout=2)
                        count += 1

                        audio_seg = AudioSegment.from_wav(io.BytesIO(segment.audio_bytes))
                        audio_seg = audio_seg.set_frame_rate(24000).set_channels(1)

                        # Remove leading silence by anchoring to first segment's start_ms
                        if prev_orig_end is None:
                            prev_orig_end = segment.start_ms

                        orig_gap = segment.start_ms - prev_orig_end
                        pause_ms = max(0, orig_gap)
                        if MAX_PAUSE_MS is not None:
                            pause_ms = min(pause_ms, MAX_PAUSE_MS)
                        if pause_ms > 0:
                            play(AudioSegment.silent(duration=int(pause_ms), frame_rate=24000))

                        print(f"📊 [{segment.speaker_id}] {segment.translated_text[:60]}...")
                        play(audio_seg)

                        prev_orig_end = segment.end_ms
                        self.playback_queue.task_done()

                    except queue.Empty:
                        if not self.processing_complete.is_set():
                            time.sleep(0.3)
                        else:
                            break

                print(f"\n✅ Playback complete! {count} segments")

            except Exception as e:
                self.error_message = str(e)
                self.error_occurred.set()
                print(f"\n❌ Playback error: {e}")

    def _compose_audio(self, segments, total_duration_sec):
        OUTPUT_SR = 24000

        if not segments:
            return AudioSegment.silent(duration=int(total_duration_sec * 1000), frame_rate=OUTPUT_SR)

        # Make a full-length silent bed
        total_ms = int(total_duration_sec * 1000)
        final_audio = AudioSegment.silent(duration=total_ms, frame_rate=OUTPUT_SR)

        for seg in segments:
            seg_audio = AudioSegment.from_wav(io.BytesIO(seg.audio_bytes))
            seg_audio = seg_audio.set_frame_rate(OUTPUT_SR).set_channels(1)

            # Optional: avoid clicks
            seg_audio = seg_audio.fade_in(10).fade_out(10)

            # Place at the ORIGINAL time
            pos = max(0, int(seg.start_ms))
            final_audio = final_audio.overlay(seg_audio, position=pos)

        return final_audio
    
    def translate_audio_file_no_playback(self, wav_path: str):
        """
        Server-friendly entrypoint:
        - no playback thread
        - returns (mp3_bytes, captions_list)
        """
        audio, sr = torchaudio.load(wav_path)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        total_duration_sec = audio.shape[1] / self.sample_rate

        speaker_segments = self._get_speaker_segments(audio, wav_path)

        if self.use_voice_cloning:
            self.speaker_voice_ids = self._create_speaker_voice_clones(
                audio, speaker_segments, self.sample_rate
            )
        else:
            self.speaker_voice_ids = self.voice_manager.assign_voices(
                audio, speaker_segments, self.sample_rate
            )

        # Process segments (parallel) but we won't play them.
        self._process_segments_parallel(audio, speaker_segments)

        # Drain queue if you want; we rely on self.all_segments for final assembly
        # while not self.playback_queue.empty():
        #     _ = self.playback_queue.get_nowait()
        #     self.playback_queue.task_done()

        with self.segments_lock:
            segments = List(self.all_segments)

        final_audio = self._compose_audio(segments, total_duration_sec)

        # Export MP3 bytes for the frontend
        mp3_io = io.BytesIO()
        final_audio.export(mp3_io, format="mp3")
        mp3_bytes = mp3_io.getvalue()

        # Captions shaped for CommentaryPlayer.jsx
        captions = [
            {
                "speaker": s.speaker_id,
                "startTime": s.start_ms / 1000.0,
                "endTime": s.end_ms / 1000.0,
                "original": s.original_text,
                "translated": s.translated_text,
            }
            for s in segments
        ]

        return mp3_bytes, captions



def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Dynamic speaker translation + voice rendering pipeline (sports commentary focused)."
    )
    p.add_argument("video", help="Input video path")
    p.add_argument("--src", "--source", dest="source_lang", default="en", help="Source language (default: en)")
    p.add_argument("--tgt", "--target", dest="target_lang", default="en", help="Target language (default: en)")
    p.add_argument("--buffer", dest="buffer_duration_sec", type=int, default=30, help="Buffer seconds before playback starts (default: 30)")
    p.add_argument("--workers", dest="max_workers", type=int, default=3, help="Parallel workers (default: 3)")

    # Voice cloning toggle (default ON to match current behavior)
    vc = p.add_mutually_exclusive_group()
    vc.add_argument("--use-voice-cloning", dest="use_voice_cloning", action="store_true", default=True, help="Enable voice cloning (default: enabled)")
    vc.add_argument("--no-voice-cloning", dest="use_voice_cloning", action="store_false", help="Disable voice cloning; use pleasant stock voices")

    # TTS / cloning backends
    p.add_argument("--tts-backend", default="qwen", help='TTS backend label (default: "qwen")')
    p.add_argument("--qwen-tts-enable", type=int, default=1, help="Enable Qwen local TTS/clone (default: 1)")
    p.add_argument("--qwen-tts-model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Qwen model id (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base)")
    p.add_argument("--qwen-tts-device", default="mps", help='Qwen device (default: "mps")')
    p.add_argument("--xtts-enable", type=int, default=1, help="Enable XTTS local TTS/clone (default: 1)")
    p.add_argument("--xtts-model-id", default="tts_models/multilingual/multi-dataset/xtts_v2", help="XTTS model id (default: tts_models/multilingual/multi-dataset/xtts_v2)")
    p.add_argument("--xtts-device", default="cpu", help='XTTS device (default: "cpu")')

    # Speaker merge tuning
    p.add_argument("--speaker-merge-enable", type=int, default=1, help="Enable speaker merge/consolidation (default: 1)")
    p.add_argument("--speaker-merge-sim", type=float, default=0.74, help="Cosine similarity threshold for merging speakers (default: 0.74)")
    p.add_argument("--speaker-tiny-total-ms", type=int, default=6000, help="Speakers with < this total ms get absorbed (default: 6000)")
    p.add_argument("--speaker-emb-min-chunk-ms", type=int, default=250, help="Minimum chunk length for speaker embeddings (default: 250)")
    p.add_argument("--speaker-merge-ref-sec", type=float, default=20.0, help="Reference audio seconds per speaker for embedding (default: 20)")

    return p


def main():
    args = _build_arg_parser().parse_args()

    tts_cfg = TTSConfig(
        tts_backend=args.tts_backend,
        qwen_enable=bool(args.qwen_tts_enable),
        qwen_model_id=args.qwen_tts_model_id,
        qwen_device=args.qwen_tts_device,
        xtts_enable=bool(args.xtts_enable),
        xtts_model_id=args.xtts_model_id,
        xtts_device=args.xtts_device,
    )

    speaker_cfg = SpeakerMergeConfig(
        merge_enable=bool(args.speaker_merge_enable),
        merge_sim=args.speaker_merge_sim,
        tiny_total_ms=args.speaker_tiny_total_ms,
        emb_min_chunk_ms=args.speaker_emb_min_chunk_ms,
        merge_ref_sec=float(args.speaker_merge_ref_sec),
    )

    translator = DynamicSpeakerTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        buffer_duration_sec=args.buffer_duration_sec,
        max_workers=args.max_workers,
        use_voice_cloning=args.use_voice_cloning,
        tts_config=tts_cfg,
        speaker_merge=speaker_cfg,
    )

    output = translator.translate_video_streaming(args.video)
    
    print(f"🎉 Complete! Output: {output}")


if __name__ == "__main__":
    main()