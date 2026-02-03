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
import io
import hashlib
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
    
    def __init__(self, use_voice_cloning: bool = False):
        self.use_voice_cloning = use_voice_cloning

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
        audio_np = audio_sample.squeeze().numpy()
        
        # 1. Estimate fundamental frequency (pitch)
        try:
            import librosa
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio_np, 
                sr=sample_rate,
                fmin=50,  # Male voices can go down to ~80Hz
                fmax=400  # Female voices typically up to ~350Hz
            )
            
            # Get average pitch (filter out zeros)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                pitch_range = np.std(pitch_values)
            else:
                # Fallback if pitch detection fails
                avg_pitch = 150  # Neutral default
                pitch_range = 20
                
        except ImportError:
            # If librosa not installed, use simple heuristics
            print(f"   ⚠️  librosa not installed, using basic voice detection")
            # Use energy in different frequency bands as proxy
            from scipy import signal
            
            # Simple frequency analysis
            freqs, psd = signal.welch(audio_np, sample_rate)
            
            # Energy in low (male) vs high (female) frequencies
            low_energy = np.sum(psd[(freqs > 80) & (freqs < 180)])
            high_energy = np.sum(psd[(freqs > 180) & (freqs < 300)])
            
            if low_energy > high_energy * 1.5:
                avg_pitch = 120  # Likely male
            else:
                avg_pitch = 220  # Likely female
            pitch_range = 20
        
        # 2. Determine gender based on pitch
        if avg_pitch < 160:
            gender = "male"
        else:
            gender = "female"
        
        # 3. Estimate speaking rate (simple)
        duration_sec = len(audio_np) / sample_rate
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


class QwenLocalVoiceCloner:
    """Optional local Qwen3-TTS voice cloning via `qwen-tts` package."""
    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device: str = "cpu"):
        self.available = False
        self.model = None
        self.model_id = model_id
        self.device = device

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
        target_lang: str = "es",
        buffer_duration_sec: int = 300,
        max_workers: int = 3,  # Reduced default to avoid ElevenLabs rate limits
        use_voice_cloning: bool = False
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.buffer_duration_sec = buffer_duration_sec
        self.sample_rate = 16000
        self.max_workers = max_workers
        self.use_voice_cloning = use_voice_cloning

        # Speaker -> cloned voice ids / prompts (persist for the full run)
        self.speaker_clone_voice_ids: Dict[str, str] = {}
        self.speaker_ref_wav: Dict[str, bytes] = {}
        self.speaker_ref_text: Dict[str, str] = {}
        self.speaker_qwen_prompt: Dict[str, object] = {}
        self.xtts_cloner = None
        
        # Initialize services
        self._initialize_services()
        
        # Voice manager
        self.voice_manager = SmartVoiceManager(use_voice_cloning)
        
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

        # 4. Optional Qwen local voice cloning (disabled by default on M1 unless you enable it)
        self.qwen_cloner = None
        if os.getenv("QWEN_TTS_ENABLE", "0") == "1":
            print("   🗣️  Loading Qwen3-TTS (local voice cloning)...")
            qwen_device = os.getenv("QWEN_TTS_DEVICE", "cpu")  # 'cpu' is safest on macOS
            qwen_model_id = os.getenv("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
            self.qwen_cloner = QwenLocalVoiceCloner(model_id=qwen_model_id, device=qwen_device)
            if self.qwen_cloner.available:
                print("   ✅ Qwen3-TTS ready (local)")
            else:
                self.qwen_cloner = None

        # 5. Optional XTTS local voice cloning (free backup clone)
        self.xtts_cloner = None
        if os.getenv("XTTS_ENABLE", "0") == "1":
            print("   🧬 Loading XTTS (free local voice cloning backup)...")
            xtts_device = os.getenv("XTTS_DEVICE", "cpu")
            xtts_model = os.getenv("XTTS_MODEL_ID", "tts_models/multilingual/multi-dataset/xtts_v2")
            self.xtts_cloner = XTTSLocalVoiceCloner(model_name=xtts_model, device=xtts_device)
            if self.xtts_cloner.available:
                print("   ✅ XTTS ready (local)")
            else:
                self.xtts_cloner = None

        print(f"✅ All services ready | Parallel workers: {self.max_workers}\n")
    
    def translate_video_streaming(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        hf_token: Optional[str] = None
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
        
        speaker_segments = self._get_speaker_segments(audio, audio_path, hf_token)
        
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
        buffer_duration = 0
        buffer_count = 0
        for seg in speaker_segments:
            if buffer_duration < self.buffer_duration_sec:
                buffer_count += 1
                buffer_duration += seg.end_sec - seg.start_sec
            else:
                break
        
        print(f"\n📊 Ready to process {len(speaker_segments)} segments")
        print(f"⏱️  Buffer: {buffer_count} segments (~{buffer_duration:.1f}s)\n")
        
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

                # Debug artifacts: captions + verification transcripts
                try:
                    self._export_debug_artifacts(audio_path, output_path, video_path)
                except Exception as e:
                    print(f"⚠️  Debug export failed (non-fatal): {e}")
        
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

        # Collect reference audio per speaker (used by both Qwen and XTTS)
        for speaker_id in unique_speakers:
            speaker_segments = [s for s in segments if s.speaker_id == speaker_id]

            collected = []
            total = 0.0
            target = float(os.getenv("CLONE_REF_SECONDS", "15"))

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

            # Qwen prompt cache (primary local clone, if enabled + language supported)
            if getattr(self, "qwen_cloner", None) and self.qwen_cloner.supports_lang(self.target_lang):
                prompt = self.qwen_cloner.create_prompt(ref_np, sample_rate, ref_text)
                if prompt is not None:
                    self.speaker_qwen_prompt[speaker_id] = prompt
                    print(f"   ✅ {speaker_id}: Qwen voice-clone prompt cached")

            # XTTS uses speaker_ref_wav bytes directly (no extra caching needed)
            if getattr(self, "xtts_cloner", None) and self.xtts_cloner.supports_lang(self.target_lang):
                # Just confirm availability in logs if we have ref audio
                print(f"   ✅ {speaker_id}: XTTS ref audio ready")

        # Absolute backup: assign ElevenLabs stock voices to EVERY speaker
        fallback_map = self.voice_manager.assign_voices(audio, segments, sample_rate)

        for speaker_id in unique_speakers:
            # Ensure we always have a stock voice id for last resort
            self.speaker_clone_voice_ids[speaker_id] = fallback_map[speaker_id]

            has_qwen = speaker_id in self.speaker_qwen_prompt
            has_xtts = (getattr(self, "xtts_cloner", None) is not None) and (speaker_id in self.speaker_ref_wav) and self.xtts_cloner.supports_lang(self.target_lang)

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
        self, audio: torch.Tensor, audio_path: str, hf_token: Optional[str]
    ) -> List[SpeakerSegment]:
        """Get speaker segments and remove overlaps"""
        total_duration = audio.shape[1] / self.sample_rate
        print(f"🎙️  Running diarization on full {total_duration:.1f}s...")
        
        if hf_token is None:
            hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        diarizer = SpeakerDiarizer(hf_token)
        raw_segments = diarizer.diarize(audio, self.sample_rate)

        segments = self._make_exclusive_turns(
            raw_segments,
            solo_keep_ms=250,
            ignore_interruptions_ms=900,
            min_turn_ms=350,
            merge_gap_ms=120,
        )

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

            chosen = max(active.items(), key=lambda kv: kv[1])[0]  # interrupter wins
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
            out.append(SpeakerSegment(spk, s, e, s / 1000.0, e / 1000.0))

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

    def _tts_bytes_to_wav(self, audio_bytes: bytes, output_sr: int = 24000) -> tuple[bytes, int]:
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

        Safe behaviors:
        - XTTS: chunk long text to avoid truncation (common ~239 char limit warning)
        - Concatenate chunk WAVs cleanly
        - Falls back to ElevenLabs stock voice if Qwen/XTTS fails
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
                vad_filter=False,
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
            # Helpers: chunking + wav concat
            # ----------------------------
            def _chunk_text_for_tts(text: str, max_chars: int) -> List[str]:
                """
                Chunk text to <= max_chars, preferring to split at punctuation/space.
                Keeps it simple and robust.
                """
                text = " ".join(text.split())
                if len(text) <= max_chars:
                    return [text]

                chunks: List[str] = []
                s = text
                preferred_breaks = [". ", "! ", "? ", "; ", ": ", ", ", " — ", " - "]

                while len(s) > max_chars:
                    window = s[: max_chars + 1]

                    cut = -1
                    # Try preferred punctuation boundaries first
                    for br in preferred_breaks:
                        pos = window.rfind(br)
                        if pos != -1 and pos >= int(max_chars * 0.55):
                            cut = pos + len(br)
                            break

                    # Fallback: last space
                    if cut == -1:
                        pos = window.rfind(" ")
                        if pos != -1 and pos >= int(max_chars * 0.55):
                            cut = pos + 1

                    # Hard fallback: forced cut
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
                """
                Concatenate multiple WAV byte blobs into one WAV (24kHz mono),
                inserting short silence between chunks.
                """
                if not wav_list:
                    raise ValueError("No wav chunks to concatenate")

                out = AudioSegment.silent(duration=0, frame_rate=24000)
                gap = AudioSegment.silent(duration=int(silence_ms), frame_rate=24000)

                for i, wb in enumerate(wav_list):
                    seg_audio = AudioSegment.from_wav(io.BytesIO(wb))
                    seg_audio = seg_audio.set_frame_rate(24000).set_channels(1)
                    # tiny fades to prevent clicks
                    seg_audio = seg_audio.fade_in(10).fade_out(10)
                    if i > 0 and silence_ms > 0:
                        out += gap
                    out += seg_audio

                bio = io.BytesIO()
                out.export(bio, format="wav")
                return bio.getvalue()

            # You can tune these via env vars:
            # - XTTS_MAX_CHARS (default 220)
            # - XTTS_CHUNK_SILENCE_MS (default 50)
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

            # 4A) Primary: Qwen local clone (if enabled + prompt exists)
            if self.use_voice_cloning and getattr(self, "qwen_cloner", None):
                prompt = self.speaker_qwen_prompt.get(seg.speaker_id)
                if prompt is not None and self.qwen_cloner.supports_lang(self.target_lang):
                    try:
                        qwen_bytes = self.qwen_cloner.synthesize(translated_text, self.target_lang, prompt)
                        if qwen_bytes:
                            wav_bytes, duration_ms = self._tts_bytes_to_wav(qwen_bytes, output_sr=24000)
                    except Exception as e:
                        print(f"   ⚠️  Qwen TTS failed for {seg.speaker_id} (segment {idx}): {e}")

            # 4B) Backup: XTTS local clone (if enabled + ref wav exists)
            if wav_bytes is None and self.use_voice_cloning and getattr(self, "xtts_cloner", None):
                ref_wav = self.speaker_ref_wav.get(seg.speaker_id)
                if ref_wav and self.xtts_cloner.supports_lang(self.target_lang):
                    try:
                        # Chunk long text to avoid truncation warnings/bugs
                        chunks = _chunk_text_for_tts(translated_text, xtts_max_chars)
                        xtts_wavs: List[bytes] = []

                        for c in chunks:
                            xtts_bytes = self.xtts_cloner.synthesize(c, self.target_lang, ref_wav)
                            if not xtts_bytes:
                                raise ValueError("XTTS returned empty bytes")
                            w, _dur = self._tts_bytes_to_wav(xtts_bytes, output_sr=24000)
                            xtts_wavs.append(w)

                        # Concatenate chunk audio
                        wav_bytes = _concat_wav_bytes(xtts_wavs, silence_ms=xtts_silence_ms)
                        wav_bytes, duration_ms = self._tts_bytes_to_wav(wav_bytes, output_sr=24000)

                    except Exception as e:
                        print(f"   ⚠️  XTTS failed for {seg.speaker_id} (segment {idx}): {e}")
                        wav_bytes = None
                        duration_ms = None

            # 4C) Absolute last resort: ElevenLabs stock voice
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
            MAX_PAUSE_MS = 500

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
                    pause_ms = max(0, min(orig_gap, MAX_PAUSE_MS))
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
        MAX_PAUSE_MS = 500  # tune 300–800

        if not segments:
            return AudioSegment.silent(duration=0, frame_rate=OUTPUT_SR)

        final_audio = AudioSegment.silent(duration=0, frame_rate=OUTPUT_SR)
        prev_orig_end = segments[0].start_ms  # removes leading silence

        for seg in segments:
            seg_audio = AudioSegment.from_wav(io.BytesIO(seg.audio_bytes))
            seg_audio = seg_audio.set_frame_rate(OUTPUT_SR).set_channels(1)

            orig_gap = seg.start_ms - prev_orig_end
            pause_ms = max(0, orig_gap)
            pause_ms = min(pause_ms, MAX_PAUSE_MS)

            if pause_ms:
                final_audio += AudioSegment.silent(duration=pause_ms, frame_rate=OUTPUT_SR)

            final_audio += seg_audio
            prev_orig_end = seg.end_ms

        return final_audio

    def _ms_to_srt_ts(self, ms: int) -> str:
        ms = max(0, int(ms))
        h = ms // 3600000
        ms -= h * 3600000
        m = ms // 60000
        ms -= m * 60000
        s = ms // 1000
        ms -= s * 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _ms_to_vtt_ts(self, ms: int) -> str:
        ms = max(0, int(ms))
        h = ms // 3600000
        ms -= h * 3600000
        m = ms // 60000
        ms -= m * 60000
        s = ms // 1000
        ms -= s * 1000
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _write_srt(self, items, path: str):
        # items: list[(start_ms, end_ms, text)]
        lines = []
        for i, (s, e, text) in enumerate(items, 1):
            lines.append(str(i))
            lines.append(f"{self._ms_to_srt_ts(s)} --> {self._ms_to_srt_ts(e)}")
            lines.append(text.strip())
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def _write_vtt(self, items, path: str):
        # items: list[(start_ms, end_ms, text)]
        lines = ["WEBVTT", ""]
        for (s, e, text) in items:
            lines.append(f"{self._ms_to_vtt_ts(s)} --> {self._ms_to_vtt_ts(e)}")
            lines.append(text.strip())
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def _build_output_timeline_captions(self, segments: List["TranslationSegment"]) -> List[tuple]:
        """
        Build captions aligned to the OUTPUT audio timeline.
        Must match the pause-capping logic in _compose_audio.
        Returns list of (out_start_ms, out_end_ms, caption_text).
        """
        OUTPUT_SR = 24000
        MAX_PAUSE_MS = 500  # MUST MATCH _compose_audio

        if not segments:
            return []

        segs = sorted(segments, key=lambda x: x.start_ms)

        out_items = []
        out_cursor = 0
        prev_orig_end = segs[0].start_ms  # removes leading silence (matches _compose_audio)

        for seg in segs:
            orig_gap = seg.start_ms - prev_orig_end
            pause_ms = max(0, orig_gap)
            pause_ms = min(pause_ms, MAX_PAUSE_MS)

            out_cursor += pause_ms
            out_start = out_cursor
            out_end = out_start + int(seg.duration_ms)

            # Include speaker label so you can visually confirm switching
            caption_text = f"[{seg.speaker_id}] {seg.translated_text}"

            out_items.append((out_start, out_end, caption_text))

            out_cursor = out_end
            prev_orig_end = seg.end_ms

        return out_items

    def _transcribe_full_audio_to_srt_and_txt(self, wav_path: str, language: str, out_prefix: str):
        """
        Full-file ASR transcript with timestamps -> SRT + TXT.
        Uses your existing whisper instance (small.en if src=en). Works best for source audio.
        """
        from faster_whisper import WhisperModel

        # Use a multilingual model if language isn't English.
        # This runs only in debug mode; it can be slow.
        model_name = "small.en" if language == "en" else "small"
        dbg_whisper = WhisperModel(model_name, device="cpu", compute_type="int8", num_workers=2)

        segs, _ = dbg_whisper.transcribe(
            wav_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        items = []
        full = []
        for s in segs:
            txt = (s.text or "").strip()
            if not txt:
                continue
            start_ms = int(s.start * 1000)
            end_ms = int(s.end * 1000)
            items.append((start_ms, end_ms, txt))
            full.append(txt)

        self._write_srt(items, f"{out_prefix}.srt")
        Path(f"{out_prefix}.txt").write_text(" ".join(full), encoding="utf-8")

    def _transcribe_output_audio_for_validation(self, wav_path: str, language: str, out_txt_path: str) -> str:
        """
        ASR the OUTPUT WAV so we can verify what was actually spoken.
        Returns recognized text.
        """
        from faster_whisper import WhisperModel

        dbg_whisper = WhisperModel("small", device="cpu", compute_type="int8", num_workers=2)
        segs, _ = dbg_whisper.transcribe(
            wav_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        parts = []
        for s in segs:
            txt = (s.text or "").strip()
            if txt:
                parts.append(txt)

        recognized = " ".join(parts)
        Path(out_txt_path).write_text(recognized, encoding="utf-8")
        return recognized

    def _simple_similarity(self, a: str, b: str) -> float:
        """
        Cheap similarity: token overlap F1-ish.
        Not perfect, but great for catching obvious missing speech.
        """
        import re
        ta = re.findall(r"\w+", (a or "").lower())
        tb = re.findall(r"\w+", (b or "").lower())
        if not ta or not tb:
            return 0.0
        sa, sb = set(ta), set(tb)
        inter = len(sa & sb)
        prec = inter / max(1, len(sa))
        rec = inter / max(1, len(sb))
        if (prec + rec) == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def _export_debug_artifacts(self, original_audio_wav: str, output_wav: str, video_path: str):
        """
        Writes:
        - Output captions aligned to output audio timeline: .srt + .vtt
        - Full source transcript (independent of diarization): .srt + .txt
        - Output ASR transcript + similarity score (to see if TTS spoke everything)
        """
        stem = Path(video_path).stem

        # 1) Output captions aligned to OUTPUT audio timeline
        with self.segments_lock:
            segs = list(self.all_segments)
        out_caps = self._build_output_timeline_captions(segs)
        out_srt = f"translated_{stem}_{self.target_lang}.srt"
        out_vtt = f"translated_{stem}_{self.target_lang}.vtt"
        self._write_srt(out_caps, out_srt)
        self._write_vtt(out_caps, out_vtt)
        print(f"✅ Wrote output captions: {out_srt}, {out_vtt}")

        # 2) Full transcript of ORIGINAL audio (independent check)
        # Enable with env so you can turn it off when latency matters.
        if os.getenv("EXPORT_SOURCE_TRANSCRIPT", "1") == "1":
            src_prefix = f"source_{stem}_{self.source_lang}"
            self._transcribe_full_audio_to_srt_and_txt(original_audio_wav, self.source_lang, src_prefix)
            print(f"✅ Wrote source transcript: {src_prefix}.srt, {src_prefix}.txt")

        # 3) ASR the OUTPUT audio and compare to intended translated_text
        if os.getenv("VALIDATE_OUTPUT_ASR", "1") == "1":
            expected = " ".join([s.translated_text for s in segs if s.translated_text])
            out_asr_path = f"output_asr_{stem}_{self.target_lang}.txt"
            recognized = self._transcribe_output_audio_for_validation(output_wav, self.target_lang, out_asr_path)
            sim = self._simple_similarity(expected, recognized)
            print(f"✅ Wrote output ASR transcript: {out_asr_path}")
            print(f"🔎 Output spoken-vs-expected similarity: {sim:.2f} (lower usually means missing speech or ASR errors)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python dynamic_voice_optimizer.py <video> [src] [tgt] [buffer] [workers]")
        print("\n✨ Supports UNLIMITED speakers with automatic voice matching!")
        print("\nExample: python dynamic_voice_optimizer.py video.mp4 en es 30 3")
        print("\nOptional: pip install librosa (for better pitch detection)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "en"
    target = sys.argv[3] if len(sys.argv) > 3 else "es"
    buffer = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    
    translator = DynamicSpeakerTranslator(
        source_lang=source,
        target_lang=target,
        buffer_duration_sec=buffer,
        max_workers=workers,
        use_voice_cloning=True
    )
    
    output = translator.translate_video_streaming(video_path)
    print(f"\n🎉 Complete! Output: {output}")


if __name__ == "__main__":
    main() 
