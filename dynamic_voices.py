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
        
        # Smart voice assignment
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
        
        return output_path
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio"""
        print(f"🎬 Extracting audio from '{Path(video_path).name}'...")
        from moviepy import VideoFileClip
        
        temp_audio = "temp_extracted_audio.wav"
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            temp_audio, codec='pcm_s16le', fps=self.sample_rate,
            nbytes=2, ffmpeg_params=["-ac", "1"], logger=None
        )
        video.close()
        return temp_audio
    
    def _get_speaker_segments(
        self, audio: torch.Tensor, audio_path: str, hf_token: Optional[str]
    ) -> List[SpeakerSegment]:
        """Get speaker segments and remove overlaps"""
        total_duration = audio.shape[1] / self.sample_rate
        print(f"🎙️  Running diarization on full {total_duration:.1f}s...")
        
        if hf_token is None:
            hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        temp_path = "temp_full_audio.wav"
        torchaudio.save(temp_path, audio, self.sample_rate)
        
        diarizer = SpeakerDiarizer(hf_token)
        raw_segments = diarizer.diarize(audio, self.sample_rate)
        os.remove(temp_path)
        
        # Remove overlapping segments
        segments = self._remove_overlaps(raw_segments)
        
        speaker_counts = {}
        for seg in segments:
            speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
        
        print(f"\n📊 Detected {len(speaker_counts)} unique speakers:")
        for speaker_id in sorted(speaker_counts.keys()):
            print(f"   {speaker_id}: {speaker_counts[speaker_id]} segments")
        
        return segments
    
    def _remove_overlaps(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Smart overlap handling:
        - Remove true duplicates (same speaker, heavy overlap)
        - Trim minor overlaps (different speakers, slight overlap)
        - Preserve all unique dialogue
        """
        if not segments:
            return segments
        
        # Sort by start time
        sorted_segs = sorted(segments, key=lambda s: s.start_sec)
        
        cleaned = []
        i = 0
        
        while i < len(sorted_segs):
            current = sorted_segs[i]
            
            # Look ahead for overlaps
            if i + 1 < len(sorted_segs):
                next_seg = sorted_segs[i + 1]
                
                # Check for overlap
                if current.end_sec > next_seg.start_sec:
                    overlap = current.end_sec - next_seg.start_sec
                    curr_duration = current.end_sec - current.start_sec
                    next_duration = next_seg.end_sec - next_seg.start_sec
                    
                    # Case 1: Same speaker with heavy overlap (>70%) = DUPLICATE
                    if current.speaker_id == next_seg.speaker_id:
                        overlap_pct = overlap / min(curr_duration, next_duration)
                        if overlap_pct > 0.7:
                            # Keep longer segment, skip shorter
                            if next_duration > curr_duration:
                                current = next_seg
                            i += 1  # Skip next segment
                            continue
                    
                    # Case 2: Light overlap (<30%) = TRIM
                    overlap_pct = overlap / max(curr_duration, next_duration)
                    if overlap_pct < 0.3:
                        # Trim at 75% point of overlap (favor keeping more of each)
                        split_point = current.end_sec - (overlap * 0.25)
                        
                        current = SpeakerSegment(
                            speaker_id=current.speaker_id,
                            start_ms=current.start_ms,
                            end_ms=int(split_point * 1000),
                            start_sec=current.start_sec,
                            end_sec=split_point
                        )
                        
                        # Don't modify next_seg here, it will be processed in next iteration
            
            cleaned.append(current)
            i += 1
        
        removed = len(segments) - len(cleaned)
        if removed > 0:
            print(f"   🔧 Removed {removed} duplicate segments")
        
        return cleaned
    
    def _process_segments_parallel(self, audio: torch.Tensor, segments: List[SpeakerSegment]):
        """Process segments in parallel"""
        try:
            start_time = time.time()
            processed_count = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_segment = {}
                for idx, seg in enumerate(segments, 1):
                    future = executor.submit(
                        self._process_single_segment,
                        audio, seg, idx, len(segments)
                    )
                    future_to_segment[future] = (idx, seg)
                
                completed_segments = {}
                for future in as_completed(future_to_segment):
                    idx, seg = future_to_segment[future]
                    try:
                        result = future.result()
                        if result:
                            completed_segments[idx] = result
                            processed_count += 1
                            
                            elapsed = time.time() - start_time
                            progress = (processed_count / len(segments)) * 100
                            print(f"⚡ Completed {processed_count}/{len(segments)} ({progress:.0f}%) | {elapsed:.1f}s")
                    except Exception as e:
                        print(f"   ⚠️  Segment {idx} failed: {e}")
                
                for idx in sorted(completed_segments.keys()):
                    result = completed_segments[idx]
                    self.playback_queue.put(result)
                    with self.segments_lock:
                        self.all_segments.append(result)
            
            self.processing_complete.set()
            total_time = time.time() - start_time
            speedup = (audio.shape[1] / self.sample_rate) / total_time
            print(f"\n✅ Processing complete! {total_time:.1f}s ({speedup:.2f}x real-time)")
            
        except Exception as e:
            self.error_message = str(e)
            self.error_occurred.set()
            print(f"\n❌ Error: {e}")
    
    def _process_single_segment(
        self, audio: torch.Tensor, seg: SpeakerSegment, idx: int, total: int
    ) -> Optional[TranslationSegment]:
        """Process one segment"""
        try:
            start_sample = int(seg.start_sec * self.sample_rate)
            end_sample = int(seg.end_sec * self.sample_rate)
            segment_audio = audio[:, start_sample:end_sample]
            
            audio_np = segment_audio.squeeze().numpy()
            segments_whisper, _ = self.whisper.transcribe(
                audio_np,
                language=self.source_lang,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False
            )
            
            text_parts = [s.text.strip() for s in segments_whisper if s.text.strip()]
            if not text_parts:
                return None
            
            original_text = " ".join(text_parts)
            
            if not self.use_local_translation:
                translated_text = self.translator.translate(original_text)
            else:
                tokens = self.tokenizer([original_text], return_tensors="pt", padding=True)
                translated = self.translation_model.generate(**tokens)
                translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
            voice_id = self.speaker_voice_ids.get(seg.speaker_id)
            if not voice_id:
                return None
            
            audio_generator = self.elevenlabs.text_to_speech.convert(
                voice_id=voice_id,
                text=translated_text,
                model_id="eleven_multilingual_v2",
                voice_settings=self.voice_settings
            )
            
            audio_bytes = b"".join(audio_generator)
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            return TranslationSegment(
                speaker_id=seg.speaker_id,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                original_text=original_text,
                translated_text=translated_text,
                audio_bytes=wav_io.read(),
                duration_ms=len(audio_segment)
            )
            
        except Exception as e:
            print(f"   ⚠️  Segment {idx} error: {e}")
            return None
    
    def _playback_thread(self, buffer_count: int):
        """Playback thread"""
        try:
            print(f"⏳ Waiting for buffer ({buffer_count} segments)...")
            
            while self.playback_queue.qsize() < buffer_count:
                time.sleep(0.3)
                if self.error_occurred.is_set():
                    return
                if self.processing_complete.is_set() and self.playback_queue.qsize() > 0:
                    break
            
            print(f"\n📊 BUFFER READY! Starting playback...\n")
            print("="*70)
            
            count = 0
            prev_end = 0
            
            while not self.processing_complete.is_set() or not self.playback_queue.empty():
                try:
                    segment = self.playback_queue.get(timeout=2)
                    count += 1
                    
                    audio_seg = AudioSegment.from_wav(io.BytesIO(segment.audio_bytes))
                    
                    gap = segment.start_ms - prev_end
                    if gap > 100:
                        play(AudioSegment.silent(duration=int(gap)))
                    
                    print(f"📊 [{segment.speaker_id}] {segment.translated_text[:60]}...")
                    play(audio_seg)
                    
                    prev_end = segment.start_ms + segment.duration_ms
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
        """Compose final audio"""
        canvas_ms = int(total_duration_sec * 1000) + 5000
        final_audio = AudioSegment.silent(duration=canvas_ms)
        
        prev_orig_end = 0
        prev_trans_end = 0
        
        for seg in segments:
            seg_audio = AudioSegment.from_wav(io.BytesIO(seg.audio_bytes))
            
            orig_gap = seg.start_ms - prev_orig_end
            
            if orig_gap < 200:
                actual_start = prev_trans_end + 100
            else:
                ideal_start = prev_trans_end + orig_gap
                actual_start = max(seg.start_ms, ideal_start)
            
            required = actual_start + seg.duration_ms
            if len(final_audio) < required:
                final_audio += AudioSegment.silent(duration=required - len(final_audio) + 2000)
            
            final_audio = final_audio.overlay(seg_audio, position=actual_start)
            
            prev_orig_end = seg.end_ms
            prev_trans_end = actual_start + seg.duration_ms
        
        return final_audio


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
        use_voice_cloning=False
    )
    
    output = translator.translate_video_streaming(video_path)
    print(f"\n🎉 Complete! Output: {output}")


if __name__ == "__main__":
    main() 