# diarizer.py
import os
import torch
from pyannote.audio import Pipeline
from typing import List, Dict
from dataclasses import dataclass

# Fix PyTorch 2.8 - add ALL pyannote safe globals at once
import torch.serialization
from pyannote.audio.core.task import Specifications, Problem, Resolution
torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    Specifications,
    Problem,
    Resolution,
])

from config import (
    MIN_SPEAKERS,
    MAX_SPEAKERS,
    MIN_SEGMENT_DURATION_MS,
    SPEAKER_MERGE_GAP_SECONDS
)

@dataclass
class SpeakerSegment:
    speaker_id: str
    start_ms: int
    end_ms: int
    start_sec: float
    end_sec: float
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

class SpeakerDiarizer:
    def __init__(self, hf_token: str):
        print("🔊 Loading speaker diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.pipeline.to(device)
        print(f"✅ Diarization pipeline loaded on {device}")
    
    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        print(f"🎙️ Analyzing speakers in audio...")
    
        # Load audio using torchaudio (bypass pyannote's torchcodec requirement)
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Pass as dictionary (pyannote's workaround format)
        audio_in_memory = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        diarization = self.pipeline(
        audio_in_memory,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS
    )
        segments = []
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                speaker_id=speaker,
                start_ms=int(turn.start * 1000),
                end_ms=int(turn.end * 1000),
                start_sec=turn.start,
                end_sec=turn.end
            )
            segments.append(segment)
        
        print(f"   Found {len(segments)} raw segments")
        segments = self._filter_short_segments(segments)
        segments = self._merge_close_segments(segments)
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        print(f"✅ Diarization complete: {len(unique_speakers)} speakers, {len(segments)} segments")
        return segments
    
    def _filter_short_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        filtered = [seg for seg in segments if seg.duration_ms >= MIN_SEGMENT_DURATION_MS]
        removed = len(segments) - len(filtered)
        if removed > 0:
            print(f"   🧹 Filtered out {removed} short segments")
        return filtered
    
    def _merge_close_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            gap = next_seg.start_sec - current.end_sec
            if current.speaker_id == next_seg.speaker_id and gap <= SPEAKER_MERGE_GAP_SECONDS:
                current = SpeakerSegment(
                    speaker_id=current.speaker_id,
                    start_ms=current.start_ms,
                    end_ms=next_seg.end_ms,
                    start_sec=current.start_sec,
                    end_sec=next_seg.end_sec
                )
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        
        merged_count = len(segments) - len(merged)
        if merged_count > 0:
            print(f"   🔗 Merged {merged_count} segments")
        return merged