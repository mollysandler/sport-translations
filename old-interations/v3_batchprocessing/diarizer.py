# diarizer.py
import os
import torch
from pyannote.audio import Pipeline
from typing import List, Dict
from dataclasses import dataclass
import numpy as np

# Fix PyTorch 2.8 - add ALL pyannote safe globals
import torch.serialization
from pyannote.audio.core.task import Specifications, Problem, Resolution
torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    Specifications,
    Problem,
    Resolution,
])

from hybrid_config import (
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
    
        # Load audio using torchaudio
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
        
        # Pass as dictionary
        audio_in_memory = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        # IMPROVED: Add clustering parameters for better accuracy
        diarization = self.pipeline(
            audio_in_memory,
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS,
            # These help with speaker separation:
            segmentation_onset=0.5,  # Lower = more sensitive to speaker changes
            segmentation_offset=0.5,
            clustering="AgglomerativeClustering",  # More accurate than default
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
        
        # Process segments in order
        segments = self._filter_short_segments(segments)
        segments = self._merge_close_segments(segments)
        
        # For Whisper pipeline, we don't need to split long segments
        # Whisper handles long audio well (up to 30s)
        # Only split if truly necessary (>25s)
        segments = self._split_long_segments(segments, max_duration_sec=25.0)
        
        # Single extension pass with reasonable threshold
        segments = self._extend_short_segments(segments, min_duration_sec=1.0)
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        print(f"✅ Diarization complete: {len(unique_speakers)} speakers, {len(segments)} segments")
        return segments
    
    def _extend_short_segments(self, segments: List[SpeakerSegment], min_duration_sec: float = 1.0) -> List[SpeakerSegment]:
        """Extend very short segments for better processing."""
        extended = []
        for seg in segments:
            duration_sec = seg.duration_ms / 1000.0
            
            if duration_sec < min_duration_sec:
                padding_needed = (min_duration_sec - duration_sec) / 2
                padding_ms = int(min(padding_needed * 1000, 300))  # Max 300ms padding
                
                new_seg = SpeakerSegment(
                    speaker_id=seg.speaker_id,
                    start_ms=max(0, seg.start_ms - padding_ms),
                    end_ms=seg.end_ms + padding_ms,
                    start_sec=max(0, seg.start_sec - padding_ms/1000),
                    end_sec=seg.end_sec + padding_ms/1000
                )
                print(f"   📏 Extended {seg.speaker_id} from {duration_sec:.2f}s to {new_seg.duration_ms/1000:.2f}s")
                extended.append(new_seg)
            else:
                extended.append(seg)
        
        return extended
    
    def _split_long_segments(self, segments: List[SpeakerSegment], max_duration_sec: float = 25.0) -> List[SpeakerSegment]:
        """Split segments longer than max_duration."""
        result = []
        
        for seg in segments:
            duration_sec = seg.duration_ms / 1000.0
            
            if duration_sec <= max_duration_sec:
                result.append(seg)
            else:
                num_chunks = int(np.ceil(duration_sec / max_duration_sec))
                chunk_duration_ms = seg.duration_ms // num_chunks
                
                print(f"   ✂️ Splitting {seg.speaker_id} ({duration_sec:.2f}s) into {num_chunks} chunks")
                
                for i in range(num_chunks):
                    chunk_start_ms = seg.start_ms + (i * chunk_duration_ms)
                    chunk_end_ms = min(seg.start_ms + ((i + 1) * chunk_duration_ms), seg.end_ms)
                    
                    chunk = SpeakerSegment(
                        speaker_id=seg.speaker_id,
                        start_ms=chunk_start_ms,
                        end_ms=chunk_end_ms,
                        start_sec=chunk_start_ms / 1000.0,
                        end_sec=chunk_end_ms / 1000.0
                    )
                    result.append(chunk)
        
        return result
    
    def _filter_short_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        filtered = [seg for seg in segments if seg.duration_ms >= MIN_SEGMENT_DURATION_MS]
        removed = len(segments) - len(filtered)
        if removed > 0:
            print(f"   🧹 Filtered out {removed} segments (< {MIN_SEGMENT_DURATION_MS}ms)")
        return filtered
    
    def _merge_close_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Merge segments from same speaker that are very close together."""
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            gap = next_seg.start_sec - current.end_sec
            
            # Only merge if same speaker AND gap is small
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
            print(f"   🔗 Merged {merged_count} close segments")
        return merged