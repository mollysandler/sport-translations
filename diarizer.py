# diarizer.py
import os
import torch
from pyannote.audio import Pipeline
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from huggingface_hub import login

# Configuration defaults
MIN_SPEAKERS = 1
MAX_SPEAKERS = 20
MIN_SEGMENT_DURATION_MS = 300
SPEAKER_MERGE_GAP_SECONDS = 0.3

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
        print("   🔊 Loading speaker diarization pipeline...")

        if hf_token:
            # Login globally to resolve authentication for internal calls
            login(token=hf_token)

        try:
            # pyannote.audio 3.x expects use_auth_token
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        except TypeError:
            # some older variants used token=
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.pipeline.to(device)
        print(f"   ✅ Diarization pipeline loaded on {device}")
    
    def diarize(self, waveform: torch.Tensor, sample_rate: int = 16000) -> List[SpeakerSegment]:
        """
        FIXED: Now accepts waveform tensor directly from dynamic_voices.py
        This bypasses the 'torchcodec' and 'AudioDecoder' errors.
        """
        print(f"   🎙️ Analyzing speakers in waveform ({waveform.shape[1]/sample_rate:.1f}s)...")
    
        # Ensure waveform is mono and correct shape
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pass as dictionary to bypass broken file-loading logic on Mac
        audio_in_memory = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        # Run pipeline
        output = self.pipeline(
            audio_in_memory,
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS
        )
                
        # --- FIX FOR DIARIZE-OUTPUT WRAPPER ---
        if hasattr(output, "annotation"):
            diarization = output.annotation
        else:
            diarization = output
                
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                speaker_id=speaker,
                start_ms=int(turn.start * 1000),
                end_ms=int(turn.end * 1000),
                start_sec=float(turn.start),
                end_sec=float(turn.end)
            )
            segments.append(segment)
        
        print(f"   Found {len(segments)} raw segments")
        
        # Process segments (filtering, merging, etc.)
        segments = self._filter_short_segments(segments)
        segments = self._merge_close_segments(segments)
        segments = self._split_long_segments(segments, max_duration_sec=25.0)
        # segments = self._extend_short_segments(segments, min_duration_sec=1.0)
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        print(f"✅ Diarization complete: {len(unique_speakers)} speakers, {len(segments)} segments")
        return segments
    
    def _extend_short_segments(self, segments, min_duration_sec=1.0):
        extended = []
        for seg in segments:
            duration_sec = seg.duration_ms / 1000.0
            if duration_sec < min_duration_sec:
                padding_ms = int(min(((min_duration_sec - duration_sec) / 2) * 1000, 300))
                extended.append(SpeakerSegment(
                    speaker_id=seg.speaker_id,
                    start_ms=max(0, seg.start_ms - padding_ms),
                    end_ms=seg.end_ms + padding_ms,
                    start_sec=max(0, seg.start_sec - padding_ms/1000),
                    end_sec=seg.end_sec + padding_ms/1000
                ))
            else:
                extended.append(seg)
        return extended

    def _split_long_segments(self, segments, max_duration_sec=25.0):
        result = []
        for seg in segments:
            duration_sec = seg.duration_ms / 1000.0
            if duration_sec <= max_duration_sec:
                result.append(seg)
            else:
                num_chunks = int(np.ceil(duration_sec / max_duration_sec))
                chunk_dur = seg.duration_ms // num_chunks
                for i in range(num_chunks):
                    start = seg.start_ms + (i * chunk_dur)
                    end = min(seg.start_ms + ((i + 1) * chunk_dur), seg.end_ms)
                    result.append(SpeakerSegment(seg.speaker_id, start, end, start/1000, end/1000))
        return result

    def _filter_short_segments(self, segments):
        return [seg for seg in segments if seg.duration_ms >= MIN_SEGMENT_DURATION_MS]

    def _merge_close_segments(self, segments):
        if not segments: return segments
        merged = []
        current = segments[0]
        for next_seg in segments[1:]:
            if current.speaker_id == next_seg.speaker_id and (next_seg.start_sec - current.end_sec) <= SPEAKER_MERGE_GAP_SECONDS:
                current = SpeakerSegment(current.speaker_id, current.start_ms, next_seg.end_ms, current.start_sec, next_seg.end_sec)
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        return merged