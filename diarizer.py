# diarizer.py
import os
import torch
from pyannote.audio import Pipeline
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import librosa
from huggingface_hub import login
from collections import defaultdict
from utils import SpeakerMergeConfig

# Configuration defaults
MIN_SPEAKERS = 1
MAX_SPEAKERS = 20
MIN_SEGMENT_DURATION_MS = 300
SPEAKER_MERGE_GAP_SECONDS = 0.3

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / (na * nb))

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
    def __init__(self, hf_token: str, merge_config: SpeakerMergeConfig | None = None):
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
        # Optional: speaker embedding model for post-merge consolidation
        self._spkrec = None
        self._spkrec_device = torch.device("cpu")

        # Speaker merge config (formerly env vars)
        self.merge_config = merge_config or SpeakerMergeConfig()

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
        segments = self._consolidate_speakers(waveform, sample_rate, segments)
        
        unique_speakers = set(seg.speaker_id for seg in segments)
        print(f"✅ Diarization complete: {len(unique_speakers)} speakers, {len(segments)} segments")
        return segments
    
    def _load_spkrec(self):
            """
            Lazy-load SpeechBrain speaker embedding model.
            """
            if self._spkrec is not None:
                return
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except Exception as e:
                print("   ⚠️  speechbrain not installed; skip speaker merge embeddings")
                return

            # Keep on CPU to avoid MPS headaches
            self._spkrec_device = torch.device("cpu")
            self._spkrec = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self._spkrec_device)},
            )
            print("   ✅ Speaker embedding model ready")

    def _filter_short_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        filtered = [seg for seg in segments if seg.duration_ms >= MIN_SEGMENT_DURATION_MS]
        if len(filtered) < len(segments):
            print(f"   Filtered out {len(segments)-len(filtered)} short segments")
        return filtered
    
    def _merge_close_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        if not segments:
            return []
        
        segments = sorted(segments, key=lambda x: x.start_ms)
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            gap_sec = (seg.start_ms - last.end_ms) / 1000.0
            
            if seg.speaker_id == last.speaker_id and gap_sec <= SPEAKER_MERGE_GAP_SECONDS:
                # Merge segments
                merged[-1] = SpeakerSegment(
                    speaker_id=last.speaker_id,
                    start_ms=last.start_ms,
                    end_ms=seg.end_ms,
                    start_sec=last.start_sec,
                    end_sec=seg.end_sec
                )
            else:
                merged.append(seg)
        
        if len(merged) < len(segments):
            print(f"   Merged {len(segments)-len(merged)} close segments")
        return merged
    
    def _split_long_segments(self, segments: List[SpeakerSegment], max_duration_sec: float = 30.0) -> List[SpeakerSegment]:
        result = []
        max_ms = int(max_duration_sec * 1000)
        
        for seg in segments:
            if seg.duration_ms <= max_ms:
                result.append(seg)
                continue
                
            # Split long segment
            start = seg.start_ms
            while start < seg.end_ms:
                end = min(start + max_ms, seg.end_ms)
                result.append(SpeakerSegment(
                    speaker_id=seg.speaker_id,
                    start_ms=start,
                    end_ms=end,
                    start_sec=start/1000.0,
                    end_sec=end/1000.0
                ))
                start = end
        
        if len(result) > len(segments):
            print(f"   Split {len(result)-len(segments)} long segments")
        return result

    def _collect_ref_audio(self, waveform: torch.Tensor, sr: int, segs: List[SpeakerSegment], target_sec: float = 12.0):
        """
        Collect up to target_sec of audio for a speaker (for embedding).
        """
        total = 0
        chunks = []
        min_chunk_ms = int(self.merge_config.emb_min_chunk_ms)
        for s in segs:
            if total >= target_sec:
                break
            ms = s.duration_ms
            if ms < min_chunk_ms:
                continue
            start = int(s.start_sec * sr)
            end = int(s.end_sec * sr)
            chunks.append(waveform[:, start:end])
            total += (end - start) / sr
        if not chunks:
            return None
        return torch.cat(chunks, dim=1)

    def _consolidate_speakers(self, waveform: torch.Tensor, sr: int, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Post-process pyannote speaker labels:
        - Merge very similar speakers using embeddings
        - Absorb tiny speakers into nearest similar big speaker
        Env vars supported (optional):
            - SPEAKER_MERGE_ENABLE (default "1")
            - SPEAKER_MERGE_SIM (default "0.78")  # higher = stricter merge
            - SPEAKER_TINY_TOTAL_MS (default "1200")  # speakers with < this total are absorbed
            - SPEAKER_MERGE_REF_SEC (default "12")  # audio used per speaker for embedding
        """
        if not self.merge_config.merge_enable:
            return segments

        # Need speechbrain for embeddings
        self._load_spkrec()
        if self._spkrec is None:
            return segments

        # group segments by speaker id
        by_spk: Dict[str, List[SpeakerSegment]] = defaultdict(list)
        for s in segments:
            by_spk[s.speaker_id].append(s)

        # Build reference audio and embedding per speaker
        ref_sec = float(self.merge_config.merge_ref_sec)
        embeds: Dict[str, np.ndarray] = {}
        totals_ms: Dict[str, int] = {}
        for spk, segs in by_spk.items():
            totals_ms[spk] = sum(x.duration_ms for x in segs)
            ref = self._collect_ref_audio(waveform, sr, segs, target_sec=ref_sec)
            if ref is None:
                continue
            # speechbrain expects batch x time, float
            wav = ref.to(self._spkrec_device).float()
            with torch.no_grad():
                emb = self._spkrec.encode_batch(wav).squeeze(0).squeeze(0).detach().cpu().numpy()
            embeds[spk] = emb

        if len(embeds) <= 1:
            return segments

        # Merge threshold + tiny absorption threshold
        sim_thr = float(self.merge_config.merge_sim)
        tiny_thr = int(self.merge_config.tiny_total_ms)

        # Determine "big" speakers
        speakers_sorted = sorted(totals_ms.items(), key=lambda kv: kv[1], reverse=True)
        big = [s for s, ms in speakers_sorted if ms >= tiny_thr and s in embeds]
        small = [s for s, ms in speakers_sorted if ms < tiny_thr and s in embeds]

        # Build union-find for merging similar big speakers
        parent = {s: s for s in embeds.keys()}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Merge big-big if very similar
        for i in range(len(big)):
            for j in range(i + 1, len(big)):
                a, b = big[i], big[j]
                sim = _cosine_sim(embeds[a], embeds[b])
                if sim >= sim_thr:
                    union(a, b)

        # Absorb small into best big if close enough (looser threshold)
        absorb_thr = float(self.merge_config.resolved_absorb_sim())
        for s in small:
            best = None
            best_sim = -1.0
            for b in big:
                sim = _cosine_sim(embeds[s], embeds[b])
                if sim > best_sim:
                    best_sim = sim
                    best = b
            if best is not None and best_sim >= absorb_thr:
                union(best, s)

        # Map each speaker to canonical representative
        mapping = {s: find(s) for s in embeds.keys()}
        # ensure stable representative names
        # pick representative with most total ms within each group
        groups = defaultdict(list)
        for s, r in mapping.items():
            groups[r].append(s)

        rep_for = {}
        for r, members in groups.items():
            rep = max(members, key=lambda m: totals_ms.get(m, 0))
            for m in members:
                rep_for[m] = rep

        # Apply mapping
        out = []
        for seg in segments:
            sid = seg.speaker_id
            if sid in rep_for:
                sid = rep_for[sid]
            out.append(SpeakerSegment(
                speaker_id=sid,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec
            ))

        # Optional: merge close segments again after relabel
        out = self._merge_close_segments(out)
        return out
