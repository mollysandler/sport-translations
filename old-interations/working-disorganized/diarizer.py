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
        # Optional: speaker embedding model for post-merge consolidation
        self._spkrec = None
        self._spkrec_device = torch.device("cpu")

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
            This is used ONLY to merge pyannote's 'duplicate speakers' post-hoc.
            """
            if self._spkrec is not None:
                return self._spkrec
            try:
                # SpeechBrain 1.0+ prefers speechbrain.inference
                from speechbrain.inference import SpeakerRecognition
            except Exception:
                from speechbrain.pretrained import SpeakerRecognition  # fallback

            # This will download on first run if not cached.
            self._spkrec = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.path.expanduser("~"), ".cache", "sb_spkrec_ecapa"),
                run_opts={"device": str(self._spkrec_device)},
            )
            return self._spkrec

    def _speaker_total_ms(self, segments: List[SpeakerSegment]) -> Dict[str, int]:
            tot = defaultdict(int)
            for s in segments:
                tot[s.speaker_id] += (s.end_ms - s.start_ms)
            return dict(tot)

    def _collect_ref_audio(self, waveform: torch.Tensor, sr: int, segments: List[SpeakerSegment], target_sec: float = 12.0):
            """
            Collect up to target_sec of audio for a speaker by concatenating their segments.
            Returns torch tensor shape [1, T].
            """
            want = int(target_sec * sr)
            chunks = []
            got = 0
            for seg in segments:
                start = int(seg.start_sec * sr)
                end = int(seg.end_sec * sr)
                if end <= start:
                    continue
                chunk = waveform[:, start:end]
                min_chunk_ms = int(os.getenv("SPEAKER_EMB_MIN_CHUNK_MS", "250"))
                if chunk.shape[1] < int((min_chunk_ms / 1000) * sr):
                    continue
                chunks.append(chunk)
                got += chunk.shape[1]
                if got >= want:
                    break
            if not chunks:
                return None
            x = torch.cat(chunks, dim=1)
            if x.shape[1] > want:
                x = x[:, :want]
            return x

    def _embed_speaker(self, waveform: torch.Tensor, sr: int, segs: List[SpeakerSegment]) -> Optional[np.ndarray]:
            """
            Compute a single embedding vector for a speaker.
            """
            spkrec = self._load_spkrec()
            ref = self._collect_ref_audio(waveform, sr, segs, target_sec=float(os.getenv("SPEAKER_MERGE_REF_SEC", "12")))
            if ref is None:
                return None

            # SpeechBrain expects [batch, time] or [batch, time, channels] depending on version.
            # We'll give [batch, time].
            wav = ref.squeeze(0).detach().cpu()
            if wav.ndim != 1:
                wav = wav.flatten()

            # encode_batch expects tensor [batch, time]
            emb = spkrec.encode_batch(wav.unsqueeze(0))
            # emb shape often [1, 1, D] or [1, D]
            emb = emb.detach().cpu().numpy()
            emb = emb.reshape(-1)
            return emb

    def _consolidate_speakers(self, waveform: torch.Tensor, sr: int, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
            """
            Post-merge speaker labels that are actually the same person.

            Controls (env vars):
            - SPEAKER_MERGE_ENABLE (default "1")
            - SPEAKER_MERGE_SIM (default "0.78")  # higher = stricter merge
            - SPEAKER_TINY_TOTAL_MS (default "1200")  # speakers with < this total are absorbed
            - SPEAKER_MERGE_REF_SEC (default "12")  # audio used per speaker for embedding
            """
            if os.getenv("SPEAKER_MERGE_ENABLE", "1") != "1":
                return segments

            # If only 0/1 speakers, nothing to do
            speakers = sorted(set(s.speaker_id for s in segments))
            if len(speakers) <= 1:
                return segments

            # Group segments per speaker
            by_spk = defaultdict(list)
            for s in segments:
                by_spk[s.speaker_id].append(s)

            # Compute embeddings
            try:
                embs = {}
                for spk in speakers:
                    e = self._embed_speaker(waveform, sr, by_spk[spk])
                    if e is not None:
                        embs[spk] = e
            except Exception as e:
                print(f"   ⚠️  Speaker-merge disabled (embedding model failed): {e}")
                return segments

            if len(embs) <= 1:
                return segments

            sim_thr = float(os.getenv("SPEAKER_MERGE_SIM", "0.78"))
            tiny_thr = int(os.getenv("SPEAKER_TINY_TOTAL_MS", "1200"))

            totals = self._speaker_total_ms(segments)
            main_speakers = sorted(embs.keys(), key=lambda k: totals.get(k, 0), reverse=True)

            # Union-Find to merge speakers above similarity threshold
            parent = {spk: spk for spk in embs.keys()}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return
                # attach smaller-total to larger-total
                ta = totals.get(ra, 0)
                tb = totals.get(rb, 0)
                if ta >= tb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

            spk_list = list(embs.keys())
            for i in range(len(spk_list)):
                for j in range(i + 1, len(spk_list)):
                    a, b = spk_list[i], spk_list[j]
                    sim = _cosine_sim(embs[a], embs[b])
                    if sim >= sim_thr:
                        union(a, b)

            # Compress parents
            clusters = defaultdict(list)
            for spk in spk_list:
                clusters[find(spk)].append(spk)

            # Build mapping: each cluster -> representative (largest total_ms)
            cluster_rep = {}
            for root, members in clusters.items():
                rep = max(members, key=lambda k: totals.get(k, 0))
                for m in members:
                    cluster_rep[m] = rep

            # Absorb tiny speakers into closest big speaker (optional)
            big = [s for s in main_speakers if totals.get(s, 0) >= tiny_thr]
            if len(big) >= 1:
                for spk in main_speakers[::-1]:
                    if totals.get(spk, 0) >= tiny_thr:
                        continue
                    if spk not in embs:
                        continue
                    # choose closest big speaker by cosine sim
                    best = None
                    best_sim = -1.0
                    for b in big:
                        if b not in embs:
                            continue
                        sim = _cosine_sim(embs[spk], embs[b])
                        if sim > best_sim:
                            best_sim = sim
                            best = b
                    absorb_thr = float(os.getenv("SPEAKER_ABSORB_SIM", str(sim_thr - 0.20)))  # default looser
                    if best is not None and best_sim >= absorb_thr:
                        cluster_rep[spk] = cluster_rep.get(best, best)

            # Apply mapping
            new_segments = []
            for s in segments:
                new_id = cluster_rep.get(s.speaker_id, s.speaker_id)
                if new_id != s.speaker_id:
                    new_segments.append(SpeakerSegment(new_id, s.start_ms, s.end_ms, s.start_sec, s.end_sec))
                else:
                    new_segments.append(s)

            # Recompute and log
            new_speakers = sorted(set(s.speaker_id for s in new_segments))
            if len(new_speakers) != len(speakers):
                print(f"✅ Speaker merge: {len(speakers)} -> {len(new_speakers)} speakers (sim≥{sim_thr})")
            else:
                print(f"ℹ️  Speaker merge: no merges (sim≥{sim_thr})")

            return new_segments

    
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