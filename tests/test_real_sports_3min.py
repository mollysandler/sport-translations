"""
Test embedding-based speaker continuity on 3 minutes of real sports commentary.

Detailed analysis comparing pitch-only vs embedding+pitch matching
with per-chunk, per-segment visibility into matching decisions.
"""
import sys, os, time
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    device = torch.device("cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    return model, device


def _load_diarizer():
    """Load the REAL SportsDiarizer (bypasses conftest stub)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "real_diarizer",
            os.path.join(_project_root, "diarizer.py")
        )
        real_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_mod)
        d = real_mod.SportsDiarizer()
        return d
    except Exception as e:
        print(f"  Could not load SportsDiarizer: {e}")
        return None


class TestThreeMinuteSports:

    def test_3min_detailed_comparison(self):
        """Compare pitch-only vs embedding+pitch on 3 minutes of real commentary."""
        import soundfile as sf
        audio_path = "/tmp/long_audio_180s.wav"
        if not os.path.exists(audio_path):
            import pytest; pytest.skip("Run ffmpeg extraction first")

        audio_np, sr = sf.read(audio_path)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        total_sec = len(audio_np) / sr
        print(f"\n  Audio: {total_sec:.1f}s, {sr}Hz")

        from main import _estimate_pitch_safe

        diarizer = _load_diarizer()
        if diarizer is None:
            import pytest; pytest.skip("Diarizer not available")

        model, device = _load_ecapa()

        CHUNK_SEC = 10.0
        num_chunks = int(total_sec / CHUNK_SEC)

        # ================================================================
        # Phase 1: Diarize all chunks first (shared between both methods)
        # ================================================================
        print(f"\n  Diarizing {num_chunks} chunks of {CHUNK_SEC}s...")
        all_chunk_data = []

        for chunk_idx in range(num_chunks):
            chunk_start_sec = chunk_idx * CHUNK_SEC
            chunk_start = int(chunk_start_sec * sr)
            chunk_end = int((chunk_start_sec + CHUNK_SEC) * sr)
            chunk_audio = audio_tensor[:, chunk_start:chunk_end]

            t0 = time.perf_counter()
            segments = diarizer.diarize(chunk_audio, sr)
            diarize_ms = (time.perf_counter() - t0) * 1000

            seg_data = []
            for seg in segments:
                if seg.end_sec - seg.start_sec < 0.5:
                    continue
                s = int(seg.start_sec * sr)
                e = int(seg.end_sec * sr)
                seg_audio = chunk_audio[:, s:e]
                seg_np = seg_audio.squeeze().numpy().astype(np.float32)

                try:
                    pitch, _ = _estimate_pitch_safe(seg_np, sr)
                except:
                    pitch = 150.0

                # Extract embedding
                t_emb = time.perf_counter()
                wav = seg_audio.to(device).float()
                with torch.no_grad():
                    emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
                emb_ms = (time.perf_counter() - t_emb) * 1000

                seg_data.append({
                    "diar_spk": seg.speaker_id,
                    "start": seg.start_sec + chunk_start_sec,
                    "end": seg.end_sec + chunk_start_sec,
                    "dur": seg.end_sec - seg.start_sec,
                    "pitch": pitch,
                    "emb": emb,
                    "emb_ms": emb_ms,
                })

            all_chunk_data.append({
                "chunk_idx": chunk_idx,
                "chunk_start": chunk_start_sec,
                "diarize_ms": diarize_ms,
                "segments": seg_data,
            })

            n_spk = len(set(s["diar_spk"] for s in seg_data))
            print(f"    Chunk {chunk_idx:2d} [{chunk_start_sec:5.0f}-{chunk_start_sec+CHUNK_SEC:5.0f}s]: "
                  f"{len(seg_data)} segs, {n_spk} spk, diarize={diarize_ms:.0f}ms")

        total_segs = sum(len(c["segments"]) for c in all_chunk_data)
        print(f"\n  Total: {total_segs} segments across {num_chunks} chunks")

        # ================================================================
        # Phase 2: Pitch-only matching
        # ================================================================
        print(f"\n{'='*74}")
        print(f"  METHOD 1: PITCH-ONLY (30Hz threshold)")
        print(f"{'='*74}")

        pitch_state = {"pitches": {}}
        pitch_log = []

        for chunk in all_chunk_data:
            for seg in chunk["segments"]:
                pitch = seg["pitch"]
                best_match = None
                best_dist = float("inf")

                for spk, stored_p in pitch_state["pitches"].items():
                    dist = abs(pitch - stored_p)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = spk

                if best_match is not None and best_dist <= 30.0:
                    assigned = best_match
                    method = f"PITCH({best_dist:.0f}Hz)"
                else:
                    assigned = f"V{len(pitch_state['pitches'])}"
                    pitch_state["pitches"][assigned] = pitch
                    method = "NEW"

                pitch_log.append({
                    "chunk": chunk["chunk_idx"],
                    "diar_spk": seg["diar_spk"],
                    "assigned": assigned,
                    "pitch": pitch,
                    "method": method,
                    "time": f"{seg['start']:.1f}-{seg['end']:.1f}s",
                })

                print(f"    [{seg['start']:6.1f}-{seg['end']:6.1f}s] "
                      f"{seg['diar_spk']:>11} pitch={pitch:5.0f}Hz -> {assigned:>4} [{method}]")

        # ================================================================
        # Phase 3: Embedding + Pitch matching
        # ================================================================
        print(f"\n{'='*74}")
        print(f"  METHOD 2: EMBEDDING + PITCH (threshold=0.40, margin=0.03)")
        print(f"{'='*74}")

        emb_state = {"pitches": {}, "embeddings": {}}
        emb_log = []

        EMB_THRESHOLD = 0.40
        EMB_MARGIN = 0.03

        for chunk in all_chunk_data:
            for seg in chunk["segments"]:
                pitch = seg["pitch"]
                emb = seg["emb"]

                assigned = None
                method = "NEW"

                # Try embedding match
                if emb_state["embeddings"]:
                    sims = {k: _cosine_sim(emb, v)
                            for k, v in emb_state["embeddings"].items()}
                    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                    best_k, best_sim = sorted_sims[0]
                    second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

                    sims_str = " ".join(f"{k}={v:.2f}" for k, v in sorted_sims)

                    if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                        assigned = best_k
                        method = f"EMB({best_sim:.2f},m={best_sim-second_sim:.2f})"
                    else:
                        # Try pitch fallback
                        best_dist = float("inf")
                        best_p = None
                        for spk, stored_p in emb_state["pitches"].items():
                            dist = abs(pitch - stored_p)
                            if dist < best_dist:
                                best_dist = dist
                                best_p = spk
                        if best_p is not None and best_dist <= 30.0:
                            assigned = best_p
                            method = f"PITCH({best_dist:.0f}Hz) [emb: {sims_str}]"
                        else:
                            method = f"NEW [emb: {sims_str}]"

                if assigned is None:
                    assigned = f"V{len(emb_state['pitches'])}"
                    emb_state["pitches"][assigned] = pitch
                    emb_state["embeddings"][assigned] = emb
                    method_short = "NEW"
                else:
                    method_short = method.split("[")[0].strip()
                    # EMA update embedding
                    if assigned in emb_state["embeddings"]:
                        old = emb_state["embeddings"][assigned]
                        emb_state["embeddings"][assigned] = 0.7 * old + 0.3 * emb

                emb_log.append({
                    "chunk": chunk["chunk_idx"],
                    "diar_spk": seg["diar_spk"],
                    "assigned": assigned,
                    "pitch": pitch,
                    "method": method,
                    "time": f"{seg['start']:.1f}-{seg['end']:.1f}s",
                })

                # Truncate method for display
                method_display = method[:55] if len(method) > 55 else method
                print(f"    [{seg['start']:6.1f}-{seg['end']:6.1f}s] "
                      f"{seg['diar_spk']:>11} pitch={pitch:5.0f}Hz -> {assigned:>4} [{method_display}]")

        # ================================================================
        # Phase 4: Analysis
        # ================================================================
        print(f"\n{'='*74}")
        print(f"  ANALYSIS")
        print(f"{'='*74}")

        def analyze(log, label):
            print(f"\n  {label}:")
            by_diar = {}
            for entry in log:
                by_diar.setdefault(entry["diar_spk"], []).append(entry["assigned"])

            session_spks = set()
            for diar_spk, ids in sorted(by_diar.items()):
                from collections import Counter
                counts = Counter(ids)
                most_common, mc_count = counts.most_common(1)[0]
                consistency = mc_count / len(ids)
                session_spks.update(ids)
                detail = ", ".join(f"{k}:{v}" for k, v in counts.most_common())
                print(f"    {diar_spk:>11}: {len(ids):2d} segs -> [{detail}] "
                      f"({consistency:.0%} consistent)")

            print(f"    Session voices created: {len(session_spks)}")

            # Cross-check: do different diarizer speakers map to different session voices?
            primaries = {}
            for diar_spk, ids in by_diar.items():
                from collections import Counter
                primaries[diar_spk] = Counter(ids).most_common(1)[0][0]

            unique_primaries = set(primaries.values())
            collision = len(unique_primaries) < len(primaries)
            if collision:
                print(f"    WARNING: {len(primaries)} diarizer speakers mapped to "
                      f"{len(unique_primaries)} session voices (collision!)")
                for d, p in sorted(primaries.items()):
                    print(f"      {d} -> {p}")
            else:
                print(f"    All {len(primaries)} diarizer speakers mapped to distinct voices")

            return session_spks, primaries

        pitch_spks, pitch_primaries = analyze(pitch_log, "PITCH-ONLY")
        emb_spks, emb_primaries = analyze(emb_log, "EMBEDDING + PITCH")

        # Method breakdown for embedding approach
        methods = [e["method"] for e in emb_log]
        emb_count = sum(1 for m in methods if m.startswith("EMB"))
        pitch_count = sum(1 for m in methods if m.startswith("PITCH"))
        new_count = sum(1 for m in methods if m.startswith("NEW"))
        print(f"\n  Embedding method breakdown: "
              f"{emb_count} embedding, {pitch_count} pitch fallback, {new_count} new "
              f"(total {len(methods)})")

        # Embedding similarity overview
        print(f"\n  Stored embedding similarities (final state):")
        emb_keys = list(emb_state["embeddings"].keys())
        for i in range(len(emb_keys)):
            for j in range(i + 1, len(emb_keys)):
                sim = _cosine_sim(emb_state["embeddings"][emb_keys[i]],
                                  emb_state["embeddings"][emb_keys[j]])
                print(f"    {emb_keys[i]} <-> {emb_keys[j]}: {sim:.3f}")

        # Summary
        print(f"\n  {'='*50}")
        print(f"  SUMMARY")
        print(f"  {'='*50}")
        print(f"  Pitch-only:      {len(pitch_spks)} voices created")
        print(f"  Embedding+pitch: {len(emb_spks)} voices created")

        # Latency
        total_emb_ms = sum(s["emb_ms"] for c in all_chunk_data for s in c["segments"])
        total_diar_ms = sum(c["diarize_ms"] for c in all_chunk_data)
        print(f"\n  Total diarization time:  {total_diar_ms:.0f}ms ({total_diar_ms/num_chunks:.0f}ms/chunk)")
        print(f"  Total embedding time:   {total_emb_ms:.0f}ms ({total_emb_ms/total_segs:.0f}ms/segment)")
        print(f"  Embedding overhead:     {total_emb_ms/total_diar_ms*100:.1f}% of diarization time")
