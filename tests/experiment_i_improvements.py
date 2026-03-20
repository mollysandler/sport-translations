"""
Experiment I: Test v7c improvements individually and in combination.

Improvements tested:
  A. Threshold tuning — raise EMB_THRESHOLD from 0.40 to 0.45/0.50/0.55
  B. Overlapping chunks — 2s overlap for better boundary diarization
  C. Min segment duration — raise from 0.5s to 1.0s for better embeddings
  D. Embedding quality gating — weight by segment duration, ignore short-seg embeddings

Runs on 3-minute sports commentary (/tmp/long_audio_180s.wav).
"""
import sys, os, time
import numpy as np
import torch
from collections import Counter

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


# ======================================================================
# Load models once, share across all experiments
# ======================================================================
_diarizer = None
_ecapa_model = None
_ecapa_device = None


def _get_diarizer():
    global _diarizer
    if _diarizer is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "real_diarizer", os.path.join(_project_root, "diarizer.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _diarizer = mod.SportsDiarizer()
    return _diarizer


def _get_ecapa():
    global _ecapa_model, _ecapa_device
    if _ecapa_model is None:
        d = _get_diarizer()
        d._load_spkrec()
        _ecapa_model = d._spkrec
        _ecapa_device = d._spkrec_device
    return _ecapa_model, _ecapa_device


def _extract_embedding(audio_tensor):
    """Extract ECAPA embedding from [1, N] tensor."""
    model, device = _get_ecapa()
    wav = audio_tensor.to(device).float()
    with torch.no_grad():
        emb = model.encode_batch(wav)
    return emb.squeeze(0).squeeze(0).detach().cpu().numpy()


def _estimate_pitch(audio_np, sr):
    from main import _estimate_pitch_safe
    try:
        import librosa
        audio_np = librosa.effects.preemphasis(audio_np)
    except Exception:
        pass
    try:
        return _estimate_pitch_safe(audio_np, sr)
    except Exception:
        return 150.0, 20.0


# ======================================================================
# Core v7c matching with configurable parameters
# ======================================================================
def run_v7c_matching(
    audio_tensor,
    sr,
    *,
    chunk_sec=10.0,
    overlap_sec=0.0,
    min_seg_dur=0.5,
    emb_threshold=0.40,
    emb_margin=0.03,
    secondary_emb_threshold=0.25,
    history_bonus=0.10,
    duration_ratio_threshold=2.0,
    pitch_match_hz=30.0,
    min_emb_dur=0.0,       # minimum duration to trust embedding (0 = use all)
    label="experiment",
):
    """Run chunked diarization + v7c voice assignment with given parameters."""
    diarizer = _get_diarizer()
    total_sec = audio_tensor.shape[1] / sr

    # Build chunk windows (with optional overlap)
    step = chunk_sec - overlap_sec
    chunk_windows = []
    pos = 0.0
    while pos + chunk_sec <= total_sec + 0.1:
        chunk_windows.append((pos, min(pos + chunk_sec, total_sec)))
        pos += step
        if pos >= total_sec:
            break

    # Session state
    state = {
        "embeddings": {},
        "pitches": {},
        "diar_history": {},
        "_next_id": 0,
    }

    all_results = []
    prev_chunk_end = 0.0  # track where previous chunk's "new" region ended

    for ci, (c_start, c_end) in enumerate(chunk_windows):
        chunk_start_sample = int(c_start * sr)
        chunk_end_sample = int(c_end * sr)
        chunk_audio = audio_tensor[:, chunk_start_sample:chunk_end_sample]

        # Diarize
        segments = diarizer.diarize(chunk_audio, sr)
        if not segments:
            continue

        # Filter: only keep segments in the "new" region (after overlap)
        # For the first chunk, everything is new. For subsequent chunks,
        # only segments starting after the overlap region.
        if overlap_sec > 0 and ci > 0:
            new_region_start = overlap_sec
            segments = [s for s in segments
                        if s.start_sec >= new_region_start - 0.2]  # 0.2s grace

        if not segments:
            continue

        # Group by speaker
        spk_groups = {}
        for seg in segments:
            spk_groups.setdefault(seg.speaker_id, []).append(seg)

        # Extract features per speaker
        speaker_data = {}
        for spk_id, segs in spk_groups.items():
            collected = []
            total_dur = 0.0
            for seg in segs:
                s = int(seg.start_sec * sr)
                e = int(seg.end_sec * sr)
                seg_audio = chunk_audio[:, s:e]
                dur = (e - s) / sr
                if dur >= 0.1:
                    collected.append(seg_audio)
                    total_dur += dur
                if total_dur >= 5.0:
                    break

            if not collected or total_dur < min_seg_dur:
                continue

            combined = torch.cat(collected, dim=1)
            audio_np = combined.squeeze().detach().cpu().numpy().astype(np.float32)
            pitch, pitch_range = _estimate_pitch(audio_np, sr)

            # Embedding — skip if below min_emb_dur
            if total_dur >= max(min_emb_dur, 0.5):
                emb = _extract_embedding(combined)
            else:
                emb = None

            speaker_data[spk_id] = {
                "pitch": pitch,
                "pitch_range": pitch_range,
                "embedding": emb,
                "seg_count": len(segs),
                "seg_duration": total_dur,
            }

        speakers = list(speaker_data.keys())
        if not speakers:
            continue

        is_multi = len(speakers) >= 2

        # --- Matching helpers ---
        def _match_emb_normal(emb):
            if emb is None or not state["embeddings"]:
                return None
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_s = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_s[0]
            second = sorted_s[1][1] if len(sorted_s) > 1 else 0.0
            if best_sim >= emb_threshold and (best_sim - second) >= emb_margin:
                return best_k, best_sim
            return None

        def _match_emb_secondary(emb, exclude, hist_pref):
            if emb is None or not state["embeddings"]:
                return None
            cands = {k: _cosine_sim(emb, v)
                     for k, v in state["embeddings"].items() if k != exclude}
            if not cands:
                return None
            boosted = {}
            for k, sim in cands.items():
                bonus = history_bonus if k == hist_pref else 0.0
                boosted[k] = sim + bonus
            best_k = max(boosted, key=boosted.get)
            if boosted[best_k] >= secondary_emb_threshold:
                return best_k, cands[best_k]
            return None

        def _match_pitch(pitch, exclude=None):
            best_d = float("inf")
            best_k = None
            for k, p in state["pitches"].items():
                if k == exclude:
                    continue
                d = abs(pitch - p)
                if d <= pitch_match_hz and d < best_d:
                    best_d = d
                    best_k = k
            return (best_k, best_d) if best_k else None

        def _assign(spk_id, data, session_spk):
            if session_spk is not None:
                if data["embedding"] is not None and session_spk in state["embeddings"]:
                    old = state["embeddings"][session_spk]
                    state["embeddings"][session_spk] = 0.7 * old + 0.3 * data["embedding"]
            else:
                session_spk = f"V{state['_next_id']}"
                state["_next_id"] += 1
                state["pitches"][session_spk] = data["pitch"]
                if data["embedding"] is not None:
                    state["embeddings"][session_spk] = data["embedding"]
            hist = state["diar_history"].setdefault(spk_id, Counter())
            hist[session_spk] += 1
            return session_spk

        def _do_normal(spk_id):
            data = speaker_data[spk_id]
            m = _match_emb_normal(data["embedding"])
            if m:
                return _assign(spk_id, data, m[0]), f"EMB({m[1]:.2f})"
            m = _match_pitch(data["pitch"])
            if m:
                return _assign(spk_id, data, m[0]), f"PITCH({m[1]:.0f}Hz)"
            return _assign(spk_id, data, None), "NEW"

        chunk_results = []

        if not is_multi:
            for spk_id in speakers:
                voice, method = _do_normal(spk_id)
                chunk_results.append({
                    "chunk": ci, "time_start": c_start, "time_end": c_end,
                    "diar_spk": spk_id, "assigned": voice, "method": method,
                    "pitch": speaker_data[spk_id]["pitch"],
                })
        else:
            # Hybrid dominance
            if len(speakers) == 2:
                a, b = speakers
                dur_a = speaker_data[a]["seg_duration"]
                dur_b = speaker_data[b]["seg_duration"]
                ratio = max(dur_a, dur_b) / max(min(dur_a, dur_b), 0.01)
                if ratio >= duration_ratio_threshold:
                    dom = a if dur_a >= dur_b else b
                else:
                    dom = a if speaker_data[a]["seg_count"] >= speaker_data[b]["seg_count"] else b
            else:
                dom = max(speakers, key=lambda s: speaker_data[s]["seg_count"])

            secondaries = [s for s in speakers if s != dom]

            # Pass 1: dominant
            dom_voice, dom_method = _do_normal(dom)
            chunk_results.append({
                "chunk": ci, "time_start": c_start, "time_end": c_end,
                "diar_spk": dom, "assigned": dom_voice, "method": f"{dom_method}[dom]",
                "pitch": speaker_data[dom]["pitch"],
            })

            # Pass 2: secondary with exclusion
            for spk_id in secondaries:
                data = speaker_data[spk_id]
                hist = state["diar_history"].get(spk_id, Counter())
                hist_pref = hist.most_common(1)[0][0] if hist else None

                m = _match_emb_secondary(data["embedding"], dom_voice, hist_pref)
                if m:
                    voice = _assign(spk_id, data, m[0])
                    bonus_str = "+hist" if m[0] == hist_pref else ""
                    method = f"SEC({m[1]:.2f}{bonus_str})"
                else:
                    pm = _match_pitch(data["pitch"], exclude=dom_voice)
                    if pm:
                        voice = _assign(spk_id, data, pm[0])
                        method = f"PITCH({pm[1]:.0f}Hz)"
                    else:
                        voice = _assign(spk_id, data, None)
                        method = "NEW(sec)"

                chunk_results.append({
                    "chunk": ci, "time_start": c_start, "time_end": c_end,
                    "diar_spk": spk_id, "assigned": voice, "method": method,
                    "pitch": data["pitch"],
                })

        all_results.extend(chunk_results)

    return all_results, state


# ======================================================================
# Analysis
# ======================================================================
def analyze_results(results, state, label):
    """Analyze and print results for one experiment run."""
    if not results:
        print(f"\n  [{label}] No results!")
        return {}

    # Group by diar speaker
    by_diar = {}
    for r in results:
        by_diar.setdefault(r["diar_spk"], []).append(r["assigned"])

    n_voices = len(set(r["assigned"] for r in results))
    n_chunks = len(set(r["chunk"] for r in results))

    metrics = {
        "label": label,
        "n_voices": n_voices,
        "n_chunks": n_chunks,
    }

    consistencies = {}
    primaries = {}
    for diar_spk, assignments in sorted(by_diar.items()):
        counts = Counter(assignments)
        mc, mc_count = counts.most_common(1)[0]
        pct = mc_count / len(assignments)
        consistencies[diar_spk] = pct
        primaries[diar_spk] = mc

    metrics["consistencies"] = consistencies
    metrics["primaries"] = primaries
    metrics["distinct"] = len(set(primaries.values())) == len(primaries)

    # Method breakdown
    methods = [r["method"] for r in results]
    emb_count = sum(1 for m in methods if "EMB" in m or "SEC" in m)
    pitch_count = sum(1 for m in methods if m.startswith("PITCH"))
    new_count = sum(1 for m in methods if "NEW" in m)
    metrics["emb_matches"] = emb_count
    metrics["pitch_matches"] = pitch_count
    metrics["new_voices"] = new_count

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"  Voices: {n_voices} | Chunks: {n_chunks} | "
          f"Distinct primaries: {'YES' if metrics['distinct'] else 'NO'}")
    print(f"  Methods: {emb_count} emb, {pitch_count} pitch, {new_count} new")
    for diar_spk, pct in consistencies.items():
        counts = Counter(by_diar[diar_spk])
        detail = ", ".join(f"{k}:{v}" for k, v in counts.most_common())
        print(f"    {diar_spk}: {pct:.0%} consistent ({len(by_diar[diar_spk])} segs) [{detail}]")

    return metrics


# ======================================================================
# Main experiment runner
# ======================================================================
def run_all():
    import soundfile as sf

    audio_path = "/tmp/long_audio_180s.wav"
    if not os.path.exists(audio_path):
        print("ERROR: /tmp/long_audio_180s.wav not found")
        print("Run: ffmpeg -y -i long.mp4 -ac 1 -ar 16000 -t 180 /tmp/long_audio_180s.wav")
        return

    audio_np, sr = sf.read(audio_path)
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    total_sec = len(audio_np) / sr
    print(f"Audio: {total_sec:.1f}s, {sr}Hz")
    print(f"Loading models...")

    # Force model load once
    _get_diarizer()
    _get_ecapa()
    print(f"Models ready.\n")

    all_metrics = []

    # ------------------------------------------------------------------
    # 0. BASELINE — current v7c (threshold=0.40)
    # ------------------------------------------------------------------
    print("\n>>> Running: BASELINE (v7c, threshold=0.40)")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, emb_threshold=0.40, label="baseline")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "0. BASELINE (v7c, thresh=0.40)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # A1. Threshold = 0.45
    # ------------------------------------------------------------------
    print("\n>>> Running: THRESHOLD 0.45")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, emb_threshold=0.45, label="thresh-0.45")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "A1. Threshold 0.45")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # A2. Threshold = 0.50
    # ------------------------------------------------------------------
    print("\n>>> Running: THRESHOLD 0.50")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, emb_threshold=0.50, label="thresh-0.50")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "A2. Threshold 0.50")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # A3. Threshold = 0.55
    # ------------------------------------------------------------------
    print("\n>>> Running: THRESHOLD 0.55")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, emb_threshold=0.55, label="thresh-0.55")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "A3. Threshold 0.55")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # B1. Overlapping chunks (2s overlap)
    # ------------------------------------------------------------------
    print("\n>>> Running: OVERLAP 2s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, overlap_sec=2.0, emb_threshold=0.40, label="overlap-2s")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "B1. Overlap 2s (thresh=0.40)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # B2. Overlapping chunks (3s overlap)
    # ------------------------------------------------------------------
    print("\n>>> Running: OVERLAP 3s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, overlap_sec=3.0, emb_threshold=0.40, label="overlap-3s")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "B2. Overlap 3s (thresh=0.40)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # C1. Min segment duration 1.0s
    # ------------------------------------------------------------------
    print("\n>>> Running: MIN_SEG_DUR 1.0s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, min_seg_dur=1.0, emb_threshold=0.40, label="min-seg-1.0")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "C1. Min seg dur 1.0s (thresh=0.40)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # C2. Min segment duration 1.5s
    # ------------------------------------------------------------------
    print("\n>>> Running: MIN_SEG_DUR 1.5s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, min_seg_dur=1.5, emb_threshold=0.40, label="min-seg-1.5")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "C2. Min seg dur 1.5s (thresh=0.40)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # D1. Embedding quality gate (min 1.0s for embedding)
    # ------------------------------------------------------------------
    print("\n>>> Running: EMB_GATE 1.0s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, min_emb_dur=1.0, emb_threshold=0.40, label="emb-gate-1.0")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "D1. Emb gate 1.0s (thresh=0.40)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # D2. Embedding quality gate (min 1.5s for embedding)
    # ------------------------------------------------------------------
    print("\n>>> Running: EMB_GATE 1.5s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, min_emb_dur=1.5, emb_threshold=0.40, label="emb-gate-1.5")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "D2. Emb gate 1.5s (thresh=0.40)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # COMBO 1: Best threshold + overlap
    # ------------------------------------------------------------------
    print("\n>>> Running: COMBO thresh=0.50 + overlap=2s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, overlap_sec=2.0, emb_threshold=0.50, label="combo-1")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "COMBO1: thresh=0.50 + overlap=2s")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # COMBO 2: Best threshold + min seg dur
    # ------------------------------------------------------------------
    print("\n>>> Running: COMBO thresh=0.50 + min_seg=1.0s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, min_seg_dur=1.0, emb_threshold=0.50, label="combo-2")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "COMBO2: thresh=0.50 + min_seg=1.0s")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # COMBO 3: threshold + overlap + min seg
    # ------------------------------------------------------------------
    print("\n>>> Running: COMBO thresh=0.50 + overlap=2s + min_seg=1.0s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, overlap_sec=2.0, min_seg_dur=1.0, emb_threshold=0.50, label="combo-3")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "COMBO3: thresh=0.50 + overlap=2s + min_seg=1.0s")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # COMBO 4: threshold + emb gate
    # ------------------------------------------------------------------
    print("\n>>> Running: COMBO thresh=0.50 + emb_gate=1.0s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, min_emb_dur=1.0, emb_threshold=0.50, label="combo-4")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "COMBO4: thresh=0.50 + emb_gate=1.0s")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # COMBO 5: all improvements
    # ------------------------------------------------------------------
    print("\n>>> Running: COMBO ALL (thresh=0.50 + overlap=2s + min_seg=1.0s + emb_gate=1.0s)")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, overlap_sec=2.0, min_seg_dur=1.0, min_emb_dur=1.0,
        emb_threshold=0.50, label="combo-all")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "COMBO5: ALL (thresh=0.50 + overlap=2s + min_seg=1.0s + emb_gate=1.0s)")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # COMBO 6: thresh=0.45 + overlap=2s + emb_gate=1.0s
    # ------------------------------------------------------------------
    print("\n>>> Running: COMBO thresh=0.45 + overlap=2s + emb_gate=1.0s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, overlap_sec=2.0, min_emb_dur=1.0,
        emb_threshold=0.45, label="combo-6")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "COMBO6: thresh=0.45 + overlap=2s + emb_gate=1.0s")
    m["time"] = elapsed
    all_metrics.append(m)

    # ------------------------------------------------------------------
    # COMBO 7: thresh=0.45 + overlap=2s + min_seg=1.0s
    # ------------------------------------------------------------------
    print("\n>>> Running: COMBO thresh=0.45 + overlap=2s + min_seg=1.0s")
    t0 = time.perf_counter()
    results, state = run_v7c_matching(
        audio_tensor, sr,
        chunk_sec=10.0, overlap_sec=2.0, min_seg_dur=1.0,
        emb_threshold=0.45, label="combo-7")
    elapsed = time.perf_counter() - t0
    m = analyze_results(results, state, "COMBO7: thresh=0.45 + overlap=2s + min_seg=1.0s")
    m["time"] = elapsed
    all_metrics.append(m)

    # ==================================================================
    # FINAL COMPARISON TABLE
    # ==================================================================
    print(f"\n\n{'='*90}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*90}")
    print(f"  {'Experiment':<55} {'Voices':>6} {'Distinct':>8} {'SPK_00':>7} {'SPK_01':>7} {'Time':>6}")
    print(f"  {'-'*55} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")

    for m in all_metrics:
        if not m:
            continue
        c = m.get("consistencies", {})
        spk00 = c.get("SPEAKER_00", 0)
        spk01 = c.get("SPEAKER_01", 0)
        distinct = "YES" if m.get("distinct") else "NO"
        t = m.get("time", 0)
        print(f"  {m['label']:<55} {m['n_voices']:>6} {distinct:>8} {spk00:>6.0%} {spk01:>6.0%} {t:>5.1f}s")

    # Find best
    best = None
    best_score = -1
    for m in all_metrics:
        if not m or not m.get("distinct"):
            continue
        c = m.get("consistencies", {})
        # Score: geometric mean of consistencies * penalty for extra voices
        vals = list(c.values())
        if not vals:
            continue
        geo_mean = np.prod(vals) ** (1.0 / len(vals))
        voice_penalty = 1.0 if m["n_voices"] <= 2 else 0.9 ** (m["n_voices"] - 2)
        score = geo_mean * voice_penalty
        if score > best_score:
            best_score = score
            best = m

    if best:
        print(f"\n  BEST: {best['label']} (score={best_score:.3f})")
    else:
        print(f"\n  No experiment achieved distinct primaries.")


if __name__ == "__main__":
    run_all()
