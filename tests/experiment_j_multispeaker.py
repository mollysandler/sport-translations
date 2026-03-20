"""
Experiment J: Test v7c on 10 minutes of sports commentary with 4+ speakers.

Runs the same improvement grid as experiment I but on longer, more diverse audio.
Also extracts per-voice embedding similarity matrix to assess false-match risk.
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


_diarizer = None


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


def _extract_embedding(audio_tensor):
    d = _get_diarizer()
    d._load_spkrec()
    wav = audio_tensor.to(d._spkrec_device).float()
    with torch.no_grad():
        emb = d._spkrec.encode_batch(wav)
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


def run_v7c(audio_tensor, sr, *, chunk_sec=10.0, emb_threshold=0.40,
            emb_margin=0.03, secondary_emb_threshold=0.25,
            history_bonus=0.10, duration_ratio_threshold=2.0,
            pitch_match_hz=30.0, min_seg_dur=0.5, min_emb_dur=0.0):
    """Run chunked diarization + v7c voice assignment."""
    diarizer = _get_diarizer()
    total_sec = audio_tensor.shape[1] / sr
    num_chunks = int(total_sec / chunk_sec)

    state = {"embeddings": {}, "pitches": {}, "diar_history": {}, "_next_id": 0}
    all_results = []

    for ci in range(num_chunks):
        cs = ci * chunk_sec
        chunk = audio_tensor[:, int(cs * sr):int((cs + chunk_sec) * sr)]
        segments = diarizer.diarize(chunk, sr)
        if not segments:
            continue

        # Group by speaker, extract features
        spk_groups = {}
        for seg in segments:
            spk_groups.setdefault(seg.speaker_id, []).append(seg)

        speaker_data = {}
        for spk_id, segs in spk_groups.items():
            collected, total_dur = [], 0.0
            for seg in segs:
                s, e = int(seg.start_sec * sr), int(seg.end_sec * sr)
                dur = (e - s) / sr
                if dur >= 0.1:
                    collected.append(chunk[:, s:e])
                    total_dur += dur
                if total_dur >= 5.0:
                    break
            if not collected or total_dur < min_seg_dur:
                continue
            combined = torch.cat(collected, dim=1)
            audio_np = combined.squeeze().detach().cpu().numpy().astype(np.float32)
            pitch, pr = _estimate_pitch(audio_np, sr)
            emb = _extract_embedding(combined) if total_dur >= max(min_emb_dur, 0.5) else None
            speaker_data[spk_id] = {
                "pitch": pitch, "embedding": emb,
                "seg_count": len(segs),
                "seg_duration": total_dur,
            }

        speakers = [s for s in sorted(spk_groups) if s in speaker_data]
        if not speakers:
            continue

        is_multi = len(speakers) >= 2

        # Matching helpers
        def _match_emb(emb, exclude=None, hist_pref=None):
            if emb is None or not state["embeddings"]:
                return None
            cands = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()
                     if k != exclude}
            if not cands:
                return None
            if exclude is not None:
                boosted = {k: sim + (history_bonus if k == hist_pref else 0)
                           for k, sim in cands.items()}
                bk = max(boosted, key=boosted.get)
                if boosted[bk] >= secondary_emb_threshold:
                    return bk, cands[bk], boosted[bk]
            else:
                ss = sorted(cands.items(), key=lambda x: x[1], reverse=True)
                bk, bs = ss[0]
                sec = ss[1][1] if len(ss) > 1 else 0.0
                if bs >= emb_threshold and (bs - sec) >= emb_margin:
                    return bk, bs, bs
            return None

        def _match_pitch(pitch, exclude=None):
            bd, bk = float("inf"), None
            for k, p in state["pitches"].items():
                if k == exclude:
                    continue
                d = abs(pitch - p)
                if d <= pitch_match_hz and d < bd:
                    bd, bk = d, k
            return (bk, bd) if bk else None

        def _assign(spk_id, data, sk):
            if sk is not None:
                if data["embedding"] is not None and sk in state["embeddings"]:
                    state["embeddings"][sk] = 0.7 * state["embeddings"][sk] + 0.3 * data["embedding"]
            else:
                sk = f"V{state['_next_id']}"
                state["_next_id"] += 1
                state["pitches"][sk] = data["pitch"]
                if data["embedding"] is not None:
                    state["embeddings"][sk] = data["embedding"]
            state["diar_history"].setdefault(spk_id, Counter())[sk] += 1
            return sk

        def _do_normal(spk_id, exclude=None, hist_pref=None):
            data = speaker_data[spk_id]
            m = _match_emb(data["embedding"], exclude, hist_pref)
            if m:
                return _assign(spk_id, data, m[0]), f"EMB({m[1]:.2f})"
            m2 = _match_pitch(data["pitch"], exclude)
            if m2:
                return _assign(spk_id, data, m2[0]), f"PITCH({m2[1]:.0f}Hz)"
            return _assign(spk_id, data, None), "NEW"

        chunk_results = []
        if not is_multi:
            for spk in speakers:
                v, mth = _do_normal(spk)
                chunk_results.append({
                    "chunk": ci, "cs": cs, "diar_spk": spk,
                    "assigned": v, "method": mth, "pitch": speaker_data[spk]["pitch"],
                })
        else:
            # Hybrid dominance
            if len(speakers) == 2:
                a, b = speakers
                da, db = speaker_data[a]["seg_duration"], speaker_data[b]["seg_duration"]
                ratio = max(da, db) / max(min(da, db), 0.01)
                if ratio >= duration_ratio_threshold:
                    dom = a if da >= db else b
                else:
                    dom = a if speaker_data[a]["seg_count"] >= speaker_data[b]["seg_count"] else b
            else:
                dom = max(speakers, key=lambda s: speaker_data[s]["seg_count"])

            # Pass 1: dominant
            dv, dm = _do_normal(dom)
            chunk_results.append({
                "chunk": ci, "cs": cs, "diar_spk": dom,
                "assigned": dv, "method": f"{dm}[dom]", "pitch": speaker_data[dom]["pitch"],
            })
            # Pass 2: secondary
            for spk in speakers:
                if spk == dom:
                    continue
                hist = state["diar_history"].get(spk, Counter())
                hp = hist.most_common(1)[0][0] if hist else None
                v, mth = _do_normal(spk, exclude=dv, hist_pref=hp)
                chunk_results.append({
                    "chunk": ci, "cs": cs, "diar_spk": spk,
                    "assigned": v, "method": mth, "pitch": speaker_data[spk]["pitch"],
                })

        all_results.extend(chunk_results)

    return all_results, state


def analyze(results, state, label):
    if not results:
        return {"label": label, "n_voices": 0, "distinct": False, "consistencies": {}}

    by_diar = {}
    for r in results:
        by_diar.setdefault(r["diar_spk"], []).append(r)

    n_voices = len(set(r["assigned"] for r in results))
    primaries = {}
    consistencies = {}
    for dk, entries in sorted(by_diar.items()):
        assignments = [e["assigned"] for e in entries]
        counts = Counter(assignments)
        mc, mc_c = counts.most_common(1)[0]
        primaries[dk] = mc
        consistencies[dk] = mc_c / len(assignments)

    distinct = len(set(primaries.values())) == len(primaries)

    methods = [r["method"] for r in results]
    emb_c = sum(1 for m in methods if "EMB" in m)
    pitch_c = sum(1 for m in methods if m.startswith("PITCH"))
    new_c = sum(1 for m in methods if "NEW" in m)

    print(f"\n  {'='*65}")
    print(f"  {label}")
    print(f"  {'='*65}")
    print(f"  Voices: {n_voices} | Distinct: {'YES' if distinct else 'NO'} | "
          f"Methods: {emb_c} emb, {pitch_c} pitch, {new_c} new")
    for dk in sorted(by_diar):
        entries = by_diar[dk]
        assignments = [e["assigned"] for e in entries]
        counts = Counter(assignments)
        pitches = [e["pitch"] for e in entries]
        detail = ", ".join(f"{k}:{v}" for k, v in counts.most_common())
        print(f"    {dk}: {consistencies[dk]:.0%} ({len(entries)} segs, "
              f"pitch={np.mean(pitches):.0f}±{np.std(pitches):.0f}Hz) [{detail}]")

    # Cross-voice embedding similarity
    if len(state["embeddings"]) >= 2:
        keys = list(state["embeddings"].keys())
        print(f"  Final embedding similarities:")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                sim = _cosine_sim(state["embeddings"][keys[i]], state["embeddings"][keys[j]])
                pitch_i = state["pitches"].get(keys[i], 0)
                pitch_j = state["pitches"].get(keys[j], 0)
                risk = " ⚠️HIGH" if sim > 0.35 else ""
                print(f"    {keys[i]}({pitch_i:.0f}Hz) <-> {keys[j]}({pitch_j:.0f}Hz): {sim:.3f}{risk}")

    return {
        "label": label, "n_voices": n_voices, "distinct": distinct,
        "consistencies": consistencies, "primaries": primaries,
        "emb_c": emb_c, "pitch_c": pitch_c, "new_c": new_c,
    }


def run_all():
    import soundfile as sf

    audio_path = "/tmp/long_audio_600s.wav"
    if not os.path.exists(audio_path):
        print("ERROR: /tmp/long_audio_600s.wav not found")
        return

    audio_np, sr = sf.read(audio_path)
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    total_sec = len(audio_np) / sr
    print(f"Audio: {total_sec:.1f}s ({total_sec/60:.1f} min), {sr}Hz")
    print("Loading models...")
    _get_diarizer()
    _get_diarizer()._load_spkrec()
    print("Models ready.\n")

    experiments = [
        ("BASELINE (thresh=0.40)", dict(emb_threshold=0.40)),
        ("thresh=0.45", dict(emb_threshold=0.45)),
        ("thresh=0.50", dict(emb_threshold=0.50)),
        ("thresh=0.55", dict(emb_threshold=0.55)),
        ("min_seg=1.0s", dict(min_seg_dur=1.0)),
        ("emb_gate=1.0s", dict(min_emb_dur=1.0)),
        ("thresh=0.45 + emb_gate=1.0", dict(emb_threshold=0.45, min_emb_dur=1.0)),
        ("thresh=0.50 + emb_gate=1.0", dict(emb_threshold=0.50, min_emb_dur=1.0)),
        ("thresh=0.45 + min_seg=1.0", dict(emb_threshold=0.45, min_seg_dur=1.0)),
        ("thresh=0.50 + min_seg=1.0", dict(emb_threshold=0.50, min_seg_dur=1.0)),
        ("margin=0.05", dict(emb_margin=0.05)),
        ("margin=0.05 + thresh=0.45", dict(emb_margin=0.05, emb_threshold=0.45)),
        ("sec_thresh=0.30", dict(secondary_emb_threshold=0.30)),
        ("sec_thresh=0.35", dict(secondary_emb_threshold=0.35)),
        ("hist_bonus=0.15", dict(history_bonus=0.15)),
        ("hist_bonus=0.05", dict(history_bonus=0.05)),
    ]

    all_metrics = []
    for label, kwargs in experiments:
        print(f"\n>>> Running: {label}")
        t0 = time.perf_counter()
        results, state = run_v7c(audio_tensor, sr, **kwargs)
        elapsed = time.perf_counter() - t0
        m = analyze(results, state, label)
        m["time"] = elapsed
        all_metrics.append(m)

    # Final table
    print(f"\n\n{'='*95}")
    print(f"  FINAL COMPARISON — 10 min sports commentary")
    print(f"{'='*95}")

    # Collect all diar speaker names
    all_diar = sorted(set(dk for m in all_metrics for dk in m.get("consistencies", {})))
    hdr = "  " + f"{'Experiment':<40} {'V':>3} {'Dist':>5}"
    for dk in all_diar:
        hdr += f" {dk[-5:]:>7}"
    hdr += f" {'Time':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for m in all_metrics:
        row = f"  {m['label']:<40} {m['n_voices']:>3} {'YES' if m['distinct'] else 'NO':>5}"
        for dk in all_diar:
            pct = m["consistencies"].get(dk, 0)
            row += f" {pct:>6.0%}"
        row += f" {m.get('time', 0):>5.1f}s"
        print(row)

    # Score and rank
    print(f"\n  RANKING (by geometric mean of consistencies, penalizing extra voices):")
    scored = []
    for m in all_metrics:
        c = m.get("consistencies", {})
        vals = list(c.values())
        if not vals:
            continue
        geo = np.prod(vals) ** (1.0 / len(vals))
        vp = 1.0 if m["n_voices"] <= len(c) else 0.9 ** (m["n_voices"] - len(c))
        dp = 1.0 if m.get("distinct") else 0.7
        score = geo * vp * dp
        scored.append((score, m["label"], m))
    scored.sort(reverse=True)
    for rank, (score, label, m) in enumerate(scored[:10], 1):
        c = m["consistencies"]
        avg = np.mean(list(c.values())) if c else 0
        print(f"  {rank:2d}. {label:<40} score={score:.3f} "
              f"(avg={avg:.0%}, voices={m['n_voices']}, distinct={m['distinct']})")


if __name__ == "__main__":
    run_all()
