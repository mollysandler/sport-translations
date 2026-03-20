"""
Experiment K: Diarizer-level improvements.

Tests modifications to the diarization step itself (before v7c voice assignment):
  1. Wider diarization window — give pyannote more context
  2. Cross-chunk label anchoring — relabel output using prev chunk embeddings
  3. Combined

All experiments feed into the same v7c voice assignment and measure
voice stability on 10 minutes of sports commentary.
"""
import sys, os, time
import numpy as np
import torch
from collections import Counter, defaultdict

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


# ======================================================================
# Diarizer wrappers — each implements a different strategy
# ======================================================================

def diarize_baseline(audio_tensor, sr, chunk_start_sec, chunk_sec):
    """Standard: diarize exactly the chunk window."""
    diarizer = _get_diarizer()
    s = int(chunk_start_sec * sr)
    e = int((chunk_start_sec + chunk_sec) * sr)
    chunk = audio_tensor[:, s:e]
    return diarizer.diarize(chunk, sr), chunk


def diarize_wider_window(audio_tensor, sr, chunk_start_sec, chunk_sec,
                         context_before=5.0):
    """Diarize a wider window (context_before + chunk_sec), return only
    segments that fall within the target chunk region."""
    diarizer = _get_diarizer()
    total_sec = audio_tensor.shape[1] / sr

    # Wider window: [chunk_start - context_before, chunk_start + chunk_sec]
    window_start = max(0, chunk_start_sec - context_before)
    window_end = min(total_sec, chunk_start_sec + chunk_sec)
    ws = int(window_start * sr)
    we = int(window_end * sr)
    wide_chunk = audio_tensor[:, ws:we]

    segments = diarizer.diarize(wide_chunk, sr)

    # Filter: only keep segments overlapping with the target region
    # Target region in wide_chunk coordinates:
    target_start_in_wide = chunk_start_sec - window_start
    target_end_in_wide = target_start_in_wide + chunk_sec

    filtered = []
    for seg in segments:
        # Segment must overlap with target region
        overlap_start = max(seg.start_sec, target_start_in_wide)
        overlap_end = min(seg.end_sec, target_end_in_wide)
        if overlap_end - overlap_start >= 0.3:  # at least 0.3s overlap
            from diarizer import SpeakerSegment
            # Clip to target region and offset to chunk-local coordinates
            new_start = max(0, seg.start_sec - target_start_in_wide)
            new_end = min(chunk_sec, seg.end_sec - target_start_in_wide)
            if new_end - new_start >= 0.3:
                filtered.append(SpeakerSegment(
                    speaker_id=seg.speaker_id,
                    start_ms=int(new_start * 1000),
                    end_ms=int(new_end * 1000),
                    start_sec=new_start,
                    end_sec=new_end,
                ))

    # Return the target chunk audio (not the wide window)
    cs = int(chunk_start_sec * sr)
    ce = int((chunk_start_sec + chunk_sec) * sr)
    target_chunk = audio_tensor[:, cs:ce]
    return filtered, target_chunk


def anchor_labels(segments, chunk_audio, sr, prev_embeddings, anchor_threshold=0.55):
    """Cross-chunk label anchoring: relabel speakers to match previous chunk's
    speaker embeddings based on cosine similarity.

    prev_embeddings: {speaker_label: embedding} from previous chunk.
    Returns (relabeled_segments, new_embeddings).
    """
    if not segments:
        return segments, prev_embeddings

    diarizer = _get_diarizer()
    diarizer._load_spkrec()

    # Extract embedding per speaker in this chunk
    by_spk = defaultdict(list)
    for seg in segments:
        by_spk[seg.speaker_id].append(seg)

    chunk_embeddings = {}
    for spk, segs in by_spk.items():
        collected = []
        total_dur = 0.0
        for seg in segs:
            s = int(seg.start_sec * sr)
            e = int(seg.end_sec * sr)
            dur = (e - s) / sr
            if dur >= 0.3:
                collected.append(chunk_audio[:, s:e])
                total_dur += dur
            if total_dur >= 5.0:
                break
        if collected and total_dur >= 0.5:
            combined = torch.cat(collected, dim=1)
            chunk_embeddings[spk] = _extract_embedding(combined)

    if not prev_embeddings or not chunk_embeddings:
        return segments, chunk_embeddings

    # Match current speakers to previous speakers via embedding similarity
    # Use Hungarian-style greedy matching
    label_map = {}
    used_prev = set()

    # Sort by total duration (match most confident first)
    spk_by_dur = sorted(chunk_embeddings.keys(),
                        key=lambda s: sum(seg.end_sec - seg.start_sec
                                          for seg in by_spk[s]),
                        reverse=True)

    for cur_spk in spk_by_dur:
        cur_emb = chunk_embeddings[cur_spk]
        best_prev = None
        best_sim = -1.0
        for prev_spk, prev_emb in prev_embeddings.items():
            if prev_spk in used_prev:
                continue
            sim = _cosine_sim(cur_emb, prev_emb)
            if sim > best_sim:
                best_sim = sim
                best_prev = prev_spk

        if best_prev is not None and best_sim >= anchor_threshold:
            label_map[cur_spk] = best_prev
            used_prev.add(best_prev)

    # Relabel segments
    if label_map:
        from diarizer import SpeakerSegment
        relabeled = []
        for seg in segments:
            new_id = label_map.get(seg.speaker_id, seg.speaker_id)
            relabeled.append(SpeakerSegment(
                speaker_id=new_id,
                start_ms=seg.start_ms, end_ms=seg.end_ms,
                start_sec=seg.start_sec, end_sec=seg.end_sec,
            ))
        segments = relabeled

    # Update embeddings: use EMA for matched, add new for unmatched
    new_embeddings = dict(prev_embeddings)
    for cur_spk, cur_emb in chunk_embeddings.items():
        final_label = label_map.get(cur_spk, cur_spk)
        if final_label in new_embeddings:
            new_embeddings[final_label] = 0.7 * new_embeddings[final_label] + 0.3 * cur_emb
        else:
            new_embeddings[final_label] = cur_emb

    return segments, new_embeddings


# ======================================================================
# V7C voice assignment (same as experiment J)
# ======================================================================

def v7c_assign(speaker_data, state, is_multi, speakers,
               emb_threshold=0.40, emb_margin=0.03,
               secondary_emb_threshold=0.25, history_bonus=0.10,
               duration_ratio_threshold=2.0, pitch_match_hz=30.0):
    """Run v7c assignment on pre-extracted speaker data."""

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
                return bk, cands[bk]
        else:
            ss = sorted(cands.items(), key=lambda x: x[1], reverse=True)
            bk, bs = ss[0]
            sec = ss[1][1] if len(ss) > 1 else 0.0
            if bs >= emb_threshold and (bs - sec) >= emb_margin:
                return bk, bs
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

    def _do_normal(spk, exclude=None, hist_pref=None):
        data = speaker_data[spk]
        m = _match_emb(data["embedding"], exclude, hist_pref)
        if m:
            return _assign(spk, data, m[0]), f"EMB({m[1]:.2f})"
        m2 = _match_pitch(data["pitch"], exclude)
        if m2:
            return _assign(spk, data, m2[0]), f"PITCH({m2[1]:.0f}Hz)"
        return _assign(spk, data, None), "NEW"

    results = []
    if not is_multi:
        for spk in speakers:
            v, mth = _do_normal(spk)
            results.append((spk, v, mth))
    else:
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

        dv, dm = _do_normal(dom)
        results.append((dom, dv, f"{dm}[dom]"))

        for spk in speakers:
            if spk == dom:
                continue
            hist = state["diar_history"].get(spk, Counter())
            hp = hist.most_common(1)[0][0] if hist else None
            v, mth = _do_normal(spk, exclude=dv, hist_pref=hp)
            results.append((spk, v, mth))

    return results


# ======================================================================
# Full pipeline runner
# ======================================================================

def run_experiment(audio_tensor, sr, *, label, diarize_fn, use_anchoring=False,
                   anchor_threshold=0.55, chunk_sec=10.0):
    """Run full pipeline: diarize → (optional anchor) → extract features → v7c assign."""
    total_sec = audio_tensor.shape[1] / sr
    num_chunks = int(total_sec / chunk_sec)

    state = {"embeddings": {}, "pitches": {}, "diar_history": {}, "_next_id": 0}
    prev_embeddings = {}
    all_results = []

    label_stability = []  # track (chunk_idx, diar_labels)

    for ci in range(num_chunks):
        cs = ci * chunk_sec

        # Diarize
        segments, chunk_audio = diarize_fn(audio_tensor, sr, cs, chunk_sec)

        # Optional cross-chunk anchoring
        if use_anchoring and segments:
            segments, prev_embeddings = anchor_labels(
                segments, chunk_audio, sr, prev_embeddings,
                anchor_threshold=anchor_threshold)
        elif not use_anchoring and segments:
            # Still track embeddings for comparison (but don't relabel)
            pass

        if not segments:
            continue

        # Extract features per speaker
        spk_groups = defaultdict(list)
        for seg in segments:
            spk_groups[seg.speaker_id].append(seg)

        speaker_data = {}
        for spk_id, segs in spk_groups.items():
            collected, total_dur = [], 0.0
            for seg in segs:
                s, e = int(seg.start_sec * sr), int(seg.end_sec * sr)
                dur = (e - s) / sr
                if dur >= 0.1:
                    collected.append(chunk_audio[:, s:e])
                    total_dur += dur
                if total_dur >= 5.0:
                    break
            if not collected or total_dur < 0.5:
                continue
            combined = torch.cat(collected, dim=1)
            audio_np = combined.squeeze().detach().cpu().numpy().astype(np.float32)
            pitch, pr = _estimate_pitch(audio_np, sr)
            emb = _extract_embedding(combined)
            speaker_data[spk_id] = {
                "pitch": pitch, "embedding": emb,
                "seg_count": len(segs), "seg_duration": total_dur,
            }

        speakers = [s for s in sorted(spk_groups) if s in speaker_data]
        if not speakers:
            continue

        is_multi = len(speakers) >= 2
        chunk_results = v7c_assign(speaker_data, state, is_multi, speakers)

        for spk, voice, method in chunk_results:
            all_results.append({
                "chunk": ci, "cs": cs, "diar_spk": spk,
                "assigned": voice, "method": method,
                "pitch": speaker_data[spk]["pitch"],
            })

        label_stability.append((ci, sorted(set(seg.speaker_id for seg in segments))))

    return all_results, state, label_stability


def analyze(results, state, label_stability, label):
    if not results:
        return {"label": label, "n_voices": 0}

    by_diar = defaultdict(list)
    for r in results:
        by_diar[r["diar_spk"]].append(r)

    n_voices = len(set(r["assigned"] for r in results))
    n_diar_labels = len(by_diar)

    primaries = {}
    consistencies = {}
    for dk, entries in sorted(by_diar.items()):
        assignments = [e["assigned"] for e in entries]
        counts = Counter(assignments)
        mc, mc_c = counts.most_common(1)[0]
        primaries[dk] = mc
        consistencies[dk] = mc_c / len(assignments)

    distinct = len(set(primaries.values())) == len(primaries)

    # Voice-centric: for each voice, how many diar labels map to it?
    voice_to_diar = defaultdict(set)
    for dk, entries in by_diar.items():
        for e in entries:
            voice_to_diar[e["assigned"]].add(dk)

    # Voice run stability
    runs = []
    prev_voice = None
    prev_start = 0
    for r in sorted(results, key=lambda r: r["cs"]):
        if r["diar_spk"] == "SPEAKER_00":
            if r["assigned"] != prev_voice:
                if prev_voice is not None:
                    runs.append((prev_voice, prev_start, r["cs"]))
                prev_voice = r["assigned"]
                prev_start = r["cs"]
    if prev_voice:
        runs.append((prev_voice, prev_start, results[-1]["cs"] + 10))

    stable_time = sum(e - s for v, s, e in runs if e - s >= 20)
    total_time = results[-1]["cs"] + 10 if results else 1

    # Diar label stability
    unique_labels_per_chunk = [len(labels) for ci, labels in label_stability]
    n_unique_diar = len(set(l for ci, labels in label_stability for l in labels))

    methods = [r["method"] for r in results]
    emb_c = sum(1 for m in methods if "EMB" in m)
    pitch_c = sum(1 for m in methods if m.startswith("PITCH"))
    new_c = sum(1 for m in methods if "NEW" in m)

    print(f"\n  {'='*65}")
    print(f"  {label}")
    print(f"  {'='*65}")
    print(f"  Voices: {n_voices} | Diar labels: {n_diar_labels} | "
          f"Distinct: {'YES' if distinct else 'NO'}")
    print(f"  Methods: {emb_c} emb, {pitch_c} pitch, {new_c} new")
    print(f"  Voice switches (SPEAKER_00): {len(runs)-1} | "
          f"Stable time (runs>=20s): {stable_time}s/{total_time:.0f}s "
          f"({stable_time/total_time:.0%})")

    for dk in sorted(by_diar):
        entries = by_diar[dk]
        assignments = [e["assigned"] for e in entries]
        counts = Counter(assignments)
        pitches = [e["pitch"] for e in entries]
        detail = ", ".join(f"{k}:{v}" for k, v in counts.most_common())
        print(f"    {dk}: {consistencies[dk]:.0%} ({len(entries)} segs, "
              f"pitch={np.mean(pitches):.0f}±{np.std(pitches):.0f}Hz) [{detail}]")

    if len(state["embeddings"]) >= 2:
        keys = sorted(state["embeddings"].keys())
        print(f"  Cross-voice similarities:")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                sim = _cosine_sim(state["embeddings"][keys[i]],
                                  state["embeddings"][keys[j]])
                pi = state["pitches"].get(keys[i], 0)
                pj = state["pitches"].get(keys[j], 0)
                print(f"    {keys[i]}({pi:.0f}Hz) <-> {keys[j]}({pj:.0f}Hz): {sim:.3f}")

    return {
        "label": label, "n_voices": n_voices, "n_diar_labels": n_diar_labels,
        "distinct": distinct, "consistencies": consistencies,
        "voice_switches": len(runs) - 1, "stable_pct": stable_time / total_time,
        "emb_c": emb_c, "pitch_c": pitch_c, "new_c": new_c,
    }


# ======================================================================
# Main
# ======================================================================

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
    _get_diarizer()._load_spkrec()
    print("Models ready.\n")

    all_metrics = []

    # 1. BASELINE
    print("\n>>> 1. BASELINE (10s chunks, no anchoring)")
    r, s, ls = run_experiment(audio_tensor, sr, label="BASELINE",
                              diarize_fn=diarize_baseline)
    all_metrics.append(analyze(r, s, ls, "BASELINE"))

    # 2. WIDER WINDOW 5s context
    print("\n>>> 2. WIDER WINDOW (5s context before)")
    def diarize_wide5(at, sr, cs, chunk_sec):
        return diarize_wider_window(at, sr, cs, chunk_sec, context_before=5.0)
    r, s, ls = run_experiment(audio_tensor, sr, label="WIDE-5s",
                              diarize_fn=diarize_wide5)
    all_metrics.append(analyze(r, s, ls, "WIDER WINDOW (5s context)"))

    # 3. WIDER WINDOW 10s context
    print("\n>>> 3. WIDER WINDOW (10s context before)")
    def diarize_wide10(at, sr, cs, chunk_sec):
        return diarize_wider_window(at, sr, cs, chunk_sec, context_before=10.0)
    r, s, ls = run_experiment(audio_tensor, sr, label="WIDE-10s",
                              diarize_fn=diarize_wide10)
    all_metrics.append(analyze(r, s, ls, "WIDER WINDOW (10s context)"))

    # 4. CROSS-CHUNK ANCHORING (threshold=0.55)
    print("\n>>> 4. ANCHORING (threshold=0.55)")
    r, s, ls = run_experiment(audio_tensor, sr, label="ANCHOR-0.55",
                              diarize_fn=diarize_baseline,
                              use_anchoring=True, anchor_threshold=0.55)
    all_metrics.append(analyze(r, s, ls, "ANCHORING (thresh=0.55)"))

    # 5. CROSS-CHUNK ANCHORING (threshold=0.45)
    print("\n>>> 5. ANCHORING (threshold=0.45)")
    r, s, ls = run_experiment(audio_tensor, sr, label="ANCHOR-0.45",
                              diarize_fn=diarize_baseline,
                              use_anchoring=True, anchor_threshold=0.45)
    all_metrics.append(analyze(r, s, ls, "ANCHORING (thresh=0.45)"))

    # 6. CROSS-CHUNK ANCHORING (threshold=0.35)
    print("\n>>> 6. ANCHORING (threshold=0.35)")
    r, s, ls = run_experiment(audio_tensor, sr, label="ANCHOR-0.35",
                              diarize_fn=diarize_baseline,
                              use_anchoring=True, anchor_threshold=0.35)
    all_metrics.append(analyze(r, s, ls, "ANCHORING (thresh=0.35)"))

    # 7. WIDE 5s + ANCHORING 0.55
    print("\n>>> 7. WIDE 5s + ANCHORING 0.55")
    r, s, ls = run_experiment(audio_tensor, sr, label="WIDE5+ANCHOR55",
                              diarize_fn=diarize_wide5,
                              use_anchoring=True, anchor_threshold=0.55)
    all_metrics.append(analyze(r, s, ls, "WIDE 5s + ANCHOR 0.55"))

    # 8. WIDE 5s + ANCHORING 0.45
    print("\n>>> 8. WIDE 5s + ANCHORING 0.45")
    r, s, ls = run_experiment(audio_tensor, sr, label="WIDE5+ANCHOR45",
                              diarize_fn=diarize_wide5,
                              use_anchoring=True, anchor_threshold=0.45)
    all_metrics.append(analyze(r, s, ls, "WIDE 5s + ANCHOR 0.45"))

    # 9. WIDE 10s + ANCHORING 0.55
    print("\n>>> 9. WIDE 10s + ANCHORING 0.55")
    r, s, ls = run_experiment(audio_tensor, sr, label="WIDE10+ANCHOR55",
                              diarize_fn=diarize_wide10,
                              use_anchoring=True, anchor_threshold=0.55)
    all_metrics.append(analyze(r, s, ls, "WIDE 10s + ANCHOR 0.55"))

    # 10. WIDE 10s + ANCHORING 0.45
    print("\n>>> 10. WIDE 10s + ANCHORING 0.45")
    r, s, ls = run_experiment(audio_tensor, sr, label="WIDE10+ANCHOR45",
                              diarize_fn=diarize_wide10,
                              use_anchoring=True, anchor_threshold=0.45)
    all_metrics.append(analyze(r, s, ls, "WIDE 10s + ANCHOR 0.45"))

    # ==================================================================
    # FINAL COMPARISON
    # ==================================================================
    print(f"\n\n{'='*100}")
    print(f"  FINAL COMPARISON — Diarizer Improvements")
    print(f"{'='*100}")
    print(f"  {'Experiment':<35} {'V':>3} {'Dist':>5} {'Switches':>8} "
          f"{'Stable%':>8} {'SPK_00':>7} {'SPK_01':>7} {'EmbM':>5} {'PitM':>5} {'New':>4}")
    print(f"  {'-'*35} {'-'*3} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*4}")

    for m in all_metrics:
        c = m.get("consistencies", {})
        s00 = c.get("SPEAKER_00", 0)
        s01 = c.get("SPEAKER_01", 0)
        dist = "YES" if m.get("distinct") else "NO"
        sw = m.get("voice_switches", 0)
        sp = m.get("stable_pct", 0)
        print(f"  {m['label']:<35} {m['n_voices']:>3} {dist:>5} {sw:>8} "
              f"{sp:>7.0%} {s00:>6.0%} {s01:>6.0%} "
              f"{m.get('emb_c',0):>5} {m.get('pitch_c',0):>5} {m.get('new_c',0):>4}")

    # Rank by composite score
    print(f"\n  RANKING:")
    scored = []
    for m in all_metrics:
        c = m.get("consistencies", {})
        vals = list(c.values())
        if not vals:
            continue
        avg_c = np.mean(vals)
        sp = m.get("stable_pct", 0)
        sw_penalty = max(0, 1.0 - m.get("voice_switches", 0) * 0.03)
        dp = 1.0 if m.get("distinct") else 0.7
        score = (0.4 * avg_c + 0.4 * sp + 0.2 * sw_penalty) * dp
        scored.append((score, m))
    scored.sort(reverse=True)

    for rank, (score, m) in enumerate(scored, 1):
        c = m.get("consistencies", {})
        print(f"  {rank:2d}. {m['label']:<35} score={score:.3f} "
              f"(voices={m['n_voices']}, switches={m.get('voice_switches',0)}, "
              f"stable={m.get('stable_pct',0):.0%})")


if __name__ == "__main__":
    run_all()
