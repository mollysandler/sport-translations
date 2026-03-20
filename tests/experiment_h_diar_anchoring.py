"""
Experiment H: Diarizer-Aware Embedding Matching

Two improvements over baseline embedding+pitch matching:

1. **Intra-chunk contrast**: If pyannote labels two segments differently within
   the same chunk, they MUST map to different session voices. Use the diarizer's
   within-chunk speaker labels as a constraint.

2. **New-speaker bias for unmatched secondary speakers**: When embedding similarity
   is below threshold for all stored voices AND the diarizer says this is a
   different speaker than the chunk's dominant speaker, create a new voice
   instead of falling back to pitch (which often collides).

Run:
    python -m pytest tests/experiment_h_diar_anchoring.py -v -s
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


def _load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    device = torch.device("cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    return model, device


def _load_diarizer():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "real_diarizer", os.path.join(_project_root, "diarizer.py")
        )
        real_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_mod)
        return real_mod.SportsDiarizer()
    except Exception as e:
        print(f"  Could not load SportsDiarizer: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Matching strategies
# ─────────────────────────────────────────────────────────────────────

def match_baseline_emb_pitch(chunk_segments, state):
    """Baseline: embedding match -> pitch fallback -> new. (from previous test)"""
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03

    results = []
    for seg in chunk_segments:
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = "NEW"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        results.append({**seg, "assigned": assigned, "method": method})
    return results


def match_diar_anchored(chunk_segments, state):
    """Experiment H: Diarizer-aware matching.

    Improvement 1 — intra-chunk contrast:
      After assigning all segments in a chunk, if two segments with different
      diarizer labels got the same session voice, reassign the minority one.

    Improvement 2 — new-speaker bias:
      When embedding sim is below threshold AND this segment's diarizer label
      is a secondary speaker in the chunk (not the dominant one), skip pitch
      fallback and create a new voice.
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03

    # Count diarizer labels in this chunk to find dominant speaker
    diar_counts = Counter(s["diar_spk"] for s in chunk_segments)
    dominant_diar = diar_counts.most_common(1)[0][0] if diar_counts else None

    results = []
    for seg in chunk_segments:
        emb, pitch = seg["emb"], seg["pitch"]
        diar_spk = seg["diar_spk"]
        is_secondary = diar_spk != dominant_diar

        assigned = None
        method = "NEW"

        # Step 1: Try embedding match
        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        # Step 2: Pitch fallback — but NOT for secondary speakers
        if assigned is None and not is_secondary:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        # Step 3: New voice (secondary speakers skip pitch, land here more often)
        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = f"NEW{'(secondary)' if is_secondary else ''}"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        results.append({**seg, "assigned": assigned, "method": method})

    # ── Intra-chunk contrast pass ──
    # Group by diarizer label. If two different diar labels map to the same
    # session voice, reassign the minority group to their next-best option.
    by_diar = {}
    for r in results:
        by_diar.setdefault(r["diar_spk"], []).append(r)

    if len(by_diar) >= 2:
        # Find the dominant session assignment in this chunk
        all_assigned = [r["assigned"] for r in results]
        dominant_voice = Counter(all_assigned).most_common(1)[0][0]

        for diar_spk, segs in by_diar.items():
            if diar_spk == dominant_diar:
                continue  # don't touch the dominant speaker
            for r in segs:
                if r["assigned"] == dominant_voice:
                    # This secondary segment collided with the dominant voice.
                    # Try to find a different stored voice via embedding.
                    emb = r["emb"]
                    sims = {k: _cosine_sim(emb, v)
                            for k, v in state["embeddings"].items()
                            if k != dominant_voice}
                    if sims:
                        best_alt_k = max(sims, key=sims.get)
                        best_alt_sim = sims[best_alt_k]
                        if best_alt_sim >= 0.20:  # lower bar for reassignment
                            r["assigned"] = best_alt_k
                            r["method"] += f"->REMAP({best_alt_k},sim={best_alt_sim:.2f})"
                            continue

                    # No good alternative — create a new voice
                    new_id = f"V{len(state['pitches'])}"
                    state["pitches"][new_id] = r["pitch"]
                    state["embeddings"][new_id] = emb
                    r["assigned"] = new_id
                    r["method"] += f"->SPLIT({new_id})"

    return results


def match_diar_anchored_v2(chunk_segments, state):
    """Experiment H v2: Same as v1 but with a stricter remap threshold
    and pitch fallback allowed for secondary speakers when they have a
    DIFFERENT best pitch match than the dominant speaker's voice."""
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03

    diar_counts = Counter(s["diar_spk"] for s in chunk_segments)
    dominant_diar = diar_counts.most_common(1)[0][0] if diar_counts else None

    # Pre-assign dominant speaker's voice for this chunk (if already known)
    dominant_voice = None

    results = []

    # Process dominant speaker segments first, then secondary
    ordered = sorted(chunk_segments, key=lambda s: s["diar_spk"] != dominant_diar)

    for seg in ordered:
        emb, pitch = seg["emb"], seg["pitch"]
        diar_spk = seg["diar_spk"]
        is_secondary = diar_spk != dominant_diar

        assigned = None
        method = "NEW"

        # Step 1: Embedding match
        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                # Secondary speakers can't match the dominant voice
                if is_secondary and best_k == dominant_voice:
                    # Try second best
                    if len(sorted_sims) > 1:
                        alt_k, alt_sim = sorted_sims[1]
                        if alt_sim >= EMB_THRESHOLD:
                            assigned = alt_k
                            method = f"EMB-ALT({alt_sim:.2f},avoided={best_k})"
                    # else: fall through to pitch/new
                else:
                    assigned = best_k
                    method = f"EMB({best_sim:.2f})"

        # Step 2: Pitch fallback
        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                # Secondary speakers can't pitch-match to the dominant voice
                if is_secondary and best_p == dominant_voice:
                    pass  # skip — force new voice
                else:
                    assigned = best_p
                    method = f"PITCH({best_dist:.0f}Hz)"

        # Step 3: New voice
        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = f"NEW{'(sec)' if is_secondary else ''}"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        # Track dominant voice for this chunk
        if not is_secondary and dominant_voice is None:
            dominant_voice = assigned

        results.append({**seg, "assigned": assigned, "method": method})

    # Re-sort by time order for display
    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v3(chunk_segments, state):
    """Experiment H v3: Dominant-voice exclusion + secondary-voice consolidation.

    Key ideas:
    1. Process dominant speaker first to establish the chunk's dominant voice.
    2. Secondary speakers CANNOT match the dominant voice (forced separation).
    3. Secondary speakers get a LOWER embedding threshold (0.20) to match
       previously-seen secondary voices (consolidation pool).
    4. Also try pitch matching against non-dominant voices.
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    SECONDARY_EMB_THRESHOLD = 0.20  # lower bar for secondary consolidation

    diar_counts = Counter(s["diar_spk"] for s in chunk_segments)
    dominant_diar = diar_counts.most_common(1)[0][0] if diar_counts else None

    dominant_voice = None
    results = []

    # Process dominant speaker first
    ordered = sorted(chunk_segments, key=lambda s: s["diar_spk"] != dominant_diar)

    for seg in ordered:
        emb, pitch = seg["emb"], seg["pitch"]
        diar_spk = seg["diar_spk"]
        is_secondary = diar_spk != dominant_diar

        assigned = None
        method = "NEW"

        if state["embeddings"]:
            if is_secondary and dominant_voice:
                # Only consider non-dominant voices, with lower threshold
                sims = {k: _cosine_sim(emb, v)
                        for k, v in state["embeddings"].items()
                        if k != dominant_voice}
                if sims:
                    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                    best_k, best_sim = sorted_sims[0]
                    if best_sim >= SECONDARY_EMB_THRESHOLD:
                        assigned = best_k
                        method = f"SEC-EMB({best_sim:.2f})"
            else:
                # Normal embedding match for dominant speaker
                sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
                sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                best_k, best_sim = sorted_sims[0]
                second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
                if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                    assigned = best_k
                    method = f"EMB({best_sim:.2f})"

        # Pitch fallback
        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                # Secondary speakers skip the dominant voice
                if is_secondary and spk == dominant_voice:
                    continue
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = f"NEW{'(sec)' if is_secondary else ''}"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        if not is_secondary and dominant_voice is None:
            dominant_voice = assigned

        results.append({**seg, "assigned": assigned, "method": method})

    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v4(chunk_segments, state):
    """Experiment H v4: Like v3 but with two refinements:

    1. Track which voices were created as "secondary" — these form a
       consolidation pool that gets checked with lower thresholds.
    2. Use the highest embedding sim among secondary voices, even if it's
       below the normal threshold, as long as it's the clear best match
       (margin > 0.05 over other secondary voices).
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    SECONDARY_EMB_THRESHOLD = 0.15
    SECONDARY_MARGIN = 0.05

    if "secondary_voices" not in state:
        state["secondary_voices"] = set()

    diar_counts = Counter(s["diar_spk"] for s in chunk_segments)
    dominant_diar = diar_counts.most_common(1)[0][0] if diar_counts else None
    dominant_voice = None
    results = []

    ordered = sorted(chunk_segments, key=lambda s: s["diar_spk"] != dominant_diar)

    for seg in ordered:
        emb, pitch = seg["emb"], seg["pitch"]
        diar_spk = seg["diar_spk"]
        is_secondary = diar_spk != dominant_diar

        assigned = None
        method = "NEW"

        if state["embeddings"]:
            if is_secondary and dominant_voice:
                # First: check secondary voice pool
                sec_sims = {k: _cosine_sim(emb, v)
                            for k, v in state["embeddings"].items()
                            if k in state["secondary_voices"]}
                if sec_sims:
                    sorted_sec = sorted(sec_sims.items(), key=lambda x: x[1], reverse=True)
                    best_k, best_sim = sorted_sec[0]
                    second_sim = sorted_sec[1][1] if len(sorted_sec) > 1 else 0.0
                    if best_sim >= SECONDARY_EMB_THRESHOLD and (best_sim - second_sim) >= SECONDARY_MARGIN:
                        assigned = best_k
                        method = f"SEC-POOL({best_sim:.2f},m={best_sim-second_sim:.2f})"

                # Second: check all non-dominant with normal threshold
                if assigned is None:
                    other_sims = {k: _cosine_sim(emb, v)
                                  for k, v in state["embeddings"].items()
                                  if k != dominant_voice}
                    if other_sims:
                        sorted_other = sorted(other_sims.items(), key=lambda x: x[1], reverse=True)
                        best_k, best_sim = sorted_other[0]
                        if best_sim >= SECONDARY_EMB_THRESHOLD:
                            assigned = best_k
                            method = f"SEC-EMB({best_sim:.2f})"
            else:
                sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
                sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                best_k, best_sim = sorted_sims[0]
                second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
                if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                    assigned = best_k
                    method = f"EMB({best_sim:.2f})"

        # Pitch fallback (skip dominant voice for secondary)
        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                if is_secondary and spk == dominant_voice:
                    continue
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            if is_secondary:
                state["secondary_voices"].add(assigned)
            method = f"NEW{'(sec)' if is_secondary else ''}"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        if not is_secondary and dominant_voice is None:
            dominant_voice = assigned

        results.append({**seg, "assigned": assigned, "method": method})

    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v5(chunk_segments, state):
    """Experiment H v5: Fixes both root causes from v4.

    Fix 1: Duration-based dominance (not segment count).
    Fix 2: Two-pass — assign dominant segments first, find most-common
           voice, THEN assign secondary segments with that voice excluded.
    Fix 3: Raise secondary pool threshold to 0.25 (0.15 was too permissive).
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    SECONDARY_EMB_THRESHOLD = 0.25

    if "secondary_voices" not in state:
        state["secondary_voices"] = set()

    # Duration-weighted dominance
    dur_by_spk = {}
    for s in chunk_segments:
        dur_by_spk[s["diar_spk"]] = dur_by_spk.get(s["diar_spk"], 0) + s["dur"]
    dominant_diar = max(dur_by_spk, key=dur_by_spk.get) if dur_by_spk else None

    dominant_segs = [s for s in chunk_segments if s["diar_spk"] == dominant_diar]
    secondary_segs = [s for s in chunk_segments if s["diar_spk"] != dominant_diar]

    results = []

    def _assign_normal(seg):
        """Normal embedding+pitch matching (no exclusions)."""
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = "NEW"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        return assigned, method

    # ── Pass 1: Assign dominant speaker segments normally ──
    dom_results = []
    for seg in dominant_segs:
        assigned, method = _assign_normal(seg)
        dom_results.append({**seg, "assigned": assigned, "method": method})

    # Find the dominant voice (most common assignment among dominant segments)
    dom_assignments = [r["assigned"] for r in dom_results]
    if dom_assignments:
        dominant_voice = Counter(dom_assignments).most_common(1)[0][0]
    else:
        dominant_voice = None

    results.extend(dom_results)

    # ── Pass 2: Assign secondary segments with dominant voice excluded ──
    for seg in secondary_segs:
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"] and dominant_voice:
            # Check secondary voice pool first (with moderate threshold)
            sec_sims = {k: _cosine_sim(emb, v)
                        for k, v in state["embeddings"].items()
                        if k in state["secondary_voices"]}
            if sec_sims:
                best_k = max(sec_sims, key=sec_sims.get)
                best_sim = sec_sims[best_k]
                if best_sim >= SECONDARY_EMB_THRESHOLD:
                    assigned = best_k
                    method = f"SEC-POOL({best_sim:.2f})"

            # Check all non-dominant voices
            if assigned is None:
                other_sims = {k: _cosine_sim(emb, v)
                              for k, v in state["embeddings"].items()
                              if k != dominant_voice}
                if other_sims:
                    best_k = max(other_sims, key=other_sims.get)
                    best_sim = other_sims[best_k]
                    if best_sim >= SECONDARY_EMB_THRESHOLD:
                        assigned = best_k
                        method = f"SEC-EMB({best_sim:.2f})"

            # Pitch fallback — exclude dominant voice
            if assigned is None:
                best_dist = float("inf")
                best_p = None
                for spk, stored_p in state["pitches"].items():
                    if spk == dominant_voice:
                        continue
                    dist = abs(pitch - stored_p)
                    if dist < best_dist:
                        best_dist = dist
                        best_p = spk
                if best_p is not None and best_dist <= 30.0:
                    assigned = best_p
                    method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            state["secondary_voices"].add(assigned)
            method = "NEW(sec)"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        results.append({**seg, "assigned": assigned, "method": method})

    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v6(chunk_segments, state):
    """Experiment H v6: v5 + diarizer label history.

    Adds: track which session voice each diarizer label has been assigned
    to across all chunks. Use this history as a soft bias — if SPEAKER_01
    has been consistently assigned V2, prefer V2 even with weak embeddings.
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    SECONDARY_EMB_THRESHOLD = 0.25
    HISTORY_BONUS = 0.10  # boost for historically associated voice

    if "secondary_voices" not in state:
        state["secondary_voices"] = set()
    if "diar_history" not in state:
        state["diar_history"] = {}  # diar_label -> Counter of session voices

    # Duration-weighted dominance
    dur_by_spk = {}
    for s in chunk_segments:
        dur_by_spk[s["diar_spk"]] = dur_by_spk.get(s["diar_spk"], 0) + s["dur"]
    dominant_diar = max(dur_by_spk, key=dur_by_spk.get) if dur_by_spk else None

    dominant_segs = [s for s in chunk_segments if s["diar_spk"] == dominant_diar]
    secondary_segs = [s for s in chunk_segments if s["diar_spk"] != dominant_diar]

    results = []

    def _assign_normal(seg):
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = "NEW"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        return assigned, method

    # ── Pass 1: dominant ──
    dom_results = []
    for seg in dominant_segs:
        assigned, method = _assign_normal(seg)
        dom_results.append({**seg, "assigned": assigned, "method": method})

    dom_assignments = [r["assigned"] for r in dom_results]
    dominant_voice = Counter(dom_assignments).most_common(1)[0][0] if dom_assignments else None
    results.extend(dom_results)

    # ── Pass 2: secondary with exclusion + history bias ──
    for seg in secondary_segs:
        emb, pitch = seg["emb"], seg["pitch"]
        diar_spk = seg["diar_spk"]
        assigned = None
        method = "NEW"

        # Get historical preference for this diarizer label
        hist = state["diar_history"].get(diar_spk, Counter())
        hist_pref = hist.most_common(1)[0][0] if hist else None

        if state["embeddings"] and dominant_voice:
            # Compute sims to all non-dominant voices
            other_sims = {k: _cosine_sim(emb, v)
                          for k, v in state["embeddings"].items()
                          if k != dominant_voice}

            if other_sims:
                # Apply history bonus
                boosted = {}
                for k, sim in other_sims.items():
                    bonus = HISTORY_BONUS if k == hist_pref else 0.0
                    boosted[k] = sim + bonus

                best_k = max(boosted, key=boosted.get)
                raw_sim = other_sims[best_k]
                boosted_sim = boosted[best_k]

                if boosted_sim >= SECONDARY_EMB_THRESHOLD:
                    assigned = best_k
                    bonus_str = f"+hist" if best_k == hist_pref else ""
                    method = f"SEC({raw_sim:.2f}{bonus_str})"

            # Pitch fallback — exclude dominant
            if assigned is None:
                best_dist = float("inf")
                best_p = None
                for spk, stored_p in state["pitches"].items():
                    if spk == dominant_voice:
                        continue
                    dist = abs(pitch - stored_p)
                    if dist < best_dist:
                        best_dist = dist
                        best_p = spk
                if best_p is not None and best_dist <= 30.0:
                    assigned = best_p
                    method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            state["secondary_voices"].add(assigned)
            method = "NEW(sec)"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        results.append({**seg, "assigned": assigned, "method": method})

    # Update diarizer label history
    for r in results:
        hist = state["diar_history"].setdefault(r["diar_spk"], Counter())
        hist[r["assigned"]] += 1

    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v7(chunk_segments, state):
    """Experiment H v7: Count-based dominance + two-pass + history bias.

    Combines v4's count-based dominance (better early stability for SPEAKER_00)
    with v6's history bias (consolidates SPEAKER_01 across chunks).
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    SECONDARY_EMB_THRESHOLD = 0.25
    HISTORY_BONUS = 0.10

    if "secondary_voices" not in state:
        state["secondary_voices"] = set()
    if "diar_history" not in state:
        state["diar_history"] = {}

    # Count-based dominance (not duration)
    diar_counts = Counter(s["diar_spk"] for s in chunk_segments)
    dominant_diar = diar_counts.most_common(1)[0][0] if diar_counts else None

    dominant_segs = [s for s in chunk_segments if s["diar_spk"] == dominant_diar]
    secondary_segs = [s for s in chunk_segments if s["diar_spk"] != dominant_diar]

    results = []

    def _assign_normal(seg):
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = "NEW"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        return assigned, method

    # ── Pass 1: dominant ──
    dom_results = []
    for seg in dominant_segs:
        assigned, method = _assign_normal(seg)
        dom_results.append({**seg, "assigned": assigned, "method": method})

    dom_assignments = [r["assigned"] for r in dom_results]
    dominant_voice = Counter(dom_assignments).most_common(1)[0][0] if dom_assignments else None
    results.extend(dom_results)

    # ── Pass 2: secondary with exclusion + history bias ──
    for seg in secondary_segs:
        emb, pitch = seg["emb"], seg["pitch"]
        diar_spk = seg["diar_spk"]
        assigned = None
        method = "NEW"

        hist = state["diar_history"].get(diar_spk, Counter())
        hist_pref = hist.most_common(1)[0][0] if hist else None

        if state["embeddings"] and dominant_voice:
            other_sims = {k: _cosine_sim(emb, v)
                          for k, v in state["embeddings"].items()
                          if k != dominant_voice}

            if other_sims:
                boosted = {}
                for k, sim in other_sims.items():
                    bonus = HISTORY_BONUS if k == hist_pref else 0.0
                    boosted[k] = sim + bonus

                best_k = max(boosted, key=boosted.get)
                raw_sim = other_sims[best_k]
                boosted_sim = boosted[best_k]

                if boosted_sim >= SECONDARY_EMB_THRESHOLD:
                    assigned = best_k
                    bonus_str = "+hist" if best_k == hist_pref else ""
                    method = f"SEC({raw_sim:.2f}{bonus_str})"

            if assigned is None:
                best_dist = float("inf")
                best_p = None
                for spk, stored_p in state["pitches"].items():
                    if spk == dominant_voice:
                        continue
                    dist = abs(pitch - stored_p)
                    if dist < best_dist:
                        best_dist = dist
                        best_p = spk
                if best_p is not None and best_dist <= 30.0:
                    assigned = best_p
                    method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            state["secondary_voices"].add(assigned)
            method = "NEW(sec)"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        results.append({**seg, "assigned": assigned, "method": method})

    for r in results:
        hist = state["diar_history"].setdefault(r["diar_spk"], Counter())
        hist[r["assigned"]] += 1

    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v7b(chunk_segments, state):
    """Experiment H v7b: v7 + only apply secondary logic in multi-speaker chunks.

    In single-speaker chunks (pyannote found only 1 speaker), use normal
    baseline matching for everyone — no dominance, no exclusion.
    Secondary path only activates when the chunk genuinely has 2+ diarizer labels.
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    SECONDARY_EMB_THRESHOLD = 0.25
    HISTORY_BONUS = 0.10

    if "secondary_voices" not in state:
        state["secondary_voices"] = set()
    if "diar_history" not in state:
        state["diar_history"] = {}

    unique_diar_labels = set(s["diar_spk"] for s in chunk_segments)
    is_multi_speaker_chunk = len(unique_diar_labels) >= 2

    results = []

    def _assign_normal(seg):
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = "NEW"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        return assigned, method

    if not is_multi_speaker_chunk:
        # Single-speaker chunk: normal matching, no dominance logic
        for seg in chunk_segments:
            assigned, method = _assign_normal(seg)
            results.append({**seg, "assigned": assigned, "method": method})
    else:
        # Multi-speaker chunk: two-pass with count-based dominance + history
        diar_counts = Counter(s["diar_spk"] for s in chunk_segments)
        dominant_diar = diar_counts.most_common(1)[0][0]

        dominant_segs = [s for s in chunk_segments if s["diar_spk"] == dominant_diar]
        secondary_segs = [s for s in chunk_segments if s["diar_spk"] != dominant_diar]

        # Pass 1: dominant segments
        dom_results = []
        for seg in dominant_segs:
            assigned, method = _assign_normal(seg)
            dom_results.append({**seg, "assigned": assigned, "method": method})

        dom_assignments = [r["assigned"] for r in dom_results]
        dominant_voice = Counter(dom_assignments).most_common(1)[0][0] if dom_assignments else None
        results.extend(dom_results)

        # Pass 2: secondary with exclusion + history
        for seg in secondary_segs:
            emb, pitch = seg["emb"], seg["pitch"]
            diar_spk = seg["diar_spk"]
            assigned = None
            method = "NEW"

            hist = state["diar_history"].get(diar_spk, Counter())
            hist_pref = hist.most_common(1)[0][0] if hist else None

            if state["embeddings"] and dominant_voice:
                other_sims = {k: _cosine_sim(emb, v)
                              for k, v in state["embeddings"].items()
                              if k != dominant_voice}

                if other_sims:
                    boosted = {}
                    for k, sim in other_sims.items():
                        bonus = HISTORY_BONUS if k == hist_pref else 0.0
                        boosted[k] = sim + bonus

                    best_k = max(boosted, key=boosted.get)
                    raw_sim = other_sims[best_k]
                    boosted_sim = boosted[best_k]

                    if boosted_sim >= SECONDARY_EMB_THRESHOLD:
                        assigned = best_k
                        bonus_str = "+hist" if best_k == hist_pref else ""
                        method = f"SEC({raw_sim:.2f}{bonus_str})"

                if assigned is None:
                    best_dist = float("inf")
                    best_p = None
                    for spk, stored_p in state["pitches"].items():
                        if spk == dominant_voice:
                            continue
                        dist = abs(pitch - stored_p)
                        if dist < best_dist:
                            best_dist = dist
                            best_p = spk
                    if best_p is not None and best_dist <= 30.0:
                        assigned = best_p
                        method = f"PITCH({best_dist:.0f}Hz)"

            if assigned is None:
                assigned = f"V{len(state['pitches'])}"
                state["pitches"][assigned] = pitch
                state["embeddings"][assigned] = emb
                state["secondary_voices"].add(assigned)
                method = "NEW(sec)"
            else:
                if assigned in state["embeddings"]:
                    state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

            results.append({**seg, "assigned": assigned, "method": method})

    # Update history
    for r in results:
        hist = state["diar_history"].setdefault(r["diar_spk"], Counter())
        hist[r["assigned"]] += 1

    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v7c(chunk_segments, state):
    """Experiment H v7c: Hybrid dominance + history bias.

    Dominance rule: if one speaker has >2x the total duration of the other,
    use duration (catches long single-segment speakers). Otherwise use count
    (more stable in balanced chunks).
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    SECONDARY_EMB_THRESHOLD = 0.25
    HISTORY_BONUS = 0.10
    DURATION_RATIO_THRESHOLD = 2.0

    if "secondary_voices" not in state:
        state["secondary_voices"] = set()
    if "diar_history" not in state:
        state["diar_history"] = {}

    unique_diar_labels = set(s["diar_spk"] for s in chunk_segments)
    is_multi_speaker_chunk = len(unique_diar_labels) >= 2

    results = []

    def _assign_normal(seg):
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = "NEW"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        return assigned, method

    if not is_multi_speaker_chunk:
        for seg in chunk_segments:
            assigned, method = _assign_normal(seg)
            results.append({**seg, "assigned": assigned, "method": method})
    else:
        # Hybrid dominance: duration if decisive, count otherwise
        dur_by_spk = {}
        count_by_spk = Counter()
        for s in chunk_segments:
            dur_by_spk[s["diar_spk"]] = dur_by_spk.get(s["diar_spk"], 0) + s["dur"]
            count_by_spk[s["diar_spk"]] += 1

        spk_list = list(dur_by_spk.keys())
        if len(spk_list) == 2:
            a, b = spk_list
            dur_a, dur_b = dur_by_spk[a], dur_by_spk[b]
            ratio = max(dur_a, dur_b) / max(min(dur_a, dur_b), 0.01)
            if ratio >= DURATION_RATIO_THRESHOLD:
                # Duration is decisive
                dominant_diar = a if dur_a >= dur_b else b
            else:
                # Close in duration — use count
                dominant_diar = count_by_spk.most_common(1)[0][0]
        else:
            # 3+ speakers: use count
            dominant_diar = count_by_spk.most_common(1)[0][0]

        dominant_segs = [s for s in chunk_segments if s["diar_spk"] == dominant_diar]
        secondary_segs = [s for s in chunk_segments if s["diar_spk"] != dominant_diar]

        # Pass 1: dominant
        dom_results = []
        for seg in dominant_segs:
            assigned, method = _assign_normal(seg)
            dom_results.append({**seg, "assigned": assigned, "method": method})

        dom_assignments = [r["assigned"] for r in dom_results]
        dominant_voice = Counter(dom_assignments).most_common(1)[0][0] if dom_assignments else None
        results.extend(dom_results)

        # Pass 2: secondary with exclusion + history
        for seg in secondary_segs:
            emb, pitch = seg["emb"], seg["pitch"]
            diar_spk = seg["diar_spk"]
            assigned = None
            method = "NEW"

            hist = state["diar_history"].get(diar_spk, Counter())
            hist_pref = hist.most_common(1)[0][0] if hist else None

            if state["embeddings"] and dominant_voice:
                other_sims = {k: _cosine_sim(emb, v)
                              for k, v in state["embeddings"].items()
                              if k != dominant_voice}

                if other_sims:
                    boosted = {}
                    for k, sim in other_sims.items():
                        bonus = HISTORY_BONUS if k == hist_pref else 0.0
                        boosted[k] = sim + bonus

                    best_k = max(boosted, key=boosted.get)
                    raw_sim = other_sims[best_k]
                    boosted_sim = boosted[best_k]

                    if boosted_sim >= SECONDARY_EMB_THRESHOLD:
                        assigned = best_k
                        bonus_str = "+hist" if best_k == hist_pref else ""
                        method = f"SEC({raw_sim:.2f}{bonus_str})"

                if assigned is None:
                    best_dist = float("inf")
                    best_p = None
                    for spk, stored_p in state["pitches"].items():
                        if spk == dominant_voice:
                            continue
                        dist = abs(pitch - stored_p)
                        if dist < best_dist:
                            best_dist = dist
                            best_p = spk
                    if best_p is not None and best_dist <= 30.0:
                        assigned = best_p
                        method = f"PITCH({best_dist:.0f}Hz)"

            if assigned is None:
                assigned = f"V{len(state['pitches'])}"
                state["pitches"][assigned] = pitch
                state["embeddings"][assigned] = emb
                state["secondary_voices"].add(assigned)
                method = "NEW(sec)"
            else:
                if assigned in state["embeddings"]:
                    state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

            results.append({**seg, "assigned": assigned, "method": method})

    for r in results:
        hist = state["diar_history"].setdefault(r["diar_spk"], Counter())
        hist[r["assigned"]] += 1

    results.sort(key=lambda r: r["start"])
    return results


def match_diar_anchored_v8(chunk_segments, state):
    """Experiment H v8: No dominance calculation at all.

    Simpler approach:
    1. Assign ALL segments with normal baseline matching.
    2. Post-assignment correction: if two different diarizer labels in the
       same chunk got the same voice, check history to see if the minority
       label has a different preferred voice. If so, reassign.
    3. If no history, create a new voice for the minority.
    """
    EMB_THRESHOLD = 0.40
    EMB_MARGIN = 0.03
    HISTORY_BONUS = 0.10

    if "diar_history" not in state:
        state["diar_history"] = {}

    results = []

    # ── Step 1: Normal baseline assignment for everyone ──
    for seg in chunk_segments:
        emb, pitch = seg["emb"], seg["pitch"]
        assigned = None
        method = "NEW"

        if state["embeddings"]:
            sims = {k: _cosine_sim(emb, v) for k, v in state["embeddings"].items()}
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_k, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
            if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                assigned = best_k
                method = f"EMB({best_sim:.2f})"

        if assigned is None:
            best_dist = float("inf")
            best_p = None
            for spk, stored_p in state["pitches"].items():
                dist = abs(pitch - stored_p)
                if dist < best_dist:
                    best_dist = dist
                    best_p = spk
            if best_p is not None and best_dist <= 30.0:
                assigned = best_p
                method = f"PITCH({best_dist:.0f}Hz)"

        if assigned is None:
            assigned = f"V{len(state['pitches'])}"
            state["pitches"][assigned] = pitch
            state["embeddings"][assigned] = emb
            method = "NEW"
        else:
            if assigned in state["embeddings"]:
                state["embeddings"][assigned] = 0.7 * state["embeddings"][assigned] + 0.3 * emb

        results.append({**seg, "assigned": assigned, "method": method})

    # ── Step 2: Post-assignment collision fix ──
    by_diar = {}
    for r in results:
        by_diar.setdefault(r["diar_spk"], []).append(r)

    if len(by_diar) >= 2:
        # Find the voice assigned to the most segments (dominant voice)
        voice_counts = Counter(r["assigned"] for r in results)
        dominant_voice = voice_counts.most_common(1)[0][0]

        # Find which diar label "owns" the dominant voice
        dominant_diar = None
        for diar_spk, segs in by_diar.items():
            dom_count = sum(1 for s in segs if s["assigned"] == dominant_voice)
            if dominant_diar is None or dom_count > by_diar.get(dominant_diar, []):
                # Pick the diar label with the most dominant_voice assignments
                pass
        # Simpler: diar label with highest total duration
        dur_by_diar = {}
        for diar_spk, segs in by_diar.items():
            dur_by_diar[diar_spk] = sum(s["dur"] for s in segs)
        dominant_diar = max(dur_by_diar, key=dur_by_diar.get)

        for diar_spk, segs in by_diar.items():
            if diar_spk == dominant_diar:
                continue
            for r in segs:
                if r["assigned"] != dominant_voice:
                    continue
                # This secondary segment collided with the dominant voice.
                # Check history for preferred voice
                hist = state["diar_history"].get(diar_spk, Counter())
                hist_pref = hist.most_common(1)[0][0] if hist else None

                if hist_pref and hist_pref != dominant_voice and hist_pref in state["embeddings"]:
                    # Reassign to historical preference
                    old = r["assigned"]
                    r["assigned"] = hist_pref
                    r["method"] += f"->HIST({hist_pref})"
                    # Update embedding
                    state["embeddings"][hist_pref] = (
                        0.7 * state["embeddings"][hist_pref] + 0.3 * r["emb"])
                else:
                    # No history — find best non-dominant voice or create new
                    emb = r["emb"]
                    other_sims = {k: _cosine_sim(emb, v)
                                  for k, v in state["embeddings"].items()
                                  if k != dominant_voice}
                    if other_sims:
                        best_k = max(other_sims, key=other_sims.get)
                        best_sim = other_sims[best_k]
                        if best_sim >= 0.20:
                            r["assigned"] = best_k
                            r["method"] += f"->ALT({best_k},{best_sim:.2f})"
                            continue

                    # Create new
                    new_id = f"V{len(state['pitches'])}"
                    state["pitches"][new_id] = r["pitch"]
                    state["embeddings"][new_id] = emb
                    r["assigned"] = new_id
                    r["method"] += f"->SPLIT({new_id})"

    # Update history
    for r in results:
        hist = state["diar_history"].setdefault(r["diar_spk"], Counter())
        hist[r["assigned"]] += 1

    results.sort(key=lambda r: r["start"])
    return results


# ─────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────

class TestExperimentH:

    def test_compare_strategies(self):
        """Run all strategies on 3min of real sports audio."""
        import soundfile as sf
        audio_path = "/tmp/long_audio_180s.wav"
        if not os.path.exists(audio_path):
            import pytest; pytest.skip("Run ffmpeg extraction first")

        audio_np, sr = sf.read(audio_path)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        total_sec = len(audio_np) / sr

        from main import _estimate_pitch_safe

        diarizer = _load_diarizer()
        if diarizer is None:
            import pytest; pytest.skip("Diarizer not available")
        model, device = _load_ecapa()

        CHUNK_SEC = 10.0
        num_chunks = int(total_sec / CHUNK_SEC)

        # ── Phase 1: Diarize + extract features ──
        print(f"\n  Diarizing {num_chunks} chunks ({total_sec:.0f}s)...")
        all_chunks = []

        for chunk_idx in range(num_chunks):
            cs = chunk_idx * CHUNK_SEC
            chunk_audio = audio_tensor[:, int(cs * sr):int((cs + CHUNK_SEC) * sr)]

            segments = diarizer.diarize(chunk_audio, sr)
            seg_data = []
            for seg in segments:
                if seg.end_sec - seg.start_sec < 0.5:
                    continue
                s, e = int(seg.start_sec * sr), int(seg.end_sec * sr)
                seg_audio = chunk_audio[:, s:e]
                seg_np = seg_audio.squeeze().numpy().astype(np.float32)

                try:
                    pitch, _ = _estimate_pitch_safe(seg_np, sr)
                except:
                    pitch = 150.0

                wav = seg_audio.to(device).float()
                with torch.no_grad():
                    emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()

                seg_data.append({
                    "diar_spk": seg.speaker_id,
                    "start": seg.start_sec + cs,
                    "end": seg.end_sec + cs,
                    "dur": seg.end_sec - seg.start_sec,
                    "pitch": pitch,
                    "emb": emb,
                })

            all_chunks.append(seg_data)
            n_spk = len(set(s["diar_spk"] for s in seg_data))
            print(f"    Chunk {chunk_idx:2d} [{cs:5.0f}-{cs+CHUNK_SEC:5.0f}s]: "
                  f"{len(seg_data)} segs, {n_spk} spk")

        total_segs = sum(len(c) for c in all_chunks)
        print(f"  Total: {total_segs} segments\n")

        # ── Phase 2: Run each strategy ──
        strategies = [
            ("Baseline (emb+pitch)", match_baseline_emb_pitch),
            ("H-v7: count dom + history", match_diar_anchored_v7),
            ("H-v7c: hybrid dom + history", match_diar_anchored_v7c),
        ]

        all_results = {}

        for name, match_fn in strategies:
            print(f"{'='*74}")
            print(f"  {name}")
            print(f"{'='*74}")

            state = {"pitches": {}, "embeddings": {}}
            full_log = []

            for chunk_idx, chunk_segs in enumerate(all_chunks):
                # Deep-copy segments so each strategy gets fresh data
                import copy
                segs_copy = copy.deepcopy(chunk_segs)
                results = match_fn(segs_copy, state)
                full_log.extend(results)

                for r in results:
                    m_display = r["method"][:58]
                    print(f"    [{r['start']:6.1f}-{r['end']:6.1f}s] "
                          f"{r['diar_spk']:>11} p={r['pitch']:3.0f} -> {r['assigned']:>4} [{m_display}]")

            all_results[name] = full_log
            print()

        # ── Phase 3: Compare ──
        print(f"\n{'='*74}")
        print(f"  COMPARISON")
        print(f"{'='*74}")

        for name, log in all_results.items():
            print(f"\n  {name}:")

            by_diar = {}
            for r in log:
                by_diar.setdefault(r["diar_spk"], []).append(r["assigned"])

            session_spks = set()
            for diar_spk, ids in sorted(by_diar.items()):
                counts = Counter(ids)
                mc, mc_n = counts.most_common(1)[0]
                consistency = mc_n / len(ids)
                session_spks.update(ids)
                detail = ", ".join(f"{k}:{v}" for k, v in counts.most_common())
                print(f"    {diar_spk:>11}: {len(ids):2d} segs -> [{detail}] "
                      f"({consistency:.0%})")

            # Check collision
            primaries = {}
            for diar_spk, ids in by_diar.items():
                primaries[diar_spk] = Counter(ids).most_common(1)[0][0]
            distinct = len(set(primaries.values())) == len(primaries)

            print(f"    Voices: {len(session_spks)} | "
                  f"Distinct primaries: {'YES' if distinct else 'NO (collision)'}")

            # Method breakdown
            methods = [r["method"] for r in log]
            emb_n = sum(1 for m in methods if m.startswith("EMB"))
            pitch_n = sum(1 for m in methods if m.startswith("PITCH"))
            new_n = sum(1 for m in methods if m.startswith("NEW"))
            remap_n = sum(1 for m in methods if "REMAP" in m or "SPLIT" in m)
            alt_n = sum(1 for m in methods if "ALT" in m)
            extra = ""
            if remap_n:
                extra += f", {remap_n} remapped/split"
            if alt_n:
                extra += f", {alt_n} alt-emb"
            print(f"    Methods: {emb_n} emb, {pitch_n} pitch, {new_n} new{extra}")
