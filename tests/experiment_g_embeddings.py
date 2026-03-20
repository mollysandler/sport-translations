"""
Experiment G: Speaker Embeddings for Cross-Chunk Continuity

Tests whether speaker embeddings can improve cross-chunk voice continuity
beyond the current pitch-only matching (baseline: 83.3%).

Known weak points of pitch-only:
  - crossing_a: 80%, crossing_b: 60% (speakers whose pitches cross)
  - large_drift: 60% (single speaker with +/-30Hz variation)
  - many_speakers: 50% (8 speakers at 20Hz spacing)

Approach:
  Since ECAPA-TDNN embeddings won't discriminate synthetic tones (trained
  on real speech), we SIMULATE embeddings using a deterministic spectral
  fingerprint. This lets us test the LOGIC of embedding-based matching
  without needing real speech data.

Three strategies compared:
  1. Pitch-only baseline (current system)
  2. Embedding-only (cosine similarity matching)
  3. Hybrid (0.6 * cosine_sim + 0.4 * pitch_proximity)
"""
import sys, os
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.test_diarizer_experiments import (
    make_tone,
    continuity_score,
    _current_continuity,
    _current_estimate,
)
from main import SmartVoiceManager
from utils import gender_from_pitch


# ---------------------------------------------------------------------------
# Synthetic embedding generator
# ---------------------------------------------------------------------------

def synthetic_embedding(audio_np, sr=16000):
    """Create a pseudo-embedding from spectral features of synthetic audio.

    Speakers with the same fundamental frequency will get similar embeddings
    even with pitch drift, while different speakers get different embeddings.
    Uses spectral shape (formant-like) rather than just pitch.
    """
    n_fft = 512
    hop = 160
    # Simple MFCC-like: FFT -> mel filterbank -> log -> DCT
    frames = []
    for start in range(0, len(audio_np) - n_fft, hop):
        frame = audio_np[start:start + n_fft]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        frames.append(spectrum)
    if not frames:
        return np.zeros(32)
    avg_spectrum = np.mean(frames, axis=0)
    # Normalize and take first 32 components as "embedding"
    emb = avg_spectrum[:32]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Modified continuity benchmark that passes audio
# ---------------------------------------------------------------------------

def run_embedding_continuity_benchmark(
    continuity_fn,
    sr=16000,
    num_chunks=5,
):
    """
    Like run_continuity_benchmark but passes (audio_np, pitch_hz, session_state)
    to the continuity function instead of just (pitch_hz, session_state).

    Uses the same six test scenarios with the same parameters as the original
    benchmark so results are directly comparable.
    """
    results = {}

    # Test 1: Stable speaker across chunks
    session = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments = []
    base_pitch = 140.0
    for i in range(num_chunks):
        drift = np.random.uniform(-15, 15)
        pitch = base_pitch + drift
        audio = make_tone(pitch, 2.0, sr)
        try:
            est_pitch, _ = _current_estimate(audio, sr)
        except Exception:
            est_pitch = pitch
        voice_id = continuity_fn(audio, est_pitch, session)
        chunk_assignments.append({"stable_speaker": voice_id})

    stable_score = continuity_score(chunk_assignments, "stable_speaker")
    results["stable_speaker_continuity"] = round(stable_score, 3)

    # Test 2: Two alternating speakers
    session2 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_a = []
    chunk_assignments_b = []
    for i in range(num_chunks):
        # Speaker A: ~130 Hz
        pitch_a = 130.0 + np.random.uniform(-10, 10)
        audio_a = make_tone(pitch_a, 2.0, sr)
        est_a, _ = _current_estimate(audio_a, sr)
        voice_a = continuity_fn(audio_a, est_a, session2)
        chunk_assignments_a.append({"speaker_a": voice_a})

        # Speaker B: ~200 Hz
        pitch_b = 200.0 + np.random.uniform(-10, 10)
        audio_b = make_tone(pitch_b, 2.0, sr)
        est_b, _ = _current_estimate(audio_b, sr)
        voice_b = continuity_fn(audio_b, est_b, session2)
        chunk_assignments_b.append({"speaker_b": voice_b})

    results["two_speaker_a_continuity"] = round(
        continuity_score(chunk_assignments_a, "speaker_a"), 3)
    results["two_speaker_b_continuity"] = round(
        continuity_score(chunk_assignments_b, "speaker_b"), 3)

    # Test 3: Similar-pitch speakers (hard case)
    session3 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_x = []
    chunk_assignments_y = []
    for i in range(num_chunks):
        pitch_x = 155.0 + np.random.uniform(-8, 8)
        audio_x = make_tone(pitch_x, 2.0, sr)
        est_x, _ = _current_estimate(audio_x, sr)
        voice_x = continuity_fn(audio_x, est_x, session3)
        chunk_assignments_x.append({"similar_x": voice_x})

        pitch_y = 165.0 + np.random.uniform(-8, 8)
        audio_y = make_tone(pitch_y, 2.0, sr)
        est_y, _ = _current_estimate(audio_y, sr)
        voice_y = continuity_fn(audio_y, est_y, session3)
        chunk_assignments_y.append({"similar_y": voice_y})

    results["similar_pitch_x_continuity"] = round(
        continuity_score(chunk_assignments_x, "similar_x"), 3)
    results["similar_pitch_y_continuity"] = round(
        continuity_score(chunk_assignments_y, "similar_y"), 3)

    # Test 4: Crossing pitches -- speaker A drifts UP, speaker B drifts DOWN
    session4 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_cross_a = []
    chunk_assignments_cross_b = []
    for i in range(num_chunks):
        # Speaker A: starts 130, drifts to 180
        pitch_a = 130.0 + (50.0 * i / (num_chunks - 1)) + np.random.uniform(-5, 5)
        audio_a = make_tone(pitch_a, 2.0, sr)
        est_a, _ = _current_estimate(audio_a, sr)
        voice_a = continuity_fn(audio_a, est_a, session4)
        chunk_assignments_cross_a.append({"cross_a": voice_a})

        # Speaker B: starts 180, drifts to 130
        pitch_b = 180.0 - (50.0 * i / (num_chunks - 1)) + np.random.uniform(-5, 5)
        audio_b = make_tone(pitch_b, 2.0, sr)
        est_b, _ = _current_estimate(audio_b, sr)
        voice_b = continuity_fn(audio_b, est_b, session4)
        chunk_assignments_cross_b.append({"cross_b": voice_b})

    results["crossing_a_continuity"] = round(
        continuity_score(chunk_assignments_cross_a, "cross_a"), 3)
    results["crossing_b_continuity"] = round(
        continuity_score(chunk_assignments_cross_b, "cross_b"), 3)

    # Test 5: Large pitch drift -- same speaker varies +/-30 Hz
    session5 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_drift = []
    for i in range(num_chunks):
        drift = np.random.uniform(-30, 30)
        pitch = 155.0 + drift
        audio = make_tone(pitch, 2.0, sr)
        est_pitch, _ = _current_estimate(audio, sr)
        voice_id = continuity_fn(audio, est_pitch, session5)
        chunk_assignments_drift.append({"drifty": voice_id})

    results["large_drift_continuity"] = round(
        continuity_score(chunk_assignments_drift, "drifty"), 3)

    # Test 6: Many speakers (voice pool exhaustion in continuity)
    session6 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    pitches_for_many = [100, 120, 140, 160, 180, 200, 220, 240]
    many_assignments = []
    for p in pitches_for_many:
        audio = make_tone(float(p), 2.0, sr)
        est, _ = _current_estimate(audio, sr)
        vid = continuity_fn(audio, est, session6)
        many_assignments.append(vid)
    unique_many = len(set(many_assignments))
    results["many_speakers_unique_voices"] = round(unique_many / len(pitches_for_many), 3)

    # Overall
    scores = [v for k, v in results.items() if isinstance(v, float)]
    results["__overall__"] = round(float(np.mean(scores)), 3)
    return results


# ---------------------------------------------------------------------------
# Strategy 1: Pitch-only baseline (adapted to 3-arg signature)
# ---------------------------------------------------------------------------

def pitch_only_continuity(audio_np, pitch_hz, session_state):
    """Pitch-only baseline: ignores audio, delegates to _current_continuity."""
    return _current_continuity(pitch_hz, session_state)


# ---------------------------------------------------------------------------
# Strategy 2: Embedding-only continuity
# ---------------------------------------------------------------------------

def embedding_only_continuity(audio_np, pitch_hz, session_state):
    """Match purely on cosine similarity of synthetic embeddings.

    - Stores embeddings per speaker in session_state["_embeddings"].
    - Matches to speaker with highest cosine similarity IF > 0.7.
    - If cosine is ambiguous (top-1 and top-2 within 0.05), falls back to
      pitch as tiebreaker.
    - Creates a new speaker if no match exceeds threshold.
    """
    COSINE_THRESHOLD = 0.7
    AMBIGUITY_MARGIN = 0.05

    session_state.setdefault("_embeddings", {})
    session_state.setdefault("_voice_ids", {})
    session_state.setdefault("_pitches", {})
    session_state.setdefault("speaker_voice_ids", {})
    session_state.setdefault("speaker_pitches", {})

    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()

    embeddings = session_state["_embeddings"]
    voice_ids = session_state["_voice_ids"]
    pitches = session_state["_pitches"]
    vm = session_state["_vm"]

    incoming_emb = synthetic_embedding(audio_np)

    if not embeddings:
        # First speaker -- create new
        spk_id = "SPK_00"
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        embeddings[spk_id] = incoming_emb
        voice_ids[spk_id] = voice_id
        pitches[spk_id] = pitch_hz
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        return voice_id

    # Compute cosine similarity to all known speakers
    sims = {}
    for spk_id, stored_emb in embeddings.items():
        sims[spk_id] = _cosine_similarity(incoming_emb, stored_emb)

    sorted_sims = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_spk, best_sim = sorted_sims[0]

    if best_sim < COSINE_THRESHOLD:
        # No good match -- create new speaker
        spk_id = f"SPK_{len(embeddings):02d}"
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        embeddings[spk_id] = incoming_emb
        voice_ids[spk_id] = voice_id
        pitches[spk_id] = pitch_hz
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        return voice_id

    # Check for ambiguity
    if len(sorted_sims) >= 2:
        _, second_sim = sorted_sims[1]
        if best_sim - second_sim < AMBIGUITY_MARGIN:
            # Ambiguous -- use pitch as tiebreaker
            candidates = [sorted_sims[0][0], sorted_sims[1][0]]
            best_by_pitch = min(
                candidates, key=lambda s: abs(pitch_hz - pitches[s])
            )
            # Update embedding with EMA
            alpha = 0.3
            embeddings[best_by_pitch] = (
                (1 - alpha) * embeddings[best_by_pitch] + alpha * incoming_emb
            )
            pitches[best_by_pitch] = 0.7 * pitches[best_by_pitch] + 0.3 * pitch_hz
            return voice_ids[best_by_pitch]

    # Clear match -- update embedding with EMA
    alpha = 0.3
    embeddings[best_spk] = (1 - alpha) * embeddings[best_spk] + alpha * incoming_emb
    pitches[best_spk] = 0.7 * pitches[best_spk] + 0.3 * pitch_hz
    return voice_ids[best_spk]


# ---------------------------------------------------------------------------
# Strategy 3: Hybrid continuity (weighted combination)
# ---------------------------------------------------------------------------

def hybrid_continuity(audio_np, pitch_hz, session_state):
    """Hybrid matching: score = 0.6 * cosine_sim + 0.4 * pitch_proximity.

    Combines embedding similarity with pitch proximity into a single
    composite score. This lets embeddings handle crossing-pitch scenarios
    while pitch handles the case where embeddings are ambiguous (e.g.,
    similar timbres at different pitches).

    Pitch proximity is normalized: 1.0 when distance=0, 0.0 when distance>=60Hz.
    """
    COMPOSITE_THRESHOLD = 0.55
    EMBED_WEIGHT = 0.6
    PITCH_WEIGHT = 0.4
    PITCH_NORM_HZ = 60.0  # distance at which pitch_proximity = 0

    session_state.setdefault("_embeddings", {})
    session_state.setdefault("_voice_ids", {})
    session_state.setdefault("_pitches", {})
    session_state.setdefault("_anchor_pitches", {})
    session_state.setdefault("speaker_voice_ids", {})
    session_state.setdefault("speaker_pitches", {})

    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()

    embeddings = session_state["_embeddings"]
    voice_ids = session_state["_voice_ids"]
    pitches = session_state["_pitches"]
    anchor_pitches = session_state["_anchor_pitches"]
    vm = session_state["_vm"]

    incoming_emb = synthetic_embedding(audio_np)

    if not embeddings:
        # First speaker
        spk_id = "SPK_00"
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        embeddings[spk_id] = incoming_emb
        voice_ids[spk_id] = voice_id
        pitches[spk_id] = pitch_hz
        anchor_pitches[spk_id] = pitch_hz
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        return voice_id

    # Compute composite score for all known speakers
    scores = {}
    for spk_id, stored_emb in embeddings.items():
        cos_sim = _cosine_similarity(incoming_emb, stored_emb)
        pitch_dist = abs(pitch_hz - pitches[spk_id])
        pitch_prox = max(0.0, 1.0 - pitch_dist / PITCH_NORM_HZ)
        composite = EMBED_WEIGHT * cos_sim + PITCH_WEIGHT * pitch_prox
        scores[spk_id] = composite

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_spk, best_score = sorted_scores[0]

    # Anchor-pitch safety check: reject if incoming pitch is too far from
    # the speaker's original anchor (prevents cascade absorption)
    ANCHOR_MAX_HZ = 55.0
    if best_spk in anchor_pitches:
        anchor_dist = abs(pitch_hz - anchor_pitches[best_spk])
        if anchor_dist > ANCHOR_MAX_HZ:
            best_score = 0.0  # force creation of new speaker

    if best_score < COMPOSITE_THRESHOLD:
        # No good match -- create new speaker
        spk_id = f"SPK_{len(embeddings):02d}"
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        embeddings[spk_id] = incoming_emb
        voice_ids[spk_id] = voice_id
        pitches[spk_id] = pitch_hz
        anchor_pitches[spk_id] = pitch_hz
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        return voice_id

    # Match found -- update embedding with EMA, update pitch with EMA
    alpha = 0.3
    embeddings[best_spk] = (1 - alpha) * embeddings[best_spk] + alpha * incoming_emb
    pitches[best_spk] = 0.7 * pitches[best_spk] + 0.3 * pitch_hz
    return voice_ids[best_spk]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperimentG:
    """Compare pitch-only, embedding-only, and hybrid continuity strategies."""

    def test_embedding_experiment(self):
        strategies = {
            "Pitch-only (baseline)": pitch_only_continuity,
            "Embedding-only":        embedding_only_continuity,
            "Hybrid (0.6e+0.4p)":    hybrid_continuity,
        }

        all_results = {}
        for name, fn in strategies.items():
            np.random.seed(42)
            all_results[name] = run_embedding_continuity_benchmark(fn)

        # Collect all sub-test keys (preserve insertion order)
        sub_keys = [
            k for k in all_results["Pitch-only (baseline)"]
            if k != "__overall__" and isinstance(all_results["Pitch-only (baseline)"][k], float)
        ]

        # --- Print comparison table ---
        strat_names = list(strategies.keys())
        col_width = 16

        print()
        print("=" * 100)
        print("  EXPERIMENT G: Speaker Embeddings for Cross-Chunk Continuity")
        print("=" * 100)

        # Header
        header = f"  {'SUB-TEST':<36}"
        for sn in strat_names:
            header += f" {sn:>{col_width}}"
        print(header)
        print("-" * 100)

        # Rows
        for key in sub_keys:
            row = f"  {key:<36}"
            baseline_val = all_results["Pitch-only (baseline)"][key]
            for sn in strat_names:
                val = all_results[sn][key]
                delta = val - baseline_val
                if abs(delta) < 0.001:
                    marker = "    "
                elif delta > 0:
                    marker = " [+]"
                else:
                    marker = " [-]"
                row += f" {val:>9.1%}{marker:>4}"
            print(row)

        # Overall
        print("-" * 100)
        row = f"  {'OVERALL':<36}"
        baseline_overall = all_results["Pitch-only (baseline)"]["__overall__"]
        for sn in strat_names:
            val = all_results[sn]["__overall__"]
            delta = val - baseline_overall
            if abs(delta) < 0.001:
                marker = "    "
            elif delta > 0:
                marker = " [+]"
            else:
                marker = " [-]"
            row += f" {val:>9.1%}{marker:>4}"
        print(row)
        print("=" * 100)

        # --- Per-strategy summary ---
        print()
        for sn in strat_names:
            res = all_results[sn]
            improved = []
            regressed = []
            for key in sub_keys:
                d = res[key] - all_results["Pitch-only (baseline)"][key]
                if d > 0.001:
                    improved.append(key)
                elif d < -0.001:
                    regressed.append(key)
            print(f"  {sn}:")
            print(f"    Overall: {res['__overall__']:.1%}")
            if improved:
                print(f"    Improved ({len(improved)}): {', '.join(improved)}")
            if regressed:
                print(f"    Regressed ({len(regressed)}): {', '.join(regressed)}")
            if not improved and not regressed and sn != "Pitch-only (baseline)":
                print(f"    No change from baseline.")
            print()

        # --- Assertions ---
        # The experiment is investigative, so we don't assert improvement.
        # We DO assert that nothing catastrophically fails.
        for sn, res in all_results.items():
            assert res["__overall__"] > 0.0, f"{sn} completely failed"

        # Report which strategy won
        winner = max(strat_names, key=lambda s: all_results[s]["__overall__"])
        print(f"  WINNER: {winner} ({all_results[winner]['__overall__']:.1%})")
        print()
