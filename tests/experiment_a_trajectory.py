"""
Experiment A: Pitch Trajectory Extrapolation

Instead of matching incoming pitch to stored (static) pitch per speaker,
store a history of recent pitch observations per speaker and use linear
extrapolation to predict where each speaker's pitch SHOULD be in the
next chunk.  This helps when two speakers' pitches cross (e.g. one
trending up while the other trends down).

Hypothesis: crossing-pitch continuity rises from 60% to 80%+ without
regressing stable / two-speaker / similar-pitch scenarios.
"""
import sys, os
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.test_diarizer_experiments import (
    run_continuity_benchmark,
    _current_continuity,
    _current_estimate,
    continuity_score,
    SmartVoiceManager,
    gender_from_pitch,
)


# ---------------------------------------------------------------------------
# Experiment A: trajectory-based continuity
# ---------------------------------------------------------------------------

def trajectory_continuity(pitch_hz: float, session_state: dict) -> str:
    """
    Continuity function that stores a pitch *history* per speaker and
    uses linear extrapolation to predict the next expected pitch.

    State kept in session_state:
        speaker_voice_ids   – {spk_id: voice_id}
        speaker_pitches     – {spk_id: ema_pitch}  (for legacy compat)
        speaker_pitch_history – {spk_id: [p0, p1, ...]}
        _vm                 – persistent SmartVoiceManager
    """
    THRESHOLD = 35.0
    EMA_ALPHA_OLD = 0.7
    EMA_ALPHA_NEW = 0.3
    MAX_HISTORY = 6  # how many past observations to keep

    # Ensure state dicts exist
    session_state.setdefault("speaker_voice_ids", {})
    session_state.setdefault("speaker_pitches", {})
    session_state.setdefault("speaker_pitch_history", {})
    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()

    vm = session_state["_vm"]
    pitches = session_state["speaker_pitches"]
    histories = session_state["speaker_pitch_history"]
    voice_ids = session_state["speaker_voice_ids"]

    # --- predict expected pitch for every known speaker ----------------
    def _extrapolate(spk_id: str) -> float:
        """Return predicted next pitch for *spk_id*."""
        hist = histories.get(spk_id, [])
        if len(hist) < 2:
            # Not enough data — fall back to stored EMA pitch
            return pitches[spk_id]
        # Simple linear extrapolation from the last two observations
        # (more robust than full linear regression for small N and
        # avoids numpy.linalg overhead).
        # Use weighted: if we have >= 3 points, average the last two deltas
        if len(hist) >= 3:
            delta1 = hist[-1] - hist[-2]
            delta2 = hist[-2] - hist[-3]
            avg_delta = 0.6 * delta1 + 0.4 * delta2  # bias toward recent
        else:
            avg_delta = hist[-1] - hist[-2]
        predicted = hist[-1] + avg_delta
        return predicted

    # --- find closest speaker by EXTRAPOLATED pitch --------------------
    best_match = None
    best_dist = float("inf")

    for spk_id in pitches:
        expected = _extrapolate(spk_id)
        dist = abs(pitch_hz - expected)
        if dist < best_dist:
            best_dist = dist
            best_match = spk_id

    if best_match is not None and best_dist <= THRESHOLD:
        # Update history
        histories.setdefault(best_match, []).append(pitch_hz)
        if len(histories[best_match]) > MAX_HISTORY:
            histories[best_match] = histories[best_match][-MAX_HISTORY:]
        # Update EMA pitch
        pitches[best_match] = (
            EMA_ALPHA_OLD * pitches[best_match] + EMA_ALPHA_NEW * pitch_hz
        )
        return voice_ids[best_match]
    else:
        # New speaker — assign a fresh voice from the PERSISTENT manager
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        spk_id = f"SPK_{len(voice_ids):02d}"
        voice_ids[spk_id] = voice_id
        pitches[spk_id] = pitch_hz
        histories[spk_id] = [pitch_hz]
        return voice_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperimentA:
    """Compare baseline (_current_continuity) vs Experiment A (trajectory)."""

    def test_trajectory_vs_baseline(self):
        # Run baseline
        np.random.seed(42)
        baseline = run_continuity_benchmark(_current_continuity, _current_estimate)
        baseline_overall = baseline["__overall__"]

        # Run experiment
        np.random.seed(42)
        experiment = run_continuity_benchmark(trajectory_continuity, _current_estimate)
        experiment_overall = experiment["__overall__"]

        # --- pretty-print comparison table ---
        all_keys = [
            k for k in baseline
            if k != "__overall__" and isinstance(baseline[k], float)
        ]

        print()
        print("=" * 72)
        print(f"{'SUB-TEST':<38} {'BASELINE':>10} {'EXPER-A':>10} {'DELTA':>10}")
        print("-" * 72)
        for key in all_keys:
            b = baseline[key]
            e = experiment[key]
            delta = e - b
            marker = "  *" if abs(delta) > 0.001 else ""
            print(f"  {key:<36} {b:>9.1%} {e:>9.1%} {delta:>+9.1%}{marker}")
        print("-" * 72)
        delta_overall = experiment_overall - baseline_overall
        print(
            f"  {'OVERALL':<36} {baseline_overall:>9.1%} "
            f"{experiment_overall:>9.1%} {delta_overall:>+9.1%}"
        )
        print("=" * 72)
        print()

        # Assert experiment is at least as good as baseline
        assert experiment_overall >= baseline_overall, (
            f"Experiment A regressed! {experiment_overall:.1%} < {baseline_overall:.1%}"
        )

        # Print individual improvements for insight
        improved = [
            k for k in all_keys
            if experiment[k] > baseline[k] + 0.001
        ]
        regressed = [
            k for k in all_keys
            if experiment[k] < baseline[k] - 0.001
        ]
        if improved:
            print(f"Improved sub-tests:  {', '.join(improved)}")
        if regressed:
            print(f"Regressed sub-tests: {', '.join(regressed)}")
        if not improved and not regressed:
            print("No changes in any sub-test.")
