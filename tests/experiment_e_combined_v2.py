# tests/experiment_e_combined_v2.py
"""
Experiment E: Kitchen Sink -- All Best Strategies Combined (v2)
===============================================================
Combines ALL of the most promising approaches from prior experiments:

  1. Persistent SmartVoiceManager in session_state (prevents voice collisions)
  2. Rich speaker profiles with pitch history, EMA, trend, last_seen
  3. Scoring function: pitch distance (with trend extrapolation) + recency bonus
  4. Safety threshold of 45Hz on raw pitch distance
  5. Trend computation via simple slope over 3+ observations

Goal: beat the baseline 81.1% overall continuity by improving the hard
sub-tests (crossing, large_drift, many_speakers) without regressing the
easy ones (stable, two_speaker, similar_pitch).
"""
import sys
import os
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
# Experiment E: Combined v2 continuity function
# ---------------------------------------------------------------------------

def combined_v2_continuity(pitch_hz: float, session_state: dict) -> str:
    """
    Kitchen-sink continuity that combines:
      - Persistent SmartVoiceManager (no voice collisions across new speakers)
      - Rich speaker profiles with pitch history, EMA, trend, last_seen
      - Scoring: pitch distance from trend-extrapolated expected pitch,
        normalized to a score, plus a recency bonus
      - Safety threshold: only accept match if raw pitch_dist < 45Hz

    State stored in session_state:
        _vm        -- persistent SmartVoiceManager instance
        _profiles  -- dict of spk_id -> {
                        pitches: [history],
                        ema_pitch: float,
                        trend: float (Hz/step),
                        last_seen: int (step number),
                        voice_id: str,
                      }
        _step      -- int, incremented each call
    """
    # ------------------------------------------------------------------
    # 1. Initialize persistent state on first call
    # ------------------------------------------------------------------
    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()
    if "_profiles" not in session_state:
        session_state["_profiles"] = {}
    if "_step" not in session_state:
        session_state["_step"] = 0

    vm = session_state["_vm"]
    profiles = session_state["_profiles"]
    step = session_state["_step"]
    session_state["_step"] = step + 1

    # Also maintain legacy keys so run_continuity_benchmark's session
    # initialization doesn't cause issues
    session_state.setdefault("speaker_voice_ids", {})
    session_state.setdefault("speaker_pitches", {})

    # ------------------------------------------------------------------
    # 2. Score every known speaker
    # ------------------------------------------------------------------
    PITCH_SAFETY = 45.0       # max raw Hz distance to accept a match
    PITCH_NORM = 60.0         # normalization range for pitch_score
    RECENCY_DECAY = 0.2       # how fast recency bonus decays per step
    RECENCY_WEIGHT = 0.3      # weight of the recency bonus in total_score
    EMA_OLD = 0.7
    EMA_NEW = 0.3

    best_speaker = None
    best_score = -1.0

    for spk_id, prof in profiles.items():
        # Compute expected pitch using trend extrapolation
        if len(prof["pitches"]) >= 2 and prof["trend"] is not None:
            expected_pitch = prof["ema_pitch"] + prof["trend"]
        else:
            expected_pitch = prof["ema_pitch"]

        pitch_dist = abs(pitch_hz - expected_pitch)

        # Safety: reject if too far away
        if pitch_dist >= PITCH_SAFETY:
            continue

        # Pitch score: 1.0 when perfect match, 0.0 at PITCH_NORM distance
        pitch_score = max(0.0, 1.0 - pitch_dist / PITCH_NORM)

        # Recency bonus: favors recently-seen speakers
        steps_ago = step - prof["last_seen"]
        recency_bonus = 1.0 / (1.0 + steps_ago * RECENCY_DECAY)

        total_score = pitch_score + recency_bonus * RECENCY_WEIGHT

        if total_score > best_score:
            best_score = total_score
            best_speaker = spk_id

    # ------------------------------------------------------------------
    # 3. If matched, update the profile
    # ------------------------------------------------------------------
    if best_speaker is not None:
        prof = profiles[best_speaker]

        # Update EMA pitch
        prof["ema_pitch"] = EMA_OLD * prof["ema_pitch"] + EMA_NEW * pitch_hz

        # Append to history (keep last 8 observations)
        prof["pitches"].append(pitch_hz)
        if len(prof["pitches"]) > 8:
            prof["pitches"] = prof["pitches"][-8:]

        # Recompute trend: simple slope over observations
        prof["trend"] = _compute_trend(prof["pitches"])

        # Update last_seen
        prof["last_seen"] = step

        return prof["voice_id"]

    # ------------------------------------------------------------------
    # 4. No match -- create new speaker using persistent VM
    # ------------------------------------------------------------------
    gender = gender_from_pitch(pitch_hz)
    voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)

    spk_id = f"SPK_{len(profiles):02d}"
    profiles[spk_id] = {
        "pitches": [pitch_hz],
        "ema_pitch": pitch_hz,
        "trend": None,
        "last_seen": step,
        "voice_id": voice_id,
    }

    # Keep legacy dicts in sync (some benchmarks read them)
    session_state["speaker_voice_ids"][spk_id] = voice_id
    session_state["speaker_pitches"][spk_id] = pitch_hz

    return voice_id


def _compute_trend(pitches: list) -> float:
    """
    Compute pitch trend (Hz/step) from observation history.
    Returns slope = (latest - earliest) / (num_steps - 1) for 3+ observations.
    Returns None if fewer than 3 observations.
    """
    if len(pitches) < 3:
        return None
    # Simple slope: latest minus earliest, divided by number of intervals
    return (pitches[-1] - pitches[0]) / (len(pitches) - 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperimentECombinedV2:
    """Compare baseline continuity vs Experiment E (kitchen sink combined v2)."""

    def test_combined_v2_vs_baseline(self):
        # Run baseline with fixed seed
        np.random.seed(42)
        baseline_results = run_continuity_benchmark(
            _current_continuity, _current_estimate
        )

        # Run experiment with same fixed seed
        np.random.seed(42)
        experiment_results = run_continuity_benchmark(
            combined_v2_continuity, _current_estimate
        )

        baseline_overall = baseline_results["__overall__"]
        experiment_overall = experiment_results["__overall__"]

        # Collect sub-test keys (all float-valued entries except __overall__)
        sub_tests = [
            k for k in baseline_results
            if k != "__overall__" and isinstance(baseline_results[k], float)
        ]

        # Print comparison table
        print(f"\n{'=' * 76}")
        print(f"  Experiment E: Kitchen Sink -- All Best Strategies Combined (v2)")
        print(f"{'=' * 76}")
        print(
            f"  {'Sub-test':<40} {'Baseline':>10} {'Exp-E':>10} {'Delta':>10}"
        )
        print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")

        for name in sub_tests:
            b_val = baseline_results[name]
            e_val = experiment_results[name]
            delta = e_val - b_val
            marker = ""
            if delta > 0.001:
                marker = " IMPROVED"
            elif delta < -0.001:
                marker = " REGRESSED"
            print(
                f"  {name:<40} {b_val:>9.1%} {e_val:>9.1%} "
                f"{delta:>+9.1%}{marker}"
            )

        print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
        overall_delta = experiment_overall - baseline_overall
        print(
            f"  {'OVERALL':<40} {baseline_overall:>9.1%} "
            f"{experiment_overall:>9.1%} {overall_delta:>+9.1%}"
        )
        print(f"{'=' * 76}")

        # List improvements and regressions
        improved = [
            k for k in sub_tests
            if experiment_results[k] > baseline_results[k] + 0.001
        ]
        regressed = [
            k for k in sub_tests
            if experiment_results[k] < baseline_results[k] - 0.001
        ]
        if improved:
            print(f"\n  Improved sub-tests:  {', '.join(improved)}")
        if regressed:
            print(f"  Regressed sub-tests: {', '.join(regressed)}")
        if not improved and not regressed:
            print(f"\n  No changes in any sub-test.")
        print()

        # Assert: overall must be >= baseline
        assert experiment_overall >= baseline_overall, (
            f"Experiment E ({experiment_overall:.1%}) regressed vs "
            f"baseline ({baseline_overall:.1%})"
        )
