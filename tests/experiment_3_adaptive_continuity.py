# tests/experiment_3_adaptive_continuity.py
"""
Experiment 3: Adaptive Live Continuity Threshold

Replaces fixed 30Hz pitch threshold with adaptive matching that:
  a) Computes threshold from session data (tightens when speakers are close)
  b) Tracks running pitch average via exponential moving average
  c) Requires best-match margin to avoid ambiguous speaker swaps
  d) Uses persistent SmartVoiceManager so each new speaker gets a distinct voice

Baseline overall continuity: 81.1%
"""

import sys, os
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from main import _estimate_pitch_safe, SmartVoiceManager
from utils import gender_from_pitch
from tests.test_diarizer_experiments import (
    run_pitch_benchmark,
    run_gender_benchmark,
    run_voice_matching_benchmark,
    run_continuity_benchmark,
    _current_continuity,
)


# ---------------------------------------------------------------------------
# Helpers shared by the experiment
# ---------------------------------------------------------------------------

def _current_estimate(audio, sr):
    """Wrap _estimate_pitch_safe to match benchmark interface."""
    return _estimate_pitch_safe(audio, sr)


def _current_gender(pitch_hz, pitch_range=None):
    return gender_from_pitch(pitch_hz, pitch_range)


def _current_voice_match(vm, pitch_hz, pitch_range):
    gender = gender_from_pitch(pitch_hz, pitch_range)
    voice_id = vm._match_best_voice(gender, pitch_hz, pitch_range)
    return voice_id


# ---------------------------------------------------------------------------
# Adaptive continuity function
# ---------------------------------------------------------------------------

def _compute_adaptive_threshold(session_state):
    """
    Compute a pitch-matching threshold from the current session speakers.

    - < 2 speakers known  -> generous 40 Hz default (encourage matching)
    - >= 2 speakers       -> 60% of the minimum gap between any two known
                             speakers, floored at 20 Hz
    """
    pitches = list(session_state.get("speaker_pitches", {}).values())
    if len(pitches) < 2:
        return 40.0

    # Compute minimum pairwise gap among stored (EMA-smoothed) pitches
    sorted_pitches = sorted(pitches)
    min_gap = min(
        sorted_pitches[i + 1] - sorted_pitches[i]
        for i in range(len(sorted_pitches) - 1)
    )
    return max(20.0, min_gap * 0.6)


def adaptive_continuity(pitch_hz, session_state):
    """
    Adaptive live continuity matching.

    Signature matches benchmark requirement:
        fn(pitch_hz: float, session_state: dict) -> str (voice_id)

    Session state keys used / mutated:
        speaker_voice_ids  dict[str, str]   spk_id -> voice_id
        speaker_pitches    dict[str, float] spk_id -> smoothed pitch
        _voice_manager     SmartVoiceManager (lazily created, persisted)
    """
    # Lazily create a persistent SmartVoiceManager for this session
    if "_voice_manager" not in session_state:
        session_state["_voice_manager"] = SmartVoiceManager()

    vm = session_state["_voice_manager"]
    threshold = _compute_adaptive_threshold(session_state)

    # ---- Find best and second-best matches among known speakers ----
    best_match = None
    best_dist = float("inf")
    second_best_dist = float("inf")

    for spk_id, spk_pitch in session_state.get("speaker_pitches", {}).items():
        dist = abs(pitch_hz - spk_pitch)
        if dist < best_dist:
            second_best_dist = best_dist
            best_dist = dist
            best_match = spk_id
        elif dist < second_best_dist:
            second_best_dist = dist

    # ---- Decide: match existing speaker or register a new one ----
    MARGIN_HZ = 5.0  # ambiguity guard
    EMA_ALPHA = 0.3   # weight of new observation

    matched = False
    if best_match is not None and best_dist <= threshold:
        # Only accept if the match is unambiguous (margin check)
        if second_best_dist - best_dist > MARGIN_HZ or second_best_dist == float("inf"):
            # Update stored pitch via exponential moving average
            old_pitch = session_state["speaker_pitches"][best_match]
            session_state["speaker_pitches"][best_match] = (
                (1 - EMA_ALPHA) * old_pitch + EMA_ALPHA * pitch_hz
            )
            matched = True

    if matched:
        return session_state["speaker_voice_ids"][best_match]

    # ---- No confident match -> assign a new voice ----
    gender = gender_from_pitch(pitch_hz)
    voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
    spk_id = f"SPK_{len(session_state['speaker_voice_ids']):02d}"
    session_state["speaker_voice_ids"][spk_id] = voice_id
    session_state["speaker_pitches"][spk_id] = pitch_hz
    return voice_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperiment3AdaptiveContinuity:
    """Run all benchmarks and compare adaptive continuity vs. baseline."""

    def test_continuity_adaptive_vs_baseline(self):
        """Core metric: continuity overall score (baseline 81.1%)."""
        np.random.seed(42)

        baseline_results = run_continuity_benchmark(
            _current_continuity, _current_estimate
        )
        baseline_overall = baseline_results["__overall__"]

        np.random.seed(42)  # Reset seed for identical random drift sequences
        adaptive_results = run_continuity_benchmark(
            adaptive_continuity, _current_estimate
        )
        adaptive_overall = adaptive_results["__overall__"]

        # ---- Pretty-print comparison ----
        print(f"\n{'=' * 68}")
        print(f"  EXPERIMENT 3: Adaptive Continuity Threshold")
        print(f"{'=' * 68}")
        print(f"{'Test':<40} {'Baseline':>10} {'Adaptive':>10} {'Delta':>8}")
        print(f"{'-' * 68}")

        all_keys = [k for k in adaptive_results if k != "__overall__"]
        for key in all_keys:
            b_val = baseline_results.get(key, 0.0)
            a_val = adaptive_results[key]
            delta = a_val - b_val
            marker = "  +" if delta > 0 else ("  -" if delta < 0 else "   ")
            print(f"  {key:<38} {b_val:>9.1%} {a_val:>9.1%} {marker}{abs(delta):.1%}")

        print(f"{'-' * 68}")
        delta_overall = adaptive_overall - baseline_overall
        marker = "+" if delta_overall > 0 else ("-" if delta_overall < 0 else " ")
        print(f"  {'OVERALL':<38} {baseline_overall:>9.1%} "
              f"{adaptive_overall:>9.1%} {marker}{abs(delta_overall):.1%}")
        print(f"{'=' * 68}\n")

        # Adaptive should beat or match baseline
        assert adaptive_overall >= baseline_overall, (
            f"Adaptive ({adaptive_overall:.1%}) should be >= baseline ({baseline_overall:.1%})"
        )

    def test_pitch_benchmark(self):
        """Pitch estimation (unchanged algorithm, just for completeness)."""
        results = run_pitch_benchmark(_current_estimate)
        overall = results["__overall__"]
        print(f"\n  Pitch estimation overall: {overall:.1%}")
        assert overall > 0.0

    def test_gender_benchmark(self):
        """Gender classification (unchanged, for completeness)."""
        results = run_gender_benchmark(_current_gender, _current_estimate)
        overall = results["__overall__"]
        print(f"\n  Gender classification overall: {overall:.1%}")
        assert overall > 0.0

    def test_voice_matching_benchmark(self):
        """Voice matching (unchanged, for completeness)."""
        results = run_voice_matching_benchmark(
            _current_voice_match, _current_estimate
        )
        overall = results["__overall__"]
        print(f"\n  Voice distinction overall: {overall:.1%}")
        assert overall > 0.0
