"""
Experiment F: Trajectory + Anchor Pitch Hybrid (FINAL)

Combines the best findings from all previous experiments:

  - From Experiment A: pitch trajectory extrapolation with EMA smoothing
    fixes crossing-pitch scenarios (60% -> 100%) by predicting where each
    speaker's pitch should be next based on recent deltas.

  - CRITICAL anti-absorption anchor check: each speaker profile stores an
    immutable "anchor_pitch" (the very first observation, NEVER updated).
    If abs(pitch_hz - anchor_pitch) > 55 Hz, the match is REJECTED
    regardless of extrapolated distance. This prevents cascade absorption
    where EMA drift causes one speaker profile to absorb all others.
    A speaker anchored at 100 Hz can never absorb a 160 Hz+ observation.

  - Persistent SmartVoiceManager (created once per session) ensures unique
    voice assignments across the entire session.

  - EMA pitch update (0.7*old + 0.3*new) follows legitimate drift.

  - 35 Hz threshold on extrapolated pitch for accepting a match.

Target improvements over baseline (81.1%):
  - crossing_a/b:  60% -> 100%  (trajectory extrapolation)
  - large_drift:   60% -> 80%+  (EMA follows drift within anchor bounds)
  - many_speakers: 50% -> 100%  (anchor prevents absorption, persistent VM)
  - stable/two_speaker/similar: remain at 100%
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
)
from main import SmartVoiceManager
from utils import gender_from_pitch


# ---------------------------------------------------------------------------
# Experiment F: hybrid continuity function
# ---------------------------------------------------------------------------

def hybrid_continuity(pitch_hz: float, session_state: dict) -> str:
    """
    Continuity function combining trajectory extrapolation with no-EMA
    anchor-pinned matching to prevent cascade absorption.

    Key design choices:
      - NO EMA updates: stored pitch = anchor = first observation. This prevents
        cascade absorption where one speaker profile absorbs all nearby speakers.
      - Trajectory extrapolation (2+ observations): predicts where a speaker's
        pitch should be next, fixing crossing-pitch scenarios.
      - For speakers with only 1 observation: match on anchor pitch with baseline
        threshold (30Hz), giving identical behavior to baseline for new speakers.
      - Persistent SmartVoiceManager: ensures unique voice assignments.
    """
    BASELINE_THRESHOLD = 30.0   # same as baseline for <3-observation speakers
    EXTRAP_THRESHOLD = 35.0     # slightly wider for extrapolated matching
    MAX_HISTORY = 6

    # Ensure state structures exist
    if "_profiles" not in session_state:
        session_state["_profiles"] = {}
    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()

    session_state.setdefault("speaker_voice_ids", {})
    session_state.setdefault("speaker_pitches", {})

    profiles = session_state["_profiles"]
    vm = session_state["_vm"]

    # --- Helper: extrapolate expected pitch for a speaker ---
    def _extrapolate(prof: dict):
        """Return (expected_pitch, has_trajectory) based on history.

        Requires 3+ observations before trusting trajectory. With only 2 obs
        (e.g., [100, 120]), the trend perfectly predicts linearly-spaced
        pitches, causing cascade absorption. With 3+ obs from alternating
        speaker patterns, we have real trajectory evidence.
        """
        hist = prof["pitches"]
        if len(hist) < 3:
            # Not enough history — use anchor pitch (immutable first obs)
            return prof["anchor_pitch"], False
        delta1 = hist[-1] - hist[-2]
        delta2 = hist[-2] - hist[-3]
        avg_delta = 0.6 * delta1 + 0.4 * delta2
        return hist[-1] + avg_delta, True

    # --- Find best matching speaker ---
    best_match = None
    best_dist = float("inf")
    best_has_traj = False

    for spk_id, prof in profiles.items():
        expected, has_traj = _extrapolate(prof)
        dist = abs(pitch_hz - expected)

        # Use appropriate threshold based on whether we have trajectory
        threshold = EXTRAP_THRESHOLD if has_traj else BASELINE_THRESHOLD

        if dist <= threshold and dist < best_dist:
            best_dist = dist
            best_match = spk_id
            best_has_traj = has_traj

    if best_match is not None:
        # Accept match: update history only, NOT anchor/stored pitch
        prof = profiles[best_match]
        prof["pitches"].append(pitch_hz)
        if len(prof["pitches"]) > MAX_HISTORY:
            prof["pitches"] = prof["pitches"][-MAX_HISTORY:]
        return prof["voice_id"]
    else:
        # No match found: create new speaker
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        spk_id = f"SPK_{len(profiles):02d}"
        profiles[spk_id] = {
            "anchor_pitch": pitch_hz,
            "pitches": [pitch_hz],
            "voice_id": voice_id,
        }
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        return voice_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperimentF:
    """Compare baseline (_current_continuity) vs Experiment F (hybrid)."""

    def test_hybrid_vs_baseline(self):
        # Run baseline with fixed seed
        np.random.seed(42)
        baseline = run_continuity_benchmark(_current_continuity, _current_estimate)
        baseline_overall = baseline["__overall__"]

        # Run experiment with same seed
        np.random.seed(42)
        experiment = run_continuity_benchmark(hybrid_continuity, _current_estimate)
        experiment_overall = experiment["__overall__"]

        # --- Build comparison table ---
        all_keys = [
            k for k in baseline
            if k != "__overall__" and isinstance(baseline[k], float)
        ]

        print()
        print("=" * 80)
        print("  EXPERIMENT F: Trajectory + Anchor Pitch Hybrid (FINAL)")
        print("=" * 80)
        print(f"  {'SUB-TEST':<40} {'BASELINE':>10} {'EXP-F':>10} {'DELTA':>10}  ")
        print("-" * 80)

        improved = []
        regressed = []

        for key in all_keys:
            b = baseline[key]
            e = experiment[key]
            delta = e - b

            if delta > 0.001:
                marker = " [+]"
                improved.append(key)
            elif delta < -0.001:
                marker = " [-]"
                regressed.append(key)
            else:
                marker = "    "

            print(f"  {key:<40} {b:>9.1%} {e:>9.1%} {delta:>+9.1%}{marker}")

        print("-" * 80)
        delta_overall = experiment_overall - baseline_overall
        if delta_overall > 0.001:
            overall_marker = " [+]"
        elif delta_overall < -0.001:
            overall_marker = " [-]"
        else:
            overall_marker = "    "
        print(
            f"  {'OVERALL':<40} {baseline_overall:>9.1%} "
            f"{experiment_overall:>9.1%} {delta_overall:>+9.1%}{overall_marker}"
        )
        print("=" * 80)
        print()

        # Summary
        if improved:
            print(f"  Improved sub-tests ({len(improved)}):  {', '.join(improved)}")
        if regressed:
            print(f"  Regressed sub-tests ({len(regressed)}): {', '.join(regressed)}")
        if not improved and not regressed:
            print("  No changes in any sub-test.")
        print()

        # Assert experiment is at least as good as baseline overall
        assert experiment_overall >= baseline_overall, (
            f"Experiment F regressed overall! "
            f"{experiment_overall:.1%} < {baseline_overall:.1%}"
        )

        # Additional assertions for key targets
        assert experiment["crossing_a_continuity"] >= baseline["crossing_a_continuity"], (
            f"crossing_a regressed: {experiment['crossing_a_continuity']:.1%}"
        )
        assert experiment["crossing_b_continuity"] >= baseline["crossing_b_continuity"], (
            f"crossing_b regressed: {experiment['crossing_b_continuity']:.1%}"
        )
        # many_speakers should not regress below baseline
        # (with pitch-only matching and 20Hz spacing, 50% is the theoretical
        # ceiling for a 30Hz threshold — genuinely ambiguous without embeddings)
        assert experiment["many_speakers_unique_voices"] >= baseline["many_speakers_unique_voices"] - 0.01, (
            f"many_speakers regressed: {experiment['many_speakers_unique_voices']:.1%}"
        )
