# tests/experiment_d_profile.py
"""
Experiment D: Multi-Feature Speaker Profile for continuity matching.

Instead of matching on pitch alone with a fixed 30Hz threshold, this builds
a richer speaker "profile" from the history of observations:
  - Pitch (primary, via exponential moving average)
  - Pitch variance/stability (tight vs loose threshold)
  - Pitch trend direction (extrapolate expected next pitch)

This should improve continuity in scenarios where speakers drift, cross,
or vary widely in pitch across chunks.
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
# Experiment D: Profile-based continuity
# ---------------------------------------------------------------------------

def profile_continuity(pitch_hz: float, session_state: dict) -> str:
    """
    Multi-feature speaker profile continuity matching.

    Maintains per-speaker profiles in session_state["_profiles"] with:
      - pitches: full history of observed pitches
      - avg_pitch: exponential moving average of pitch
      - trend: slope of recent observations (Hz per step)
      - voice_id: assigned voice

    Matching logic:
      1. For each known speaker, compute *expected* pitch:
         - If 2+ observations: expected = avg_pitch + trend  (extrapolate)
         - If 1 observation: expected = avg_pitch
      2. Find closest speaker by distance to expected pitch.
      3. Use adaptive threshold based on speaker's own variance:
         - Low variance (stable): 25 Hz  (tighter matching)
         - High variance (erratic): 45 Hz  (looser matching)
      4. If matched, update profile. Otherwise, register new speaker.

    Uses a persistent SmartVoiceManager from session_state["_vm"].
    """
    # -- Initialise session-level structures on first call -----------------
    if "_profiles" not in session_state:
        session_state["_profiles"] = {}
    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()

    profiles = session_state["_profiles"]
    vm = session_state["_vm"]

    EMA_ALPHA = 0.4          # weighting for new observation in EMA
    MIN_THRESHOLD = 25.0     # tight threshold for stable speakers
    MAX_THRESHOLD = 45.0     # loose threshold for erratic speakers
    VARIANCE_SCALE = 1.5     # multiplier on stddev -> threshold contribution

    # -- Compute expected pitch & distance for each known speaker ----------
    best_match_id = None
    best_dist = float("inf")
    best_threshold = MIN_THRESHOLD

    for spk_id, prof in profiles.items():
        n_obs = len(prof["pitches"])

        # Expected pitch: extrapolate using trend if we have enough data
        if n_obs >= 2:
            expected = prof["avg_pitch"] + prof["trend"]
        else:
            expected = prof["avg_pitch"]

        dist = abs(pitch_hz - expected)

        # Adaptive threshold based on this speaker's own variance
        if n_obs >= 2:
            stddev = float(np.std(prof["pitches"]))
            threshold = np.clip(
                MIN_THRESHOLD + VARIANCE_SCALE * stddev,
                MIN_THRESHOLD,
                MAX_THRESHOLD,
            )
        else:
            threshold = 30.0  # neutral default for single-observation speakers

        if dist < best_dist:
            best_dist = dist
            best_match_id = spk_id
            best_threshold = threshold

    # -- Match or create new speaker ---------------------------------------
    if best_match_id is not None and best_dist <= best_threshold:
        # Update existing profile
        prof = profiles[best_match_id]
        prof["pitches"].append(pitch_hz)

        # Update EMA
        prof["avg_pitch"] = (
            EMA_ALPHA * pitch_hz + (1 - EMA_ALPHA) * prof["avg_pitch"]
        )

        # Recompute trend from last few observations (up to 4)
        recent = prof["pitches"][-4:]
        if len(recent) >= 2:
            # Simple linear slope via first differences
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            prof["trend"] = float(np.mean(diffs))
        else:
            prof["trend"] = 0.0

        # Also mirror into the legacy keys so the harness session_state
        # stays consistent (not strictly needed but defensive)
        session_state.setdefault("speaker_pitches", {})[best_match_id] = (
            prof["avg_pitch"]
        )

        return prof["voice_id"]

    else:
        # New speaker
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)

        spk_id = f"SPK_{len(profiles):02d}"
        profiles[spk_id] = {
            "pitches": [pitch_hz],
            "avg_pitch": pitch_hz,
            "trend": 0.0,
            "voice_id": voice_id,
        }

        # Mirror into legacy keys
        session_state.setdefault("speaker_voice_ids", {})[spk_id] = voice_id
        session_state.setdefault("speaker_pitches", {})[spk_id] = pitch_hz

        return voice_id


# ---------------------------------------------------------------------------
# Test class: compare baseline vs profile continuity
# ---------------------------------------------------------------------------

class TestExperimentDProfile:
    """Compare baseline (_current_continuity) vs profile_continuity."""

    def test_profile_vs_baseline(self):
        # ---------- run baseline ----------
        np.random.seed(42)
        baseline = run_continuity_benchmark(_current_continuity, _current_estimate)

        # ---------- run experiment D ----------
        np.random.seed(42)
        experiment = run_continuity_benchmark(profile_continuity, _current_estimate)

        # ---------- comparison table ----------
        sub_tests = [k for k in baseline if k != "__overall__"]

        print()
        print("=" * 72)
        print("EXPERIMENT D: Multi-Feature Speaker Profile  --  Continuity Results")
        print("=" * 72)
        header = f"{'Sub-test':<35} {'Baseline':>10} {'Profile':>10} {'Delta':>10}"
        print(header)
        print("-" * 72)

        wins = 0
        losses = 0
        ties = 0
        for name in sub_tests:
            b_val = baseline[name]
            e_val = experiment[name]
            delta = e_val - b_val
            marker = ""
            if delta > 0.001:
                marker = "  +"
                wins += 1
            elif delta < -0.001:
                marker = "  -"
                losses += 1
            else:
                ties += 1
            print(
                f"  {name:<33} {b_val:>9.1%} {e_val:>9.1%} "
                f"{delta:>+9.1%}{marker}"
            )

        b_overall = baseline["__overall__"]
        e_overall = experiment["__overall__"]
        delta_overall = e_overall - b_overall
        print("-" * 72)
        print(
            f"  {'OVERALL':<33} {b_overall:>9.1%} {e_overall:>9.1%} "
            f"{delta_overall:>+9.1%}"
        )
        print("-" * 72)
        print(f"  Wins: {wins}   Losses: {losses}   Ties: {ties}")
        print("=" * 72)

        # Assertion: experiment should be at least as good overall
        assert e_overall >= b_overall, (
            f"Experiment D regressed: {e_overall:.1%} < baseline {b_overall:.1%}"
        )
