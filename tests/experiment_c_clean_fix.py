# tests/experiment_c_clean_fix.py
"""
Experiment C: Persistent VM + EMA + Wider Threshold

Clean fix combining the parts that showed promise in previous experiments:
1. Persistent SmartVoiceManager in session_state (prevents voice reuse)
2. EMA pitch updates (tracks drift smoothly)
3. Wider 40Hz threshold (tolerates larger drift)
4. Crowded-pool penalty (prefer matching when many speakers exist)
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
# Experiment C: clean_continuity
# ---------------------------------------------------------------------------

def clean_continuity(pitch_hz, session_state):
    """
    Improved continuity logic with:
    - Persistent SmartVoiceManager across the session
    - EMA pitch updates to follow drift
    - Wider 40Hz matching threshold
    - Crowded-pool relaxation (45Hz when >= 4 known speakers)
    """
    THRESHOLD = 40.0
    CROWDED_THRESHOLD = 50.0
    CROWDED_MIN_SPEAKERS = 4
    EMA_ALPHA = 0.3  # weight for new observation: new = 0.7*old + 0.3*current

    # Ensure persistent VoiceManager
    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()
    vm = session_state["_vm"]

    # Ensure pitch/voice tracking dicts exist
    if "speaker_pitches" not in session_state:
        session_state["speaker_pitches"] = {}
    if "speaker_voice_ids" not in session_state:
        session_state["speaker_voice_ids"] = {}

    pitches = session_state["speaker_pitches"]
    voice_ids = session_state["speaker_voice_ids"]

    # Find best matching existing speaker
    best_match = None
    best_dist = float("inf")

    for spk_id, spk_pitch in pitches.items():
        dist = abs(pitch_hz - spk_pitch)
        if dist < best_dist:
            best_dist = dist
            best_match = spk_id

    # Decision logic
    num_known = len(pitches)

    if best_match is not None and best_dist <= THRESHOLD:
        # Normal match -- update pitch with EMA
        pitches[best_match] = (1.0 - EMA_ALPHA) * pitches[best_match] + EMA_ALPHA * pitch_hz
        return voice_ids[best_match]

    elif (num_known >= CROWDED_MIN_SPEAKERS
          and best_match is not None
          and best_dist <= CROWDED_THRESHOLD):
        # Crowded pool -- prefer matching over creating yet another speaker
        pitches[best_match] = (1.0 - EMA_ALPHA) * pitches[best_match] + EMA_ALPHA * pitch_hz
        return voice_ids[best_match]

    else:
        # Create new speaker using the PERSISTENT VM (tracks used_voice_ids)
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        spk_id = f"SPK_{len(voice_ids):02d}"
        voice_ids[spk_id] = voice_id
        pitches[spk_id] = pitch_hz
        return voice_id


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestExperimentC:
    """Compare baseline vs Experiment C (clean fix) on continuity benchmark."""

    def test_comparison(self):
        # --- Baseline ---
        np.random.seed(42)
        baseline = run_continuity_benchmark(_current_continuity, _current_estimate)

        # --- Experiment C ---
        np.random.seed(42)
        experiment = run_continuity_benchmark(clean_continuity, _current_estimate)

        # --- Comparison table ---
        sub_tests = [k for k in baseline if k != "__overall__"]

        print()
        print("=" * 72)
        print("  EXPERIMENT C: Persistent VM + EMA + Wider Threshold")
        print("=" * 72)
        print(f"  {'Sub-test':<35} {'Baseline':>10} {'Exp C':>10} {'Delta':>10}")
        print("-" * 72)

        wins = 0
        losses = 0
        ties = 0
        for name in sub_tests:
            b = baseline[name]
            e = experiment[name]
            delta = e - b
            marker = ""
            if delta > 0.001:
                marker = " (+)"
                wins += 1
            elif delta < -0.001:
                marker = " (-)"
                losses += 1
            else:
                ties += 1
            print(f"  {name:<35} {b:>9.1%} {e:>9.1%} {delta:>+9.1%}{marker}")

        b_overall = baseline["__overall__"]
        e_overall = experiment["__overall__"]
        d_overall = e_overall - b_overall
        print("-" * 72)
        print(f"  {'OVERALL':<35} {b_overall:>9.1%} {e_overall:>9.1%} {d_overall:>+9.1%}")
        print(f"\n  Wins: {wins}  Ties: {ties}  Losses: {losses}")
        print("=" * 72)

        # Assertions: experiment should not regress overall
        assert e_overall >= b_overall, (
            f"Experiment C regressed: {e_overall:.1%} < {b_overall:.1%}"
        )
        # Should improve on at least one sub-test
        assert wins > 0, "Experiment C did not improve any sub-test"
