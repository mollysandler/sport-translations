# tests/experiment_b_recency.py
"""
Experiment B: Recency + Pitch Combined Scoring for Continuity

Instead of matching purely by pitch distance (30Hz threshold),
combine pitch proximity with recency. Track a `last_seen_step` counter
for each speaker. Score = pitch_proximity_score * recency_weight.

A speaker seen 1 step ago gets higher weight than one seen 5 steps ago.
This helps with crossing pitches: even when two speakers have similar
pitches, the one seen most recently in the alternation pattern gets
priority.
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
# Experiment B: Recency + Pitch Combined Scoring
# ---------------------------------------------------------------------------

def recency_continuity(pitch_hz, session_state):
    """
    Continuity function that combines pitch proximity with recency scoring.

    For each known speaker, compute:
      - pitch_score = max(0, 1.0 - abs(pitch_hz - stored_pitch) / 50.0)
      - steps_ago = current_step - last_seen_step
      - recency_score = 1.0 / (1.0 + steps_ago * 0.3)
      - combined_score = pitch_score * recency_score

    Pick the speaker with the highest combined_score, but only if the
    pitch distance is < 45Hz (safety threshold).

    After matching, update the stored pitch with EMA (0.7*old + 0.3*new)
    and update last_seen step.
    """
    # Initialize persistent state on first call
    if "_step" not in session_state:
        session_state["_step"] = 0
    if "_last_seen" not in session_state:
        session_state["_last_seen"] = {}
    if "speaker_pitches" not in session_state:
        session_state["speaker_pitches"] = {}
    if "speaker_voice_ids" not in session_state:
        session_state["speaker_voice_ids"] = {}
    if "_vm" not in session_state:
        session_state["_vm"] = SmartVoiceManager()

    current_step = session_state["_step"]
    session_state["_step"] = current_step + 1

    PITCH_SAFETY_THRESHOLD = 45.0
    PITCH_NORM = 50.0
    RECENCY_DECAY = 0.3
    EMA_OLD = 0.7
    EMA_NEW = 0.3

    best_speaker = None
    best_combined = -1.0

    for spk_id, spk_pitch in session_state["speaker_pitches"].items():
        pitch_dist = abs(pitch_hz - spk_pitch)

        # Safety: reject if pitch is too far away
        if pitch_dist >= PITCH_SAFETY_THRESHOLD:
            continue

        pitch_score = max(0.0, 1.0 - pitch_dist / PITCH_NORM)

        last_seen = session_state["_last_seen"].get(spk_id, -1)
        steps_ago = current_step - last_seen
        recency_score = 1.0 / (1.0 + steps_ago * RECENCY_DECAY)

        combined = pitch_score * recency_score
        if combined > best_combined:
            best_combined = combined
            best_speaker = spk_id

    if best_speaker is not None:
        # Update pitch with EMA
        old_pitch = session_state["speaker_pitches"][best_speaker]
        session_state["speaker_pitches"][best_speaker] = (
            EMA_OLD * old_pitch + EMA_NEW * pitch_hz
        )
        # Update last seen
        session_state["_last_seen"][best_speaker] = current_step
        return session_state["speaker_voice_ids"][best_speaker]
    else:
        # No match found -- create a new speaker
        vm = session_state["_vm"]
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)

        spk_id = f"SPK_{len(session_state['speaker_voice_ids']):02d}"
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        session_state["_last_seen"][spk_id] = current_step
        return voice_id


# ---------------------------------------------------------------------------
# Test class: compare baseline vs recency on continuity benchmark
# ---------------------------------------------------------------------------

class TestExperimentBRecency:
    """Compare baseline continuity (30Hz threshold) vs recency+pitch scoring."""

    def test_recency_vs_baseline(self):
        # Run baseline
        np.random.seed(42)
        baseline_results = run_continuity_benchmark(
            _current_continuity, _current_estimate
        )

        # Run experiment
        np.random.seed(42)
        experiment_results = run_continuity_benchmark(
            recency_continuity, _current_estimate
        )

        baseline_overall = baseline_results["__overall__"]
        experiment_overall = experiment_results["__overall__"]

        # Print comparison table
        print(f"\n{'=' * 72}")
        print(f"  Experiment B: Recency + Pitch Combined Scoring")
        print(f"{'=' * 72}")
        print(f"  {'Sub-test':<40} {'Baseline':>10} {'Recency':>10} {'Delta':>10}")
        print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")

        sub_tests = [
            k for k in baseline_results
            if k != "__overall__" and isinstance(baseline_results[k], float)
        ]

        for name in sub_tests:
            b_val = baseline_results[name]
            e_val = experiment_results[name]
            delta = e_val - b_val
            delta_str = f"{delta:+.1%}"
            print(f"  {name:<40} {b_val:>9.1%} {e_val:>9.1%} {delta_str:>10}")

        print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
        overall_delta = experiment_overall - baseline_overall
        print(
            f"  {'OVERALL':<40} {baseline_overall:>9.1%} "
            f"{experiment_overall:>9.1%} {overall_delta:+.1%}"
        )
        print(f"{'=' * 72}")

        # Assert experiment is at least as good as baseline
        assert experiment_overall >= baseline_overall, (
            f"Experiment B ({experiment_overall:.1%}) regressed vs "
            f"baseline ({baseline_overall:.1%})"
        )
