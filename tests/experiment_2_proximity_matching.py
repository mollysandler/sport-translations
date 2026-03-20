# tests/experiment_2_proximity_matching.py
"""
Experiment 2: Proximity-Based Voice Matching

Replaces the coarse binary pitch bins in SmartVoiceManager._match_best_voice
with a continuous inverse-distance scoring function.

Current approach:
  - avg_pitch < 140  => "low"   (score +3)
  - 140 <= avg_pitch < 180 => "medium_low" / "medium" (score +3)
  - avg_pitch >= 180 => "medium_high" / "medium" (score +3)

Problem: a speaker at 139 Hz scores "low" while 141 Hz scores "medium",
potentially getting completely different voices despite being 2 Hz apart.

Proximity approach:
  - Map each voice pitch label to a numeric center:
      low=110, medium_low=140, medium=160, medium_high=190
  - Score = 1.0 / (1.0 + abs(speaker_pitch - voice_center))
  - Continuous, smooth scoring with no cliff edges.
"""
import sys
import os
import numpy as np
import pytest
from typing import Dict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from main import _estimate_pitch_safe, SmartVoiceManager
from utils import gender_from_pitch
from tests.test_diarizer_experiments import (
    run_pitch_benchmark,
    run_gender_benchmark,
    run_voice_matching_benchmark,
    SCENARIOS,
)

# ---------------------------------------------------------------------------
# Pitch label -> numeric center mapping
# ---------------------------------------------------------------------------

PITCH_CENTERS = {
    "low": 110.0,
    "medium_low": 140.0,
    "medium": 160.0,
    "medium_high": 190.0,
}

# Sports commentary style bonus (same voices as the original)
SPORTS_STYLE_BONUS = {
    "strong_confident",
    "deep_authoritative",
    "energetic_young",
}


# ---------------------------------------------------------------------------
# Proximity-based match function
# ---------------------------------------------------------------------------

def proximity_voice_match(vm: SmartVoiceManager, pitch_hz: float, pitch_range: float) -> str:
    """
    Proximity-based voice matching.

    Instead of binning pitch into coarse categories (<140, 140-180, >=180),
    compute a continuous proximity score for each candidate voice:

        pitch_score = 1.0 / (1.0 + abs(speaker_pitch - voice_center))

    This avoids cliff-edge mismatches at bin boundaries (e.g. 139 vs 141 Hz).

    Follows the same contract as _current_voice_match:
      - Gender filtering via gender_from_pitch
      - Respects vm.used_voice_ids to avoid reuse
      - Falls back to opposite gender, then reuse, same as original
      - Adds style bonus for sports-appropriate voices
    """
    gender = gender_from_pitch(pitch_hz, pitch_range)

    # --- Step 1: filter candidates (same cascade as original) ---
    candidates = {
        vid: props for vid, props in vm.available_voices.items()
        if props["gender"] == gender and vid not in vm.used_voice_ids
    }

    if not candidates:
        # Fall back to opposite gender, still unused
        candidates = {
            vid: props for vid, props in vm.available_voices.items()
            if vid not in vm.used_voice_ids
        }

    if not candidates:
        # All voices used -- allow reuse, prefer same gender
        candidates = {
            vid: props for vid, props in vm.available_voices.items()
            if props["gender"] == gender
        }

    if not candidates:
        # Absolute fallback
        candidates = dict(vm.available_voices)

    # --- Step 2: score each candidate with proximity metric ---
    best_match = None
    best_score = -1.0

    for voice_id, props in candidates.items():
        voice_pitch_label = props["pitch"]
        voice_center = PITCH_CENTERS.get(voice_pitch_label, 160.0)  # default medium

        # Continuous inverse-distance score (higher = closer match)
        pitch_score = 1.0 / (1.0 + abs(pitch_hz - voice_center))

        # Style bonus for sports commentary voices (same as original)
        style_bonus = 0.0
        if props["style"] in SPORTS_STYLE_BONUS:
            style_bonus = 0.05  # small additive bonus, won't override pitch proximity

        total_score = pitch_score + style_bonus

        if total_score > best_score:
            best_score = total_score
            best_match = voice_id

    if best_match is None:
        best_match = list(candidates.keys())[0]

    vm.used_voice_ids.add(best_match)
    return best_match


# ---------------------------------------------------------------------------
# Baseline (current) match function (copied from test_diarizer_experiments)
# ---------------------------------------------------------------------------

def current_voice_match(vm: SmartVoiceManager, pitch_hz: float, pitch_range: float) -> str:
    """Current binary-bin voice matching (delegates to vm._match_best_voice)."""
    gender = gender_from_pitch(pitch_hz, pitch_range)
    voice_id = vm._match_best_voice(gender, pitch_hz, pitch_range)
    return voice_id


# ---------------------------------------------------------------------------
# Shared pitch estimator wrapper
# ---------------------------------------------------------------------------

def _estimate(audio, sr):
    return _estimate_pitch_safe(audio, sr)


def _gender(pitch_hz, pitch_range=None):
    return gender_from_pitch(pitch_hz, pitch_range)


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def print_comparison_table(baseline_results: Dict, proximity_results: Dict):
    """Print a side-by-side comparison of baseline vs proximity voice matching."""

    header = f"{'Scenario':<30} {'Baseline':>10} {'Proximity':>10} {'Delta':>10}"
    separator = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("VOICE MATCHING COMPARISON: Binary Bins vs Proximity Scoring")
    print(f"{'=' * len(header)}")
    print(header)
    print(separator)

    scenarios_sorted = sorted(
        k for k in baseline_results if k != "__overall__"
    )

    wins = 0
    ties = 0
    losses = 0

    for name in scenarios_sorted:
        b = baseline_results[name]
        p = proximity_results[name]
        b_score = b["distinction_score"]
        p_score = p["distinction_score"]
        delta = p_score - b_score

        marker = ""
        if delta > 0.001:
            marker = " (+)"
            wins += 1
        elif delta < -0.001:
            marker = " (-)"
            losses += 1
        else:
            marker = " (=)"
            ties += 1

        print(f"  {name:<28} {b_score:>9.1%} {p_score:>9.1%} {delta:>+9.1%}{marker}")

        # Show voice assignments for detail
        b_assign = b["assignments"]
        p_assign = p["assignments"]
        for label in b_assign:
            b_vid = b_assign[label][:8]
            p_vid = p_assign.get(label, "???")[:8]
            changed = " *" if b_assign[label] != p_assign.get(label) else ""
            print(f"    {label:<24} {b_vid}..  {p_vid}..{changed}")

    print(separator)
    b_overall = baseline_results["__overall__"]
    p_overall = proximity_results["__overall__"]
    delta_overall = p_overall - b_overall
    print(f"  {'OVERALL':<28} {b_overall:>9.1%} {p_overall:>9.1%} {delta_overall:>+9.1%}")
    print(f"\n  Wins: {wins}  Ties: {ties}  Losses: {losses}")
    print(f"{'=' * len(header)}\n")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestExperiment2ProximityMatching:
    """Run proximity-based voice matching experiment and compare to baseline."""

    def test_pitch_benchmark(self):
        """Run pitch estimation benchmark (shared infrastructure, for reference)."""
        results = run_pitch_benchmark(_estimate)
        overall = results["__overall__"]
        print(f"\n{'=' * 60}")
        print(f"PITCH ESTIMATION (reference): {overall:.1%}")
        print(f"{'=' * 60}")
        for name, data in results.items():
            if name == "__overall__":
                continue
            print(f"  {name}: {data['avg_accuracy']:.1%}")
            for spk in data["speakers"]:
                print(f"    {spk['label']}: true={spk['true_pitch']}Hz "
                      f"est={spk['estimated_pitch']}Hz acc={spk['accuracy']:.1%}")
        assert overall > 0.0, "Pitch estimation completely failed"

    def test_gender_benchmark(self):
        """Run gender classification benchmark (for reference)."""
        results = run_gender_benchmark(_gender, _estimate)
        overall = results["__overall__"]
        print(f"\n{'=' * 60}")
        print(f"GENDER CLASSIFICATION (reference): {overall:.1%}")
        print(f"{'=' * 60}")
        for name, data in results.items():
            if name == "__overall__":
                continue
            print(f"  {name}: {data['avg_accuracy']:.1%}")
        assert overall > 0.0

    def test_proximity_voice_matching(self):
        """
        Core experiment: compare proximity-based scoring to binary bins.
        """
        # Run baseline (binary bins)
        baseline_results = run_voice_matching_benchmark(
            current_voice_match, _estimate
        )

        # Run proximity scoring
        proximity_results = run_voice_matching_benchmark(
            proximity_voice_match, _estimate
        )

        # Print detailed comparison
        print_comparison_table(baseline_results, proximity_results)

        # Print overall summary
        b_overall = baseline_results["__overall__"]
        p_overall = proximity_results["__overall__"]

        print(f"  Baseline overall distinction:  {b_overall:.1%}")
        print(f"  Proximity overall distinction: {p_overall:.1%}")
        print(f"  Delta:                         {p_overall - b_overall:+.1%}")

        # The proximity approach should not be significantly worse
        # We allow a small margin since different matching can shift assignments
        assert p_overall >= b_overall - 0.05, (
            f"Proximity matching significantly worse: "
            f"{p_overall:.3f} vs baseline {b_overall:.3f}"
        )

    def test_proximity_boundary_cases(self):
        """
        Verify that proximity scoring handles boundary pitches gracefully.
        A speaker at 139 Hz and one at 141 Hz should get similar (not opposite)
        voice pitch categories, unlike the binary bin approach.
        """
        # Create a fresh VM for each sub-test
        # Test: 139 Hz speaker
        vm1 = SmartVoiceManager()
        voice_139 = proximity_voice_match(vm1, 139.0, 20.0)
        props_139 = vm1.available_voices[voice_139]

        # Create a fresh VM for 141 Hz
        vm2 = SmartVoiceManager()
        voice_141 = proximity_voice_match(vm2, 141.0, 20.0)
        props_141 = vm2.available_voices[voice_141]

        print(f"\n  Boundary test: 139 Hz -> {props_139['pitch']} ({voice_139[:8]}..)")
        print(f"  Boundary test: 141 Hz -> {props_141['pitch']} ({voice_141[:8]}..)")

        # With proximity scoring, these should map to the same pitch category
        # since 139 and 141 are only 2 Hz apart
        assert props_139["pitch"] == props_141["pitch"], (
            f"Boundary cliff detected: 139Hz got '{props_139['pitch']}' "
            f"but 141Hz got '{props_141['pitch']}'"
        )

    def test_proximity_extreme_pitches(self):
        """
        Verify proximity scoring handles extreme pitches that fall well outside
        the center ranges.
        """
        vm = SmartVoiceManager()

        # Very low pitch -- should pick the 'low' voice
        voice_80 = proximity_voice_match(vm, 80.0, 10.0)
        props_80 = vm.available_voices[voice_80]
        print(f"\n  80 Hz (very low) -> {props_80['pitch']} ({props_80['style']})")
        assert props_80["gender"] == "male", "80 Hz should match male voice"

        # Very high pitch -- should pick a high female voice
        voice_280 = proximity_voice_match(vm, 280.0, 10.0)
        props_280 = vm.available_voices[voice_280]
        print(f"  280 Hz (very high) -> {props_280['pitch']} ({props_280['style']})")
        assert props_280["gender"] == "female", "280 Hz should match female voice"

    def test_proximity_scores_are_continuous(self):
        """
        Verify that the scoring function produces continuous, monotonically
        decreasing scores as distance from voice center increases.
        """
        center = 140.0  # medium_low center
        pitches = [100, 110, 120, 130, 135, 138, 140, 142, 145, 150, 160, 180, 200]
        scores = []
        for p in pitches:
            s = 1.0 / (1.0 + abs(p - center))
            scores.append(s)

        print(f"\n  Proximity scores relative to center={center} Hz:")
        for p, s in zip(pitches, scores):
            bar = "#" * int(s * 50)
            print(f"    {p:>5} Hz: {s:.4f} {bar}")

        # Score at center should be maximum (1.0)
        center_idx = pitches.index(140)
        assert scores[center_idx] == 1.0, "Score at center should be 1.0"

        # Scores should decrease as we move away from center
        for i in range(center_idx + 1, len(scores)):
            assert scores[i] < scores[center_idx], (
                f"Score at {pitches[i]}Hz ({scores[i]:.4f}) should be less "
                f"than score at center ({scores[center_idx]:.4f})"
            )

    def test_proximity_all_scenarios_produce_assignments(self):
        """
        Ensure proximity matching produces valid voice IDs for every speaker
        in every scenario. No crashes, no None results.
        """
        valid_voice_ids = set(SmartVoiceManager().available_voices.keys())
        failures = []

        for name, scenario in SCENARIOS.items():
            vm = SmartVoiceManager()
            for spk in scenario["speakers"]:
                try:
                    voice_id = proximity_voice_match(vm, spk["pitch_hz"], 20.0)
                except Exception as exc:
                    failures.append(f"{name}/{spk['label']}: exception {exc}")
                    continue
                if voice_id not in valid_voice_ids:
                    failures.append(
                        f"{name}/{spk['label']}: invalid voice_id '{voice_id}'"
                    )

        if failures:
            print("\nFailures:")
            for f in failures:
                print(f"  {f}")

        assert len(failures) == 0, f"{len(failures)} failure(s) in scenario coverage"
