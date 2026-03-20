# tests/experiment_5_combined.py
"""
Experiment 5: Combined Best-of-All Improvements
================================================
Combines four improvements into one unified solution:
  1. YIN pitch estimation (better octave handling)
  2. Proximity voice matching (continuous distance instead of bins)
  3. Adaptive continuity threshold (replaces fixed 30Hz)
  4. Expanded voice library (20+ voices)

Tests all four benchmarks and prints a comprehensive comparison table.
"""
import sys, os
import numpy as np
import pytest
from typing import Dict, Tuple, Optional, List

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import benchmark harness
from tests.test_diarizer_experiments import (
    run_pitch_benchmark,
    run_gender_benchmark,
    run_voice_matching_benchmark,
    run_continuity_benchmark,
    pitch_accuracy,
    gender_accuracy,
    voice_distinction_score,
    continuity_score,
    make_tone,
    make_multi_speaker_audio,
    SCENARIOS,
    # Baseline functions for comparison
    _current_estimate,
    _current_gender,
    _current_voice_match,
    _current_continuity,
    SmartVoiceManager,
)


# ============================================================================
# (A) YIN Pitch Estimator -- pure numpy
# ============================================================================

def _yin_difference(frame: np.ndarray, max_tau: int) -> np.ndarray:
    """
    Compute the YIN difference function d(tau) for a single frame.
    d(tau) = sum_j (x[j] - x[j + tau])^2
    Uses the efficient autocorrelation-based formulation:
      d(tau) = r(0) + r_shifted(0) - 2*r(tau)
    """
    n = len(frame)
    # d[0] is always 0 by definition
    d = np.zeros(max_tau, dtype=np.float64)
    # Cumulative energy terms
    # r(0) = sum(x[j]^2 for j in 0..n-tau-1)
    # r_shifted(0) = sum(x[j+tau]^2 for j in 0..n-tau-1)
    for tau in range(1, max_tau):
        if tau >= n:
            break
        window = n - tau
        diff = frame[:window] - frame[tau:tau + window]
        d[tau] = np.sum(diff * diff)
    return d


def _yin_cmnd(d: np.ndarray) -> np.ndarray:
    """
    Cumulative Mean Normalized Difference function.
    d'(0) = 1, d'(tau) = d(tau) / ((1/tau) * sum(d[j], j=1..tau))
    This eliminates the bias towards tau=0 and helps find the true fundamental.
    """
    n = len(d)
    cmnd = np.ones(n, dtype=np.float64)
    running_sum = 0.0
    for tau in range(1, n):
        running_sum += d[tau]
        if running_sum == 0:
            cmnd[tau] = 1.0
        else:
            cmnd[tau] = d[tau] * tau / running_sum
    return cmnd


def yin_estimate_pitch(audio: np.ndarray, sr: int,
                       fmin: float = 50.0, fmax: float = 350.0,
                       threshold: float = 0.15,
                       frame_dur: float = 0.04,
                       hop_dur: float = 0.01) -> Tuple[float, float]:
    """
    YIN pitch estimator (pure numpy, no librosa/numba).

    Steps per frame:
      1. Compute difference function d(tau)
      2. Compute cumulative mean normalized difference d'(tau)
      3. Absolute threshold: find first tau where d'(tau) < threshold
      4. Parabolic interpolation around the minimum for sub-sample accuracy

    Returns (median_pitch_hz, std_pitch_hz).
    Falls back to (150.0, 20.0) on failure.
    """
    audio = audio.astype(np.float64)
    frame_len = int(sr * frame_dur)
    hop_len = max(1, int(sr * hop_dur))

    # Lag limits from frequency range
    min_tau = max(2, int(sr / fmax))
    max_tau = min(frame_len // 2, int(sr / fmin))

    if max_tau <= min_tau + 1:
        return 150.0, 20.0

    pitches = []

    for start in range(0, max(1, len(audio) - frame_len + 1), hop_len):
        frame = audio[start:start + frame_len]
        if len(frame) < frame_len:
            break

        # Skip silence
        if np.max(np.abs(frame)) < 1e-4:
            continue

        # Step 1: Difference function
        d = _yin_difference(frame, max_tau + 1)

        # Step 2: CMND
        cmnd = _yin_cmnd(d)

        # Step 3: Absolute threshold -- find first dip below threshold
        # in the valid range [min_tau, max_tau]
        best_tau = None
        for tau in range(min_tau, min(max_tau + 1, len(cmnd))):
            if cmnd[tau] < threshold:
                # Found a dip -- now find the local minimum from here
                while (tau + 1 < min(max_tau + 1, len(cmnd)) and
                       cmnd[tau + 1] < cmnd[tau]):
                    tau += 1
                best_tau = tau
                break

        # If no dip below threshold, pick the global minimum in range
        if best_tau is None:
            search_range = cmnd[min_tau:min(max_tau + 1, len(cmnd))]
            if len(search_range) == 0:
                continue
            best_tau = int(np.argmin(search_range)) + min_tau
            # Only accept if the minimum is reasonably low
            if cmnd[best_tau] > 0.5:
                continue

        # Step 4: Parabolic interpolation for sub-sample accuracy
        if 1 <= best_tau - 1 and best_tau + 1 < len(cmnd):
            alpha = cmnd[best_tau - 1]
            beta = cmnd[best_tau]
            gamma = cmnd[best_tau + 1]
            denom = 2.0 * (2.0 * beta - alpha - gamma)
            if abs(denom) > 1e-10:
                shift = (alpha - gamma) / denom
                refined_tau = best_tau + shift
            else:
                refined_tau = float(best_tau)
        else:
            refined_tau = float(best_tau)

        if refined_tau > 0:
            pitch = sr / refined_tau
            if fmin <= pitch <= fmax:
                pitches.append(pitch)

    if not pitches:
        return 150.0, 20.0

    arr = np.array(pitches, dtype=np.float64)

    # Outlier filtering: remove pitches more than 1.5 IQR outside Q1/Q3
    if len(arr) > 4:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
            filtered = arr[mask]
            if len(filtered) > 0:
                arr = filtered

    return float(np.median(arr)), float(np.std(arr))


# ============================================================================
# (B) Improved Gender Classification
# ============================================================================

def improved_gender(pitch_hz: float, pitch_range: float = None) -> str:
    """
    Wider deadband gender classification with confidence.
    Male < 150 Hz, Female > 190 Hz. Middle zone = use pitch_range hint.
    """
    if pitch_hz is None:
        return "unknown"
    if pitch_hz < 150.0:
        return "male"
    if pitch_hz > 190.0:
        return "female"

    # Ambiguous zone 150-190 Hz: use range as a tiebreaker
    # Higher pitch range (more variation) is slightly more female-associated
    # But mostly just pick the closer boundary
    if pitch_hz < 170.0:
        return "male"
    else:
        return "female"


# ============================================================================
# (C) Expanded Voice Manager with Proximity Scoring (20+ voices)
# ============================================================================

class CombinedVoiceManager:
    """
    Expanded voice library with 20+ voices.
    Uses numeric pitch centers and proximity-based scoring
    instead of categorical pitch bins.
    """

    def __init__(self):
        # 24 voices with numeric pitch_center for proximity scoring
        self.available_voices = {
            # Male voices -- low range (80-130 Hz)
            "male_bass_01": {
                "gender": "male", "pitch_center": 90.0,
                "style": "deep_authoritative", "sports": True,
            },
            "male_bass_02": {
                "gender": "male", "pitch_center": 100.0,
                "style": "deep_calm", "sports": False,
            },
            "male_low_01": {
                "gender": "male", "pitch_center": 110.0,
                "style": "strong_confident", "sports": True,
            },
            "male_low_02": {
                "gender": "male", "pitch_center": 120.0,
                "style": "warm_narrator", "sports": True,
            },
            "male_low_03": {
                "gender": "male", "pitch_center": 125.0,
                "style": "crisp_professional", "sports": True,
            },
            # Male voices -- mid range (130-165 Hz)
            "male_mid_01": {
                "gender": "male", "pitch_center": 130.0,
                "style": "energetic_young", "sports": True,
            },
            "male_mid_02": {
                "gender": "male", "pitch_center": 137.0,
                "style": "well_rounded", "sports": True,
            },
            "male_mid_03": {
                "gender": "male", "pitch_center": 145.0,
                "style": "casual_conversational", "sports": False,
            },
            "male_mid_04": {
                "gender": "male", "pitch_center": 150.0,
                "style": "dynamic_sports", "sports": True,
            },
            "male_mid_05": {
                "gender": "male", "pitch_center": 155.0,
                "style": "excited_pbp", "sports": True,
            },
            "male_mid_06": {
                "gender": "male", "pitch_center": 163.0,
                "style": "bright_analyst", "sports": True,
            },
            # Female voices -- mid range (170-220 Hz)
            "female_mid_01": {
                "gender": "female", "pitch_center": 175.0,
                "style": "strong_confident", "sports": True,
            },
            "female_mid_02": {
                "gender": "female", "pitch_center": 185.0,
                "style": "warm_calm", "sports": False,
            },
            "female_mid_03": {
                "gender": "female", "pitch_center": 190.0,
                "style": "energetic_reporter", "sports": True,
            },
            "female_mid_04": {
                "gender": "female", "pitch_center": 200.0,
                "style": "emotional_expressive", "sports": False,
            },
            "female_mid_05": {
                "gender": "female", "pitch_center": 210.0,
                "style": "crisp_professional", "sports": True,
            },
            "female_mid_06": {
                "gender": "female", "pitch_center": 215.0,
                "style": "dynamic_sideline", "sports": True,
            },
            # Female voices -- high range (220-300 Hz)
            "female_high_01": {
                "gender": "female", "pitch_center": 225.0,
                "style": "soft_friendly", "sports": False,
            },
            "female_high_02": {
                "gender": "female", "pitch_center": 240.0,
                "style": "bright_energetic", "sports": True,
            },
            "female_high_03": {
                "gender": "female", "pitch_center": 255.0,
                "style": "youthful_vibrant", "sports": False,
            },
            "female_high_04": {
                "gender": "female", "pitch_center": 270.0,
                "style": "clear_soprano", "sports": False,
            },
            "female_high_05": {
                "gender": "female", "pitch_center": 285.0,
                "style": "high_expressive", "sports": False,
            },
            # Extra male voices for closely spaced pitches
            "male_high_01": {
                "gender": "male", "pitch_center": 168.0,
                "style": "tenor_pbp", "sports": True,
            },
            "male_high_02": {
                "gender": "male", "pitch_center": 172.0,
                "style": "tenor_analyst", "sports": True,
            },
        }
        self.used_voice_ids = set()

    def _match_best_voice(self, gender: str, pitch_hz: float,
                          pitch_range: float) -> str:
        """
        Proximity-based voice matching:
          score = proximity + gender_bonus + sports_bonus - used_penalty
        proximity = 1.0 / (1.0 + abs(pitch - center) / 30.0)
        """
        candidates = {}

        # First pass: prefer unused voices of the correct gender
        for vid, props in self.available_voices.items():
            if vid in self.used_voice_ids:
                continue
            if props["gender"] == gender:
                candidates[vid] = props

        # Fallback: unused voices of any gender
        if not candidates:
            for vid, props in self.available_voices.items():
                if vid not in self.used_voice_ids:
                    candidates[vid] = props

        # Fallback: allow reuse of same-gender voices
        if not candidates:
            for vid, props in self.available_voices.items():
                if props["gender"] == gender:
                    candidates[vid] = props

        # Last resort: everything
        if not candidates:
            candidates = dict(self.available_voices)

        best_vid = None
        best_score = -float("inf")

        for vid, props in candidates.items():
            center = props["pitch_center"]
            # Proximity: continuous score, highest when pitch == center
            proximity = 1.0 / (1.0 + abs(pitch_hz - center) / 30.0)

            # Gender match bonus
            gender_bonus = 0.3 if props["gender"] == gender else 0.0

            # Sports commentary style bonus
            sports_bonus = 0.1 if props.get("sports", False) else 0.0

            score = proximity + gender_bonus + sports_bonus

            if score > best_score:
                best_score = score
                best_vid = vid

        if best_vid is None:
            best_vid = list(self.available_voices.keys())[0]

        self.used_voice_ids.add(best_vid)
        return best_vid


# ============================================================================
# (D) Adaptive Continuity with EMA and Dynamic Threshold
# ============================================================================

def combined_continuity(pitch_hz: float, session_state: dict) -> str:
    """
    Adaptive continuity:
      1. Exponential moving average (alpha=0.3) on speaker pitches
      2. Adaptive threshold = min_gap / 2 (from inter-speaker distances)
         with a floor of 15 Hz and a ceiling of 40 Hz
      3. Margin requirement: best match must beat second-best by >= margin
         to avoid ambiguous reassignments
    """
    EMA_ALPHA = 0.3
    THRESHOLD_FLOOR = 15.0
    THRESHOLD_CEILING = 40.0
    MARGIN_RATIO = 0.4  # best must be margin_ratio * threshold better than 2nd

    pitches = session_state.get("speaker_pitches", {})
    voice_ids = session_state.get("speaker_voice_ids", {})

    # Compute adaptive threshold from inter-speaker gaps
    if len(pitches) >= 2:
        pitch_vals = sorted(pitches.values())
        gaps = [pitch_vals[i + 1] - pitch_vals[i] for i in range(len(pitch_vals) - 1)]
        min_gap = min(gaps) if gaps else 60.0
        threshold = max(THRESHOLD_FLOOR, min(THRESHOLD_CEILING, min_gap / 2.0))
    else:
        threshold = THRESHOLD_CEILING  # generous when few speakers known

    # Find best and second-best matches
    matches = []
    for spk_id, spk_pitch in pitches.items():
        dist = abs(pitch_hz - spk_pitch)
        matches.append((dist, spk_id))

    matches.sort(key=lambda x: x[0])

    if matches:
        best_dist, best_spk = matches[0]

        # Check margin: if there is a second match, ensure sufficient separation
        margin_ok = True
        if len(matches) >= 2:
            second_dist = matches[1][0]
            margin = threshold * MARGIN_RATIO
            if best_dist < threshold and (second_dist - best_dist) < margin:
                # Ambiguous -- still accept if best_dist is very small
                if best_dist > threshold * 0.5:
                    margin_ok = False

        if best_dist <= threshold and margin_ok:
            # Match found -- update EMA
            old_pitch = pitches[best_spk]
            new_pitch = EMA_ALPHA * pitch_hz + (1.0 - EMA_ALPHA) * old_pitch
            session_state["speaker_pitches"][best_spk] = new_pitch
            return voice_ids[best_spk]

    # No match -- assign new voice
    vm = CombinedVoiceManager()
    # Pass used voice IDs from session to prevent collisions
    for vid in voice_ids.values():
        vm.used_voice_ids.add(vid)

    gender = improved_gender(pitch_hz)
    voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)

    spk_id = f"SPK_{len(voice_ids):02d}"
    session_state["speaker_voice_ids"][spk_id] = voice_id
    session_state["speaker_pitches"][spk_id] = pitch_hz
    return voice_id


# ============================================================================
# Wrapper functions for the benchmark harness
# ============================================================================

def combined_estimate(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """Wrap YIN estimator to match benchmark interface."""
    return yin_estimate_pitch(audio, sr)


def combined_gender_fn(pitch_hz: float, pitch_range: float = None) -> str:
    """Wrap improved gender for benchmark interface."""
    return improved_gender(pitch_hz, pitch_range)


def combined_voice_match(vm, pitch_hz: float, pitch_range: float) -> str:
    """
    Voice matching using proximity scoring.
    Note: the benchmark passes a SmartVoiceManager instance (baseline),
    but we use our own CombinedVoiceManager. We track used voices
    via a hack: we store a CombinedVoiceManager on the SmartVoiceManager
    instance so it persists across calls within a scenario.
    """
    if not hasattr(vm, "_combined_vm"):
        vm._combined_vm = CombinedVoiceManager()
    cvm = vm._combined_vm
    gender = improved_gender(pitch_hz, pitch_range)
    return cvm._match_best_voice(gender, pitch_hz, pitch_range or 20.0)


def combined_continuity_fn(pitch_hz: float, session_state: dict) -> str:
    """Wrap adaptive continuity for benchmark interface."""
    return combined_continuity(pitch_hz, session_state)


# ============================================================================
# Tests
# ============================================================================

class TestExperiment5Combined:
    """Run all four benchmarks with the combined improvements."""

    # Store results for the comparison table
    results = {}

    def test_01_pitch_estimation(self):
        """Benchmark: YIN pitch estimation vs baseline autocorrelation."""
        combined_results = run_pitch_benchmark(combined_estimate)
        baseline_results = run_pitch_benchmark(_current_estimate)

        combined_overall = combined_results["__overall__"]
        baseline_overall = baseline_results["__overall__"]

        TestExperiment5Combined.results["pitch"] = {
            "baseline": baseline_overall,
            "combined": combined_overall,
        }

        print(f"\n{'='*70}")
        print(f"PITCH ESTIMATION: Combined={combined_overall:.1%} "
              f"(Baseline={baseline_overall:.1%}, "
              f"Delta={combined_overall - baseline_overall:+.1%})")
        print(f"{'='*70}")

        for name, data in combined_results.items():
            if name == "__overall__":
                continue
            bl_data = baseline_results[name]
            print(f"  {name}: {data['avg_accuracy']:.1%} "
                  f"(baseline: {bl_data['avg_accuracy']:.1%})")
            for i, spk in enumerate(data["speakers"]):
                bl_spk = bl_data["speakers"][i]
                marker = " <<< IMPROVED" if spk["accuracy"] > bl_spk["accuracy"] + 0.005 else ""
                marker = " <<< REGRESSED" if spk["accuracy"] < bl_spk["accuracy"] - 0.005 else marker
                print(f"    {spk['label']}: true={spk['true_pitch']}Hz "
                      f"est={spk['estimated_pitch']}Hz "
                      f"acc={spk['accuracy']:.1%} "
                      f"(was {bl_spk['estimated_pitch']}Hz / {bl_spk['accuracy']:.1%}){marker}")

        assert combined_overall > 0.90, f"Pitch accuracy too low: {combined_overall:.1%}"

    def test_02_gender_classification(self):
        """Benchmark: improved gender (wider bands) vs baseline."""
        combined_results = run_gender_benchmark(combined_gender_fn, combined_estimate)
        baseline_results = run_gender_benchmark(_current_gender, _current_estimate)

        combined_overall = combined_results["__overall__"]
        baseline_overall = baseline_results["__overall__"]

        TestExperiment5Combined.results["gender"] = {
            "baseline": baseline_overall,
            "combined": combined_overall,
        }

        print(f"\n{'='*70}")
        print(f"GENDER CLASSIFICATION: Combined={combined_overall:.1%} "
              f"(Baseline={baseline_overall:.1%}, "
              f"Delta={combined_overall - baseline_overall:+.1%})")
        print(f"{'='*70}")

        for name, data in combined_results.items():
            if name == "__overall__":
                continue
            bl_data = baseline_results[name]
            print(f"  {name}: {data['avg_accuracy']:.1%} "
                  f"(baseline: {bl_data['avg_accuracy']:.1%})")
            for spk in data["speakers"]:
                print(f"    {spk['label']}: true_pitch={spk['true_pitch']}Hz "
                      f"pred={spk['predicted_gender']} "
                      f"{'OK' if spk['correct'] else 'WRONG'}")

        assert combined_overall >= 0.90, f"Gender accuracy too low: {combined_overall:.1%}"

    def test_03_voice_matching(self):
        """Benchmark: proximity scoring + expanded library vs baseline bins."""
        combined_results = run_voice_matching_benchmark(
            combined_voice_match, combined_estimate)
        baseline_results = run_voice_matching_benchmark(
            _current_voice_match, _current_estimate)

        combined_overall = combined_results["__overall__"]
        baseline_overall = baseline_results["__overall__"]

        TestExperiment5Combined.results["voice"] = {
            "baseline": baseline_overall,
            "combined": combined_overall,
        }

        print(f"\n{'='*70}")
        print(f"VOICE DISTINCTION: Combined={combined_overall:.1%} "
              f"(Baseline={baseline_overall:.1%}, "
              f"Delta={combined_overall - baseline_overall:+.1%})")
        print(f"{'='*70}")

        for name, data in combined_results.items():
            if name == "__overall__":
                continue
            bl_data = baseline_results[name]
            print(f"  {name}: distinction={data['distinction_score']:.1%} "
                  f"voices={data['unique_voices']}/{data['num_speakers']}spk "
                  f"(baseline: {bl_data['distinction_score']:.1%})")

        assert combined_overall >= 0.90, f"Voice distinction too low: {combined_overall:.1%}"

    def test_04_continuity(self):
        """Benchmark: adaptive EMA continuity vs fixed 30Hz threshold."""
        np.random.seed(42)
        combined_results = run_continuity_benchmark(
            combined_continuity_fn, combined_estimate)
        np.random.seed(42)
        baseline_results = run_continuity_benchmark(
            _current_continuity, _current_estimate)

        combined_overall = combined_results["__overall__"]
        baseline_overall = baseline_results["__overall__"]

        TestExperiment5Combined.results["continuity"] = {
            "baseline": baseline_overall,
            "combined": combined_overall,
        }

        print(f"\n{'='*70}")
        print(f"CONTINUITY: Combined={combined_overall:.1%} "
              f"(Baseline={baseline_overall:.1%}, "
              f"Delta={combined_overall - baseline_overall:+.1%})")
        print(f"{'='*70}")

        for name, val in combined_results.items():
            if name == "__overall__":
                continue
            bl_val = baseline_results[name]
            marker = ""
            if isinstance(val, float) and isinstance(bl_val, float):
                if val > bl_val + 0.005:
                    marker = " <<< IMPROVED"
                elif val < bl_val - 0.005:
                    marker = " <<< REGRESSED"
            print(f"  {name}: {val:.1%} (baseline: {bl_val:.1%}){marker}")

        assert combined_overall > 0.70, f"Continuity too low: {combined_overall:.1%}"

    def test_05_comparison_table(self):
        """Print the final comprehensive comparison table."""
        r = TestExperiment5Combined.results

        # If previous tests didn't run (e.g. running this test alone), compute now
        if "pitch" not in r:
            r["pitch"] = {
                "baseline": run_pitch_benchmark(_current_estimate)["__overall__"],
                "combined": run_pitch_benchmark(combined_estimate)["__overall__"],
            }
        if "gender" not in r:
            r["gender"] = {
                "baseline": run_gender_benchmark(_current_gender, _current_estimate)["__overall__"],
                "combined": run_gender_benchmark(combined_gender_fn, combined_estimate)["__overall__"],
            }
        if "voice" not in r:
            r["voice"] = {
                "baseline": run_voice_matching_benchmark(_current_voice_match, _current_estimate)["__overall__"],
                "combined": run_voice_matching_benchmark(combined_voice_match, combined_estimate)["__overall__"],
            }
        if "continuity" not in r:
            np.random.seed(42)
            bl_cont = run_continuity_benchmark(_current_continuity, _current_estimate)["__overall__"]
            np.random.seed(42)
            cb_cont = run_continuity_benchmark(combined_continuity_fn, combined_estimate)["__overall__"]
            r["continuity"] = {"baseline": bl_cont, "combined": cb_cont}

        print(f"\n")
        print(f"{'='*70}")
        print(f"  EXPERIMENT 5: COMBINED BEST-OF-ALL -- FINAL RESULTS")
        print(f"{'='*70}")
        print(f"")
        print(f"  {'Metric':<24s} | {'Baseline':>10s} | {'Combined':>10s} | {'Delta':>10s}")
        print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        metrics = [
            ("Pitch Estimation", "pitch"),
            ("Gender Classification", "gender"),
            ("Voice Distinction", "voice"),
            ("Continuity", "continuity"),
        ]

        total_baseline = 0.0
        total_combined = 0.0

        for label, key in metrics:
            bl = r[key]["baseline"]
            cb = r[key]["combined"]
            delta = cb - bl
            total_baseline += bl
            total_combined += cb
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<24s} | {bl:>9.1%} | {cb:>9.1%} | {sign}{delta:>8.1%}")

        avg_bl = total_baseline / len(metrics)
        avg_cb = total_combined / len(metrics)
        avg_delta = avg_cb - avg_bl
        sign = "+" if avg_delta >= 0 else ""
        print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        print(f"  {'AVERAGE':<24s} | {avg_bl:>9.1%} | {avg_cb:>9.1%} | {sign}{avg_delta:>8.1%}")
        print(f"{'='*70}")
        print(f"")

        # The combined solution should not regress on any metric
        for label, key in metrics:
            bl = r[key]["baseline"]
            cb = r[key]["combined"]
            # Allow small tolerance for stochastic variance
            assert cb >= bl - 0.02, (
                f"{label} regressed: baseline={bl:.1%}, combined={cb:.1%}"
            )
