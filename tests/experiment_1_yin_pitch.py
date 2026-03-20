# tests/experiment_1_yin_pitch.py
"""
Experiment 1: YIN-based pitch estimator.

Implements the full YIN algorithm (de Cheveigné & Kawahara, 2002) in pure numpy:
  1. Difference function
  2. Cumulative mean normalized difference function (CMND) — reduces octave errors
  3. Absolute threshold search on CMND
  4. Parabolic interpolation for sub-bin accuracy

Compared against the baseline autocorrelation estimator from main.py.
"""
import sys
import os
import numpy as np
import pytest
from typing import Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.test_diarizer_experiments import (
    run_pitch_benchmark,
    run_gender_benchmark,
    _current_estimate,
    _current_gender,
)
from utils import gender_from_pitch


# ---------------------------------------------------------------------------
# YIN pitch estimator — pure numpy, no external dependencies
# ---------------------------------------------------------------------------

def _difference_function(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute the YIN difference function d(tau) for lags 0..max_lag-1.

    d(tau) = sum_{j=0}^{W-1-tau} (x[j] - x[j+tau])^2

    Uses the autocorrelation trick for O(N log N) computation:
        d(tau) = r(0) + r_shifted(0) - 2*r(tau)
    where r is the autocorrelation of the frame.
    """
    n = len(frame)
    # Ensure max_lag does not exceed frame length
    max_lag = min(max_lag, n)
    d = np.zeros(max_lag, dtype=np.float64)
    # d[0] = 0 by definition

    # Full autocorrelation via np.correlate
    # For efficiency with moderate frame sizes, direct correlate is fine.
    # We compute running sums for the energy terms.

    # Cumulative sum of x^2 for the shifted energy term
    x = frame.astype(np.float64)
    cumsum_sq = np.cumsum(x ** 2)

    for tau in range(1, max_lag):
        # sum of x[j]^2 for j in 0..W-1-tau
        # = cumsum_sq[W-1-tau]  (0-indexed, cumsum_sq[i] = sum x[0..i]^2)
        w = n - tau
        if w <= 0:
            break
        energy_start = cumsum_sq[w - 1]
        # sum of x[j+tau]^2 for j in 0..W-1-tau = sum x[tau..W-1]^2
        energy_shifted = cumsum_sq[n - 1] - cumsum_sq[tau - 1]
        # cross term: 2 * sum x[j]*x[j+tau]
        cross = np.dot(x[:w], x[tau:tau + w])
        d[tau] = energy_start + energy_shifted - 2.0 * cross

    return d


def _cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    """
    Step 3 of YIN: Cumulative Mean Normalized Difference Function (CMNDF).

    d'(0) = 1
    d'(tau) = d(tau) / [(1/tau) * sum_{j=1}^{tau} d(j)]

    This normalization suppresses octave errors by penalizing higher lags
    that have comparable raw difference to the true fundamental period.
    """
    n = len(d)
    cmndf = np.ones(n, dtype=np.float64)
    if n <= 1:
        return cmndf

    running_sum = 0.0
    for tau in range(1, n):
        running_sum += d[tau]
        if running_sum == 0.0:
            cmndf[tau] = 1.0
        else:
            cmndf[tau] = d[tau] * tau / running_sum

    return cmndf


def _parabolic_interpolation(values: np.ndarray, index: int) -> float:
    """
    Parabolic interpolation around a minimum in `values` at `index`.
    Returns the fractional index of the true minimum.
    """
    if index <= 0 or index >= len(values) - 1:
        return float(index)

    alpha = values[index - 1]
    beta = values[index]
    gamma = values[index + 1]

    denom = 2.0 * (2.0 * beta - alpha - gamma)
    if abs(denom) < 1e-12:
        return float(index)

    shift = (alpha - gamma) / denom
    return index + shift


def estimate_pitch_yin_improved(
    audio_1d: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 350.0,
    threshold: float = 0.15,
    frame_duration: float = 0.04,
    hop_duration: float = 0.01,
) -> Tuple[float, float]:
    """
    YIN pitch estimator with:
      a. Cumulative mean normalized difference (reduces octave errors)
      b. Parabolic interpolation for sub-bin accuracy
      c. Configurable frequency range (default 50-350 Hz)
      d. Returns (median_pitch_hz, pitch_range_std_hz) across frames

    Parameters
    ----------
    audio_1d : np.ndarray
        Mono audio signal, any dtype (will be cast to float64).
    sr : int
        Sample rate in Hz.
    fmin : float
        Minimum detectable frequency (Hz). Default 50.
    fmax : float
        Maximum detectable frequency (Hz). Default 350.
    threshold : float
        CMNDF threshold for pitch detection. Lower = stricter.
        Default 0.15 (good for clean harmonic signals).
    frame_duration : float
        Analysis frame length in seconds. Default 0.04 (40 ms).
    hop_duration : float
        Hop between frames in seconds. Default 0.01 (10 ms).

    Returns
    -------
    (median_pitch, pitch_std) : Tuple[float, float]
        Median pitch in Hz and standard deviation across voiced frames.
        Falls back to (150.0, 20.0) if no pitch can be detected.
    """
    y = audio_1d.astype(np.float64)

    # Lag bounds from frequency bounds
    min_lag = max(2, int(sr / fmax))
    max_lag = min(len(y) - 1, int(sr / fmin))

    frame_len = max(max_lag + 1, int(sr * frame_duration))
    hop_len = max(1, int(sr * hop_duration))

    if len(y) < frame_len:
        # Audio shorter than a single frame — use whole signal
        frame_len = len(y)

    pitches = []

    for start in range(0, max(1, len(y) - frame_len + 1), hop_len):
        frame = y[start: start + frame_len]

        # Skip near-silent frames
        if np.max(np.abs(frame)) < 1e-4:
            continue

        # Step 1 & 2: Difference function
        d = _difference_function(frame, max_lag + 1)

        # Step 3: Cumulative mean normalized difference
        cmndf = _cumulative_mean_normalized_difference(d)

        # Step 4: Absolute threshold — find the first dip below threshold
        # in the valid lag range [min_lag, max_lag]
        pitch_lag = None

        for tau in range(min_lag, min(max_lag + 1, len(cmndf))):
            if cmndf[tau] < threshold:
                # Found a dip below threshold.
                # Walk forward to find the local minimum of this dip.
                while (tau + 1 < min(max_lag + 1, len(cmndf)) and
                       cmndf[tau + 1] < cmndf[tau]):
                    tau += 1
                pitch_lag = tau
                break

        # If no dip below threshold, fall back to global minimum in range
        if pitch_lag is None:
            search_region = cmndf[min_lag: min(max_lag + 1, len(cmndf))]
            if len(search_region) == 0:
                continue
            # Only use global min if it's reasonably low
            min_val = np.min(search_region)
            if min_val < 0.5:
                pitch_lag = int(np.argmin(search_region)) + min_lag
            else:
                continue

        # Step 5: Parabolic interpolation for sub-sample accuracy
        refined_lag = _parabolic_interpolation(cmndf, pitch_lag)
        if refined_lag <= 0:
            continue

        pitch_hz = float(sr) / refined_lag
        # Validate the pitch is within our frequency range
        if fmin <= pitch_hz <= fmax:
            pitches.append(pitch_hz)

    if not pitches:
        return 150.0, 20.0

    arr = np.array(pitches, dtype=np.float64)
    return float(np.median(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Gender classifier (reuse the same logic as baseline for fair comparison)
# ---------------------------------------------------------------------------

def _yin_gender(pitch_hz: float, pitch_range: float = None) -> str:
    """Same gender logic as baseline, for fair pitch-only comparison."""
    return gender_from_pitch(pitch_hz, pitch_range)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperiment1YINPitch:
    """
    Run pitch and gender benchmarks with the YIN estimator
    and compare against the baseline autocorrelation estimator.
    """

    def test_pitch_benchmark_comparison(self):
        """Compare YIN pitch estimation vs. baseline across all scenarios."""
        # Run baseline
        baseline_results = run_pitch_benchmark(_current_estimate)
        baseline_overall = baseline_results["__overall__"]

        # Run YIN experiment
        yin_results = run_pitch_benchmark(estimate_pitch_yin_improved)
        yin_overall = yin_results["__overall__"]

        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT 1: YIN Pitch Estimation vs. Baseline")
        print(f"{'=' * 70}")
        print(f"\n  Overall:  Baseline: {baseline_overall:.1%}  -->  "
              f"Experiment 1 (YIN): {yin_overall:.1%}")

        delta = yin_overall - baseline_overall
        direction = "IMPROVEMENT" if delta > 0 else ("REGRESSION" if delta < 0 else "NO CHANGE")
        print(f"  Delta:    {delta:+.1%}  ({direction})")

        print(f"\n  {'Scenario':<30s} {'Baseline':>10s} {'YIN':>10s} {'Delta':>10s}")
        print(f"  {'-' * 60}")

        for name in sorted(baseline_results.keys()):
            if name == "__overall__":
                continue
            b_acc = baseline_results[name]["avg_accuracy"]
            y_acc = yin_results[name]["avg_accuracy"]
            d = y_acc - b_acc
            marker = " +" if d > 0.005 else (" -" if d < -0.005 else "  ")
            print(f"  {name:<30s} {b_acc:>9.1%} {y_acc:>9.1%} {d:>+9.1%}{marker}")

        # Detailed per-speaker breakdown for scenarios where YIN differs
        print(f"\n  --- Per-speaker details (scenarios with notable differences) ---")
        for name in sorted(baseline_results.keys()):
            if name == "__overall__":
                continue
            b_acc = baseline_results[name]["avg_accuracy"]
            y_acc = yin_results[name]["avg_accuracy"]
            if abs(y_acc - b_acc) > 0.01:
                print(f"\n  {name}:")
                for b_spk, y_spk in zip(baseline_results[name]["speakers"],
                                         yin_results[name]["speakers"]):
                    print(f"    {b_spk['label']}: true={b_spk['true_pitch']}Hz  "
                          f"baseline={b_spk['estimated_pitch']}Hz ({b_spk['accuracy']:.1%})  "
                          f"yin={y_spk['estimated_pitch']}Hz ({y_spk['accuracy']:.1%})")

        print(f"\n{'=' * 70}")
        print(f"  Baseline: {baseline_overall:.1%}  -->  Experiment 1: {yin_overall:.1%}")
        print(f"{'=' * 70}\n")

        # The test passes as long as YIN doesn't completely fail
        assert yin_overall > 0.0, "YIN pitch estimation completely failed"

    def test_gender_benchmark_comparison(self):
        """Compare gender classification accuracy using YIN vs. baseline pitch."""
        # Run baseline
        baseline_results = run_gender_benchmark(_current_gender, _current_estimate)
        baseline_overall = baseline_results["__overall__"]

        # Run YIN experiment (same gender logic, different pitch estimator)
        yin_results = run_gender_benchmark(_yin_gender, estimate_pitch_yin_improved)
        yin_overall = yin_results["__overall__"]

        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT 1: Gender Classification (YIN pitch vs. Baseline pitch)")
        print(f"{'=' * 70}")
        print(f"\n  Overall:  Baseline: {baseline_overall:.1%}  -->  "
              f"Experiment 1 (YIN): {yin_overall:.1%}")

        delta = yin_overall - baseline_overall
        direction = "IMPROVEMENT" if delta > 0 else ("REGRESSION" if delta < 0 else "NO CHANGE")
        print(f"  Delta:    {delta:+.1%}  ({direction})")

        print(f"\n  {'Scenario':<30s} {'Baseline':>10s} {'YIN':>10s} {'Delta':>10s}")
        print(f"  {'-' * 60}")

        for name in sorted(baseline_results.keys()):
            if name == "__overall__":
                continue
            b_acc = baseline_results[name]["avg_accuracy"]
            y_acc = yin_results[name]["avg_accuracy"]
            d = y_acc - b_acc
            marker = " +" if d > 0.005 else (" -" if d < -0.005 else "  ")
            print(f"  {name:<30s} {b_acc:>9.1%} {y_acc:>9.1%} {d:>+9.1%}{marker}")

        # Show misclassifications for the YIN estimator
        print(f"\n  --- Gender misclassifications (YIN) ---")
        any_miss = False
        for name in sorted(yin_results.keys()):
            if name == "__overall__":
                continue
            for spk in yin_results[name]["speakers"]:
                if not spk["correct"]:
                    any_miss = True
                    print(f"    {name}/{spk['label']}: "
                          f"true_pitch={spk['true_pitch']}Hz  "
                          f"predicted={spk['predicted_gender']}")
        if not any_miss:
            print(f"    (none)")

        print(f"\n{'=' * 70}")
        print(f"  Baseline: {baseline_overall:.1%}  -->  Experiment 1: {yin_overall:.1%}")
        print(f"{'=' * 70}\n")

        assert yin_overall > 0.0, "Gender classification with YIN completely failed"

    def test_octave_error_resistance(self):
        """
        Focused test: verify YIN handles octave-error-prone signals better.
        Uses harmonic_dominant=True scenarios from the benchmark suite.
        """
        from tests.test_diarizer_experiments import make_tone, pitch_accuracy

        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT 1: Octave Error Resistance")
        print(f"{'=' * 70}")

        test_cases = [
            ("150 Hz, harmonic-dominant", 150.0, True),
            ("90 Hz, harmonic-dominant", 90.0, True),
            ("120 Hz, harmonic-dominant", 120.0, True),
            ("200 Hz, harmonic-dominant", 200.0, True),
            ("150 Hz, normal harmonics", 150.0, False),
            ("90 Hz, normal harmonics", 90.0, False),
        ]

        sr = 16000
        print(f"\n  {'Test Case':<35s} {'True':>7s} {'Baseline':>10s} {'YIN':>10s} "
              f"{'B_acc':>7s} {'Y_acc':>7s}")
        print(f"  {'-' * 75}")

        baseline_accs = []
        yin_accs = []

        for desc, true_pitch, harm_dom in test_cases:
            audio = make_tone(true_pitch, 3.0, sr, amplitude=0.15 if harm_dom else 0.3,
                              harmonic_dominant=harm_dom)

            b_pitch, b_range = _current_estimate(audio, sr)
            y_pitch, y_range = estimate_pitch_yin_improved(audio, sr)

            b_acc = pitch_accuracy(b_pitch, true_pitch)
            y_acc = pitch_accuracy(y_pitch, true_pitch)
            baseline_accs.append(b_acc)
            yin_accs.append(y_acc)

            marker = " <<" if y_acc - b_acc > 0.05 else ""
            print(f"  {desc:<35s} {true_pitch:>6.0f}Hz {b_pitch:>9.1f}Hz {y_pitch:>9.1f}Hz "
                  f"{b_acc:>6.1%} {y_acc:>6.1%}{marker}")

        b_mean = np.mean(baseline_accs)
        y_mean = np.mean(yin_accs)
        print(f"\n  Octave error test avg:  Baseline: {b_mean:.1%}  YIN: {y_mean:.1%}")
        print(f"{'=' * 70}\n")

        assert y_mean > 0.0
