# tests/test_diarizer_experiments.py
"""
Benchmarking harness for diarizer & voice assignment experiments.

Tests the *logic layers* we control (pitch estimation, gender classification,
voice matching, live continuity) using synthetic audio with known ground truth.
Does NOT require pyannote/Whisper models — runs in seconds.

Each experiment module can import and run these benchmarks to measure impact.
"""
import sys, os, types
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Synthetic audio generation helpers
# ---------------------------------------------------------------------------

def make_tone(freq_hz: float, duration_sec: float, sr: int = 16000,
              amplitude: float = 0.3, noise_level: float = 0.02,
              harmonics: bool = True,
              harmonic_dominant: bool = False) -> np.ndarray:
    """
    Generate a speech-like harmonic signal at a given fundamental frequency.
    With harmonics=True, adds overtones simulating vocal fold harmonics.
    With harmonic_dominant=True, the 2nd harmonic is STRONGER than the
    fundamental — this triggers octave errors in naive autocorrelation.
    """
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if harmonics:
        if harmonic_dominant:
            # 2nd harmonic stronger than fundamental — classic octave error trigger
            for h, gain in [(2, 1.3), (3, 0.7), (4, 0.3), (5, 0.15)]:
                harm_freq = freq_hz * h
                if harm_freq < sr / 2:
                    tone += amplitude * gain * np.sin(2 * np.pi * harm_freq * t)
        else:
            for h, gain in [(2, 0.6), (3, 0.35), (4, 0.2), (5, 0.1)]:
                harm_freq = freq_hz * h
                if harm_freq < sr / 2:
                    tone += amplitude * gain * np.sin(2 * np.pi * harm_freq * t)
    if noise_level > 0:
        tone += noise_level * np.random.randn(len(tone)).astype(np.float32)
    return tone


def make_multi_speaker_audio(
    speakers: List[Dict],
    sr: int = 16000,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Build a synthetic waveform with known speaker segments.

    speakers: list of dicts with keys:
        - pitch_hz: fundamental frequency
        - segments: list of (start_sec, end_sec) tuples
        - label: ground-truth speaker name
        - amplitude: optional (default 0.3)
        - noise: optional noise level (default 0.02)

    Returns (waveform_1d, ground_truth_segments) where ground_truth_segments
    have keys: label, start_sec, end_sec, pitch_hz
    """
    # Find total duration
    max_end = max(end for spk in speakers for (_, end) in spk["segments"])
    total_samples = int(max_end * sr)
    waveform = np.zeros(total_samples, dtype=np.float32)
    ground_truth = []

    for spk in speakers:
        amp = spk.get("amplitude", 0.3)
        noise = spk.get("noise", 0.02)
        harm_dom = spk.get("harmonic_dominant", False)
        for start, end in spk["segments"]:
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            dur = end - start
            tone = make_tone(spk["pitch_hz"], dur, sr, amp, noise,
                             harmonic_dominant=harm_dom)
            # Mix into waveform (additive for overlaps)
            waveform[s_idx:s_idx + len(tone)] += tone[:e_idx - s_idx]
            ground_truth.append({
                "label": spk["label"],
                "start_sec": start,
                "end_sec": end,
                "pitch_hz": spk["pitch_hz"],
            })

    # Clip to prevent overflow
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform, ground_truth


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def pitch_accuracy(estimated_hz: float, true_hz: float) -> float:
    """Return accuracy as 1 - relative_error. 1.0 = perfect."""
    if true_hz == 0:
        return 0.0
    return max(0.0, 1.0 - abs(estimated_hz - true_hz) / true_hz)


def gender_accuracy(predicted: str, true_pitch_hz: float) -> float:
    """Score gender prediction. Ground truth: <155 = male, >185 = female, else ambiguous."""
    if true_pitch_hz < 155:
        return 1.0 if predicted == "male" else 0.0
    elif true_pitch_hz > 185:
        return 1.0 if predicted == "female" else 0.0
    else:
        # Ambiguous zone — any answer is acceptable
        return 1.0 if predicted in ("male", "female", "unknown") else 0.0


def voice_distinction_score(
    assignments: Dict[str, str],
    ground_truth_labels: List[str],
) -> float:
    """
    Score how well voice assignment distinguishes different speakers.
    1.0 = each ground-truth speaker got a unique voice.
    0.0 = all speakers got the same voice.
    """
    if len(ground_truth_labels) <= 1:
        return 1.0
    voices = [assignments.get(label, "none") for label in ground_truth_labels]
    unique_voices = len(set(voices))
    return unique_voices / len(ground_truth_labels)


def continuity_score(
    chunk_assignments: List[Dict[str, str]],
    expected_speaker: str,
) -> float:
    """
    Score cross-chunk continuity for a single speaker.
    1.0 = same voice in every chunk. 0.0 = different voice each time.
    """
    voices = []
    for assignments in chunk_assignments:
        if expected_speaker in assignments:
            voices.append(assignments[expected_speaker])
    if len(voices) <= 1:
        return 1.0
    most_common = max(set(voices), key=voices.count)
    return voices.count(most_common) / len(voices)


# ---------------------------------------------------------------------------
# Standard test scenarios (shared across all experiments)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "clear_male_female": {
        "description": "Two clearly different speakers (male 120Hz, female 220Hz)",
        "speakers": [
            {"label": "male_commentator", "pitch_hz": 120.0,
             "segments": [(0, 3), (6, 9)]},
            {"label": "female_commentator", "pitch_hz": 220.0,
             "segments": [(3, 6), (9, 12)]},
        ],
    },
    "two_similar_males": {
        "description": "Two male speakers with similar pitch (150Hz vs 165Hz)",
        "speakers": [
            {"label": "host", "pitch_hz": 150.0,
             "segments": [(0, 3), (6, 9)]},
            {"label": "analyst", "pitch_hz": 165.0,
             "segments": [(3, 6), (9, 12)]},
        ],
    },
    "three_speakers": {
        "description": "Three speakers: low male (110Hz), mid male (160Hz), female (210Hz)",
        "speakers": [
            {"label": "pbp", "pitch_hz": 110.0,
             "segments": [(0, 3), (9, 12)]},
            {"label": "color", "pitch_hz": 160.0,
             "segments": [(3, 6)]},
            {"label": "sideline", "pitch_hz": 210.0,
             "segments": [(6, 9)]},
        ],
    },
    "noisy_environment": {
        "description": "Two speakers in high-noise conditions",
        "speakers": [
            {"label": "speaker_a", "pitch_hz": 130.0, "noise": 0.1,
             "segments": [(0, 4), (8, 12)]},
            {"label": "speaker_b", "pitch_hz": 200.0, "noise": 0.1,
             "segments": [(4, 8)]},
        ],
    },
    "pitch_drift": {
        "description": "Speaker whose pitch drifts significantly across segments",
        "speakers": [
            # Simulated as two separate entries with different pitches for same speaker
            {"label": "drifter", "pitch_hz": 140.0,
             "segments": [(0, 3)]},
            {"label": "drifter_excited", "pitch_hz": 175.0,  # same speaker, excited
             "segments": [(3, 6)]},
            {"label": "stable", "pitch_hz": 220.0,
             "segments": [(6, 9)]},
        ],
    },
    "short_utterances": {
        "description": "Very short speaker turns (< 1 second)",
        "speakers": [
            {"label": "quick_a", "pitch_hz": 130.0,
             "segments": [(0, 0.8), (1.6, 2.4), (3.2, 4.0)]},
            {"label": "quick_b", "pitch_hz": 190.0,
             "segments": [(0.8, 1.6), (2.4, 3.2)]},
        ],
    },
    "boundary_pitch": {
        "description": "Speaker right at gender classification boundary (170Hz)",
        "speakers": [
            {"label": "ambiguous", "pitch_hz": 170.0,
             "segments": [(0, 4)]},
            {"label": "clear_male", "pitch_hz": 110.0,
             "segments": [(4, 8)]},
        ],
    },
    "extreme_pitches": {
        "description": "Very low (80Hz) and very high (280Hz) speakers",
        "speakers": [
            {"label": "bass", "pitch_hz": 80.0,
             "segments": [(0, 4)]},
            {"label": "soprano", "pitch_hz": 280.0,
             "segments": [(4, 8)]},
        ],
    },
    "overlapping_speakers": {
        "description": "Two speakers talking simultaneously (mixed signal)",
        "speakers": [
            {"label": "speaker_1", "pitch_hz": 125.0,
             "segments": [(0, 5)]},
            {"label": "speaker_2", "pitch_hz": 195.0,
             "segments": [(2, 7)]},  # overlaps 2-5s
        ],
    },
    "octave_ambiguous": {
        "description": "Speaker at 150Hz with 2nd harmonic dominant — triggers octave errors",
        "speakers": [
            {"label": "octave_test", "pitch_hz": 150.0, "amplitude": 0.15,
             "segments": [(0, 5)], "harmonic_dominant": True},
        ],
    },
    "octave_error_low": {
        "description": "Low voice (90Hz) with dominant 2nd harmonic — may report 180Hz",
        "speakers": [
            {"label": "bass_harmonic", "pitch_hz": 90.0,
             "segments": [(0, 4)], "harmonic_dominant": True},
        ],
    },
    "crowd_plus_speaker": {
        "description": "Speaker with heavy crowd/stadium noise",
        "speakers": [
            {"label": "commentator", "pitch_hz": 145.0, "noise": 0.2, "amplitude": 0.2,
             "segments": [(0, 5)]},
        ],
    },
    "very_close_pitches": {
        "description": "Three male speakers within 20Hz of each other (140, 150, 160)",
        "speakers": [
            {"label": "male_a", "pitch_hz": 140.0,
             "segments": [(0, 3)]},
            {"label": "male_b", "pitch_hz": 150.0,
             "segments": [(3, 6)]},
            {"label": "male_c", "pitch_hz": 160.0,
             "segments": [(6, 9)]},
        ],
    },
    "whisper_then_shout": {
        "description": "Same speaker: quiet whisper then loud shout",
        "speakers": [
            {"label": "quiet", "pitch_hz": 160.0, "amplitude": 0.05,
             "segments": [(0, 3)]},
            {"label": "loud", "pitch_hz": 160.0, "amplitude": 0.5,
             "segments": [(3, 6)]},
        ],
    },
}


def run_pitch_benchmark(estimate_fn, sr: int = 16000) -> Dict:
    """
    Run pitch estimation benchmark across all scenarios.
    estimate_fn(audio_1d: np.ndarray, sr: int) -> (median_pitch, pitch_range)
    Returns dict of scenario_name -> {per_speaker_results, avg_accuracy}
    """
    results = {}
    for name, scenario in SCENARIOS.items():
        waveform, gt = make_multi_speaker_audio(scenario["speakers"], sr)
        speaker_results = []
        for spk in scenario["speakers"]:
            # Extract just this speaker's audio
            combined = np.concatenate([
                waveform[int(s * sr):int(e * sr)]
                for s, e in spk["segments"]
            ])
            try:
                est_pitch, est_range = estimate_fn(combined, sr)
            except Exception as exc:
                est_pitch, est_range = 150.0, 20.0  # default on failure
            acc = pitch_accuracy(est_pitch, spk["pitch_hz"])
            speaker_results.append({
                "label": spk["label"],
                "true_pitch": spk["pitch_hz"],
                "estimated_pitch": round(est_pitch, 1),
                "accuracy": round(acc, 3),
            })
        avg_acc = np.mean([r["accuracy"] for r in speaker_results])
        results[name] = {
            "speakers": speaker_results,
            "avg_accuracy": round(float(avg_acc), 3),
        }
    overall = np.mean([r["avg_accuracy"] for r in results.values()])
    results["__overall__"] = round(float(overall), 3)
    return results


def run_gender_benchmark(gender_fn, estimate_fn, sr: int = 16000) -> Dict:
    """
    Run gender classification benchmark.
    gender_fn(pitch_hz, pitch_range) -> str
    estimate_fn(audio_1d, sr) -> (pitch, range)
    """
    results = {}
    for name, scenario in SCENARIOS.items():
        waveform, gt = make_multi_speaker_audio(scenario["speakers"], sr)
        speaker_results = []
        for spk in scenario["speakers"]:
            combined = np.concatenate([
                waveform[int(s * sr):int(e * sr)]
                for s, e in spk["segments"]
            ])
            try:
                est_pitch, est_range = estimate_fn(combined, sr)
            except Exception:
                est_pitch, est_range = 150.0, 20.0
            predicted = gender_fn(est_pitch, est_range)
            acc = gender_accuracy(predicted, spk["pitch_hz"])
            speaker_results.append({
                "label": spk["label"],
                "true_pitch": spk["pitch_hz"],
                "predicted_gender": predicted,
                "correct": acc == 1.0,
            })
        avg_acc = np.mean([r["correct"] for r in speaker_results])
        results[name] = {
            "speakers": speaker_results,
            "avg_accuracy": round(float(avg_acc), 3),
        }
    overall = np.mean([r["avg_accuracy"] for r in results.values()])
    results["__overall__"] = round(float(overall), 3)
    return results


def run_voice_matching_benchmark(match_fn, estimate_fn, sr: int = 16000) -> Dict:
    """
    Run voice matching benchmark.
    match_fn(vm: SmartVoiceManager, pitch_hz, pitch_range) -> str (voice_id)
        NOTE: receives a SHARED VoiceManager so used_voice_ids accumulates
        across speakers within a scenario (realistic).
    estimate_fn(audio_1d, sr) -> (pitch, range)
    Returns distinction scores per scenario.
    """
    results = {}
    for name, scenario in SCENARIOS.items():
        waveform, gt = make_multi_speaker_audio(scenario["speakers"], sr)
        vm = SmartVoiceManager()  # shared across speakers in this scenario
        assignments = {}
        used_voices = set()
        for spk in scenario["speakers"]:
            combined = np.concatenate([
                waveform[int(s * sr):int(e * sr)]
                for s, e in spk["segments"]
            ])
            try:
                est_pitch, est_range = estimate_fn(combined, sr)
            except Exception:
                est_pitch, est_range = 150.0, 20.0
            voice_id = match_fn(vm, est_pitch, est_range)
            assignments[spk["label"]] = voice_id
            used_voices.add(voice_id)

        labels = [spk["label"] for spk in scenario["speakers"]]
        unique_speakers = set()
        for l in labels:
            base = l.replace("_excited", "")
            unique_speakers.add(base)

        distinction = voice_distinction_score(assignments, labels)
        results[name] = {
            "assignments": assignments,
            "distinction_score": round(distinction, 3),
            "unique_voices": len(used_voices),
            "num_speakers": len(unique_speakers),
        }
    overall = np.mean([r["distinction_score"] for r in results.values()])
    results["__overall__"] = round(float(overall), 3)
    return results


def run_continuity_benchmark(
    continuity_fn,
    estimate_fn,
    sr: int = 16000,
    num_chunks: int = 5,
) -> Dict:
    """
    Simulate live streaming: same speaker appears across N chunks with
    slight pitch variation. Test if continuity_fn maintains voice assignment.

    continuity_fn(pitch_hz, session_state) -> voice_id
        Must accept a session_state dict and maintain state across calls.
    estimate_fn(audio_1d, sr) -> (pitch, range)
    """
    results = {}

    # Test 1: Stable speaker across chunks
    session = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments = []
    base_pitch = 140.0
    for i in range(num_chunks):
        # Simulate natural pitch drift: ±15 Hz
        drift = np.random.uniform(-15, 15)
        pitch = base_pitch + drift
        audio = make_tone(pitch, 2.0, sr)
        try:
            est_pitch, _ = estimate_fn(audio, sr)
        except Exception:
            est_pitch = pitch
        voice_id = continuity_fn(est_pitch, session)
        chunk_assignments.append({"stable_speaker": voice_id})

    stable_score = continuity_score(chunk_assignments, "stable_speaker")
    results["stable_speaker_continuity"] = round(stable_score, 3)

    # Test 2: Two alternating speakers
    session2 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_a = []
    chunk_assignments_b = []
    for i in range(num_chunks):
        # Speaker A: ~130 Hz
        pitch_a = 130.0 + np.random.uniform(-10, 10)
        audio_a = make_tone(pitch_a, 2.0, sr)
        est_a, _ = estimate_fn(audio_a, sr)
        voice_a = continuity_fn(est_a, session2)
        chunk_assignments_a.append({"speaker_a": voice_a})

        # Speaker B: ~200 Hz
        pitch_b = 200.0 + np.random.uniform(-10, 10)
        audio_b = make_tone(pitch_b, 2.0, sr)
        est_b, _ = estimate_fn(audio_b, sr)
        voice_b = continuity_fn(est_b, session2)
        chunk_assignments_b.append({"speaker_b": voice_b})

    results["two_speaker_a_continuity"] = round(
        continuity_score(chunk_assignments_a, "speaker_a"), 3)
    results["two_speaker_b_continuity"] = round(
        continuity_score(chunk_assignments_b, "speaker_b"), 3)

    # Test 3: Similar-pitch speakers (hard case)
    session3 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_x = []
    chunk_assignments_y = []
    for i in range(num_chunks):
        pitch_x = 155.0 + np.random.uniform(-8, 8)
        audio_x = make_tone(pitch_x, 2.0, sr)
        est_x, _ = estimate_fn(audio_x, sr)
        voice_x = continuity_fn(est_x, session3)
        chunk_assignments_x.append({"similar_x": voice_x})

        pitch_y = 165.0 + np.random.uniform(-8, 8)
        audio_y = make_tone(pitch_y, 2.0, sr)
        est_y, _ = estimate_fn(audio_y, sr)
        voice_y = continuity_fn(est_y, session3)
        chunk_assignments_y.append({"similar_y": voice_y})

    results["similar_pitch_x_continuity"] = round(
        continuity_score(chunk_assignments_x, "similar_x"), 3)
    results["similar_pitch_y_continuity"] = round(
        continuity_score(chunk_assignments_y, "similar_y"), 3)

    # Test 4: Crossing pitches — speaker A drifts UP, speaker B drifts DOWN
    # They cross in the middle, which should confuse pitch-only matching
    session4 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_cross_a = []
    chunk_assignments_cross_b = []
    for i in range(num_chunks):
        # Speaker A: starts 130, drifts to 180 (excited commentator)
        pitch_a = 130.0 + (50.0 * i / (num_chunks - 1)) + np.random.uniform(-5, 5)
        audio_a = make_tone(pitch_a, 2.0, sr)
        est_a, _ = estimate_fn(audio_a, sr)
        voice_a = continuity_fn(est_a, session4)
        chunk_assignments_cross_a.append({"cross_a": voice_a})

        # Speaker B: starts 180, drifts to 130 (calming down)
        pitch_b = 180.0 - (50.0 * i / (num_chunks - 1)) + np.random.uniform(-5, 5)
        audio_b = make_tone(pitch_b, 2.0, sr)
        est_b, _ = estimate_fn(audio_b, sr)
        voice_b = continuity_fn(est_b, session4)
        chunk_assignments_cross_b.append({"cross_b": voice_b})

    results["crossing_a_continuity"] = round(
        continuity_score(chunk_assignments_cross_a, "cross_a"), 3)
    results["crossing_b_continuity"] = round(
        continuity_score(chunk_assignments_cross_b, "cross_b"), 3)

    # Test 5: Large pitch drift — same speaker varies ±30 Hz (shouting at a game)
    session5 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    chunk_assignments_drift = []
    for i in range(num_chunks):
        drift = np.random.uniform(-30, 30)
        pitch = 155.0 + drift
        audio = make_tone(pitch, 2.0, sr)
        est_pitch, _ = estimate_fn(audio, sr)
        voice_id = continuity_fn(est_pitch, session5)
        chunk_assignments_drift.append({"drifty": voice_id})

    results["large_drift_continuity"] = round(
        continuity_score(chunk_assignments_drift, "drifty"), 3)

    # Test 6: Many speakers (voice pool exhaustion in continuity)
    session6 = {"speaker_voice_ids": {}, "speaker_pitches": {}}
    pitches_for_many = [100, 120, 140, 160, 180, 200, 220, 240]
    many_assignments = []
    for p in pitches_for_many:
        audio = make_tone(float(p), 2.0, sr)
        est, _ = estimate_fn(audio, sr)
        vid = continuity_fn(est, session6)
        many_assignments.append(vid)
    # Check: how many unique voices assigned?
    unique_many = len(set(many_assignments))
    results["many_speakers_unique_voices"] = round(unique_many / len(pitches_for_many), 3)

    # Overall
    scores = [v for k, v in results.items() if isinstance(v, float)]
    results["__overall__"] = round(float(np.mean(scores)), 3)
    return results


# ---------------------------------------------------------------------------
# Baseline test: run current implementation through benchmarks
# ---------------------------------------------------------------------------

from main import _estimate_pitch_safe, SmartVoiceManager
from utils import gender_from_pitch


def _current_estimate(audio, sr):
    """Wrap _estimate_pitch_safe to match benchmark interface."""
    return _estimate_pitch_safe(audio, sr)


def _current_gender(pitch_hz, pitch_range=None):
    return gender_from_pitch(pitch_hz, pitch_range)


def _current_voice_match(vm, pitch_hz, pitch_range):
    """Use shared VoiceManager (realistic — voices get marked used)."""
    gender = gender_from_pitch(pitch_hz, pitch_range)
    voice_id = vm._match_best_voice(gender, pitch_hz, pitch_range)
    return voice_id


def _current_continuity(pitch_hz, session_state):
    """Simulate current 30Hz threshold continuity logic."""
    THRESHOLD = 30.0
    best_match = None
    best_dist = float("inf")

    for spk_id, spk_pitch in session_state.get("speaker_pitches", {}).items():
        dist = abs(pitch_hz - spk_pitch)
        if dist < best_dist:
            best_dist = dist
            best_match = spk_id

    if best_match is not None and best_dist <= THRESHOLD:
        return session_state["speaker_voice_ids"][best_match]
    else:
        # Assign new voice
        vm = SmartVoiceManager()
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        spk_id = f"SPK_{len(session_state['speaker_voice_ids']):02d}"
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        return voice_id


class TestBaseline:
    """Run benchmarks against current implementation to establish baseline scores."""

    def test_pitch_estimation_baseline(self):
        results = run_pitch_benchmark(_current_estimate)
        overall = results["__overall__"]
        print(f"\n{'='*60}")
        print(f"PITCH ESTIMATION BASELINE: {overall:.1%}")
        print(f"{'='*60}")
        for name, data in results.items():
            if name == "__overall__":
                continue
            print(f"  {name}: {data['avg_accuracy']:.1%}")
            for spk in data["speakers"]:
                print(f"    {spk['label']}: true={spk['true_pitch']}Hz "
                      f"est={spk['estimated_pitch']}Hz acc={spk['accuracy']:.1%}")
        # Baseline should at least work — no hard assertion on quality
        assert overall > 0.0, "Pitch estimation completely failed"

    def test_gender_classification_baseline(self):
        results = run_gender_benchmark(_current_gender, _current_estimate)
        overall = results["__overall__"]
        print(f"\n{'='*60}")
        print(f"GENDER CLASSIFICATION BASELINE: {overall:.1%}")
        print(f"{'='*60}")
        for name, data in results.items():
            if name == "__overall__":
                continue
            print(f"  {name}: {data['avg_accuracy']:.1%}")
        assert overall > 0.0

    def test_voice_matching_baseline(self):
        results = run_voice_matching_benchmark(
            _current_voice_match, _current_estimate)
        overall = results["__overall__"]
        print(f"\n{'='*60}")
        print(f"VOICE DISTINCTION BASELINE: {overall:.1%}")
        print(f"{'='*60}")
        for name, data in results.items():
            if name == "__overall__":
                continue
            print(f"  {name}: distinction={data['distinction_score']:.1%} "
                  f"voices={data['unique_voices']}/{data['num_speakers']}spk")
        assert overall > 0.0

    def test_continuity_baseline(self):
        np.random.seed(42)
        results = run_continuity_benchmark(
            _current_continuity, _current_estimate)
        overall = results["__overall__"]
        print(f"\n{'='*60}")
        print(f"CONTINUITY BASELINE: {overall:.1%}")
        print(f"{'='*60}")
        for name, val in results.items():
            if name == "__overall__":
                continue
            print(f"  {name}: {val:.1%}")
        assert overall > 0.0
