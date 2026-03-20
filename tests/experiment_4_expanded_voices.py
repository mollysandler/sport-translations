# tests/experiment_4_expanded_voices.py
"""
Experiment 4: Expanded Voice Library + Multi-Attribute Scoring

Goals:
- Increase voice pool from 10 to 22+ (10 female, 12 male)
- Replace coarse string pitch labels with numeric pitch_center floats
- Score candidates via multi-attribute function:
    score = pitch_proximity + gender_match + style_bonus
- Never reuse a voice while unused alternatives of the same gender exist

Key metrics: voice distinction score AND many_speakers_unique_voices in
continuity benchmark.
"""

import sys
import os
import numpy as np
import pytest
from typing import Dict, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.test_diarizer_experiments import (
    SCENARIOS,
    make_multi_speaker_audio,
    make_tone,
    run_pitch_benchmark,
    run_voice_matching_benchmark,
    run_continuity_benchmark,
    voice_distinction_score,
    continuity_score,
)
from main import _estimate_pitch_safe, SmartVoiceManager
from utils import gender_from_pitch


# ---------------------------------------------------------------------------
# ExpandedVoiceManager
# ---------------------------------------------------------------------------

class ExpandedVoiceManager:
    """
    Voice manager with 22 voices (10 female + 12 male), numeric pitch
    centres, and multi-attribute scoring.

    Drop-in replacement for SmartVoiceManager in benchmark harness:
    exposes ``_match_best_voice(gender, avg_pitch, pitch_range) -> str``
    and ``used_voice_ids: set``.
    """

    # -- voice catalogue ---------------------------------------------------

    VOICES = {
        # ---------- Female voices (10) ----------
        "voice_f01": {
            "gender": "female",
            "pitch_center": 190.0,
            "style": "warm_calm",
            "age": "young_adult",
            "sports_compatible": True,
        },
        "voice_f02": {
            "gender": "female",
            "pitch_center": 200.0,
            "style": "strong_confident",
            "age": "adult",
            "sports_compatible": True,
        },
        "voice_f03": {
            "gender": "female",
            "pitch_center": 210.0,
            "style": "soft_friendly",
            "age": "young_adult",
            "sports_compatible": False,
        },
        "voice_f04": {
            "gender": "female",
            "pitch_center": 220.0,
            "style": "emotional_expressive",
            "age": "young_adult",
            "sports_compatible": True,
        },
        "voice_f05": {
            "gender": "female",
            "pitch_center": 230.0,
            "style": "crisp_authoritative",
            "age": "adult",
            "sports_compatible": True,
        },
        "voice_f06": {
            "gender": "female",
            "pitch_center": 240.0,
            "style": "energetic_young",
            "age": "young_adult",
            "sports_compatible": True,
        },
        "voice_f07": {
            "gender": "female",
            "pitch_center": 250.0,
            "style": "deep_calm",
            "age": "mature",
            "sports_compatible": False,
        },
        "voice_f08": {
            "gender": "female",
            "pitch_center": 260.0,
            "style": "casual_conversational",
            "age": "young_adult",
            "sports_compatible": False,
        },
        "voice_f09": {
            "gender": "female",
            "pitch_center": 270.0,
            "style": "bright_upbeat",
            "age": "young_adult",
            "sports_compatible": True,
        },
        "voice_f10": {
            "gender": "female",
            "pitch_center": 280.0,
            "style": "warm_smooth",
            "age": "adult",
            "sports_compatible": True,
        },
        # ---------- Male voices (12) ----------
        "voice_m01": {
            "gender": "male",
            "pitch_center": 85.0,
            "style": "deep_authoritative",
            "age": "mature",
            "sports_compatible": True,
        },
        "voice_m02": {
            "gender": "male",
            "pitch_center": 95.0,
            "style": "crisp_strong",
            "age": "adult",
            "sports_compatible": True,
        },
        "voice_m03": {
            "gender": "male",
            "pitch_center": 105.0,
            "style": "warm_calm",
            "age": "adult",
            "sports_compatible": False,
        },
        "voice_m04": {
            "gender": "male",
            "pitch_center": 115.0,
            "style": "well_rounded",
            "age": "adult",
            "sports_compatible": True,
        },
        "voice_m05": {
            "gender": "male",
            "pitch_center": 125.0,
            "style": "strong_confident",
            "age": "adult",
            "sports_compatible": True,
        },
        "voice_m06": {
            "gender": "male",
            "pitch_center": 130.0,
            "style": "energetic_young",
            "age": "young_adult",
            "sports_compatible": True,
        },
        "voice_m07": {
            "gender": "male",
            "pitch_center": 140.0,
            "style": "deep_calm",
            "age": "adult",
            "sports_compatible": False,
        },
        "voice_m08": {
            "gender": "male",
            "pitch_center": 148.0,
            "style": "casual_conversational",
            "age": "young_adult",
            "sports_compatible": True,
        },
        "voice_m09": {
            "gender": "male",
            "pitch_center": 155.0,
            "style": "emotional_expressive",
            "age": "young_adult",
            "sports_compatible": True,
        },
        "voice_m10": {
            "gender": "male",
            "pitch_center": 160.0,
            "style": "bright_upbeat",
            "age": "adult",
            "sports_compatible": True,
        },
        "voice_m11": {
            "gender": "male",
            "pitch_center": 168.0,
            "style": "crisp_authoritative",
            "age": "mature",
            "sports_compatible": True,
        },
        "voice_m12": {
            "gender": "male",
            "pitch_center": 175.0,
            "style": "warm_smooth",
            "age": "young_adult",
            "sports_compatible": False,
        },
    }

    # Sports-friendly styles receive a bonus during scoring
    _SPORTS_STYLES = frozenset({
        "strong_confident",
        "deep_authoritative",
        "energetic_young",
        "crisp_authoritative",
        "crisp_strong",
        "emotional_expressive",
    })

    def __init__(self):
        self.available_voices: Dict[str, dict] = dict(self.VOICES)
        self.used_voice_ids: set = set()

    # -- public API (benchmark-compatible) ---------------------------------

    def _match_best_voice(
        self,
        gender: str,
        avg_pitch: float,
        pitch_range: float,
    ) -> str:
        """
        Find the best-matching voice using multi-attribute scoring.

        Scoring breakdown (per candidate voice):
            pitch_proximity : max(0, 1.0 - |speaker_pitch - voice_center| / 50)
                              Continuous 0-1 score; full marks within 0 Hz,
                              zero at 50+ Hz distance.
            gender_match    : 2.0 if candidate gender == requested gender,
                              0.0 otherwise.
            style_bonus     : 0.5 if the voice has a sports-compatible style.

        Reuse policy:
            1. Prefer unused voices of matching gender.
            2. If all same-gender voices are used, allow reuse of same-gender
               voices (pick best score among them).
            3. If no same-gender voices exist at all, fall back to any unused
               voice, then any voice.
        """
        best_id, best_score = self._pick(gender, avg_pitch, used_ok=False)

        if best_id is None:
            # All same-gender voices are used; allow reuse within gender
            best_id, best_score = self._pick(gender, avg_pitch, used_ok=True)

        if best_id is None:
            # No voices of that gender at all -- fall back to any voice
            best_id, best_score = self._pick(None, avg_pitch, used_ok=False)
            if best_id is None:
                best_id, best_score = self._pick(None, avg_pitch, used_ok=True)

        # Should never be None at this point, but safety net
        if best_id is None:
            best_id = next(iter(self.available_voices))

        self.used_voice_ids.add(best_id)
        return best_id

    # -- internals ---------------------------------------------------------

    def _pick(
        self,
        gender: Optional[str],
        avg_pitch: float,
        used_ok: bool,
    ):
        """
        Score and pick the best voice with optional gender and used filters.
        Returns (voice_id, score) or (None, -1) if nothing qualifies.
        """
        best_id = None
        best_score = -1.0

        for vid, props in self.available_voices.items():
            # Filter: gender
            if gender is not None and props["gender"] != gender:
                continue
            # Filter: already used (when reuse not allowed)
            if not used_ok and vid in self.used_voice_ids:
                continue

            score = self._score(props, gender, avg_pitch)
            if score > best_score:
                best_score = score
                best_id = vid

        return best_id, best_score

    @classmethod
    def _score(cls, voice_props: dict, requested_gender: Optional[str], avg_pitch: float) -> float:
        """Multi-attribute voice score."""
        s = 0.0

        # 1. Pitch proximity (0 - 1)
        pitch_diff = abs(avg_pitch - voice_props["pitch_center"])
        s += max(0.0, 1.0 - pitch_diff / 50.0)

        # 2. Gender match (0 or 2)
        if requested_gender is not None and voice_props["gender"] == requested_gender:
            s += 2.0

        # 3. Sports style bonus (0 or 0.5)
        if voice_props.get("sports_compatible", False):
            s += 0.5

        return s


# ---------------------------------------------------------------------------
# Benchmark adapter functions
# ---------------------------------------------------------------------------

def _expanded_estimate(audio, sr):
    """Wrap _estimate_pitch_safe for the benchmark interface."""
    return _estimate_pitch_safe(audio, sr)


def expanded_voice_match(vm, pitch_hz, pitch_range):
    """
    Voice-matching function following the run_voice_matching_benchmark API.

    ``vm`` is an ExpandedVoiceManager (created by our custom benchmark
    runner below).  We classify gender, then delegate to the manager.
    """
    gender = gender_from_pitch(pitch_hz, pitch_range)
    voice_id = vm._match_best_voice(gender, pitch_hz, pitch_range)
    return voice_id


# Stateful wrapper for the continuity benchmark. Each call to
# ``expanded_continuity`` gets (pitch_hz, session_state) and must return
# a voice_id while persisting state across calls.

_CONTINUITY_THRESHOLD = 35.0  # Hz — slightly wider than baseline 30 Hz


def expanded_continuity(pitch_hz, session_state):
    """
    Continuity function using ExpandedVoiceManager.

    Differences from baseline ``_current_continuity``:
    - A *single* ExpandedVoiceManager is stored inside session_state and
      reused for the entire session, so voice-pool exhaustion is tracked
      across all speakers.
    - A slightly wider threshold (35 Hz) tolerates sports-commentator
      pitch drift.
    - Uses an exponential moving average (EMA) for speaker pitch, so
      gradual drift does not break continuity.
    """
    # Lazily initialise the voice manager inside the session
    if "_evm" not in session_state:
        session_state["_evm"] = ExpandedVoiceManager()

    evm: ExpandedVoiceManager = session_state["_evm"]

    # Try to match an existing speaker via pitch proximity
    best_match = None
    best_dist = float("inf")

    for spk_id, spk_pitch in session_state.get("speaker_pitches", {}).items():
        dist = abs(pitch_hz - spk_pitch)
        if dist < best_dist:
            best_dist = dist
            best_match = spk_id

    if best_match is not None and best_dist <= _CONTINUITY_THRESHOLD:
        # Update the stored pitch with an EMA (alpha=0.3) to track drift
        old = session_state["speaker_pitches"][best_match]
        session_state["speaker_pitches"][best_match] = 0.7 * old + 0.3 * pitch_hz
        return session_state["speaker_voice_ids"][best_match]

    # No close match -- assign a new voice
    gender = gender_from_pitch(pitch_hz)
    voice_id = evm._match_best_voice(gender, pitch_hz, 20.0)
    spk_id = f"SPK_{len(session_state['speaker_voice_ids']):02d}"
    session_state["speaker_voice_ids"][spk_id] = voice_id
    session_state["speaker_pitches"][spk_id] = pitch_hz
    return voice_id


# ---------------------------------------------------------------------------
# Custom benchmark runners (mirror harness but create ExpandedVoiceManager)
# ---------------------------------------------------------------------------

def run_expanded_voice_matching_benchmark(match_fn, estimate_fn, sr=16000):
    """
    Same logic as ``run_voice_matching_benchmark`` but instantiates an
    ``ExpandedVoiceManager`` per scenario instead of ``SmartVoiceManager``.
    """
    results = {}
    for name, scenario in SCENARIOS.items():
        waveform, gt = make_multi_speaker_audio(scenario["speakers"], sr)
        vm = ExpandedVoiceManager()  # <-- expanded pool
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


# ---------------------------------------------------------------------------
# Baseline adapter (for side-by-side comparison output)
# ---------------------------------------------------------------------------

def _baseline_voice_match(vm, pitch_hz, pitch_range):
    gender = gender_from_pitch(pitch_hz, pitch_range)
    return vm._match_best_voice(gender, pitch_hz, pitch_range)


def _baseline_continuity(pitch_hz, session_state):
    """Reproduces the baseline 30 Hz threshold continuity logic."""
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
        vm = SmartVoiceManager()
        gender = gender_from_pitch(pitch_hz)
        voice_id = vm._match_best_voice(gender, pitch_hz, 20.0)
        spk_id = f"SPK_{len(session_state['speaker_voice_ids']):02d}"
        session_state["speaker_voice_ids"][spk_id] = voice_id
        session_state["speaker_pitches"][spk_id] = pitch_hz
        return voice_id


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _print_comparison(title, baseline, expanded, key_fn=None):
    """Print a two-column comparison table."""
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")
    print(f"  {'Scenario':<35} {'Baseline':>10}  {'Expanded':>10}  {'Delta':>8}")
    print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*8}")
    for name in baseline:
        if name == "__overall__":
            continue
        bv = key_fn(baseline[name]) if key_fn else baseline[name]
        ev = key_fn(expanded[name]) if key_fn else expanded[name]
        delta = ev - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<35} {bv:>10.1%}  {ev:>10.1%}  {sign}{delta:>7.1%}")
    bo = baseline["__overall__"]
    eo = expanded["__overall__"]
    d = eo - bo
    sign = "+" if d >= 0 else ""
    print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'OVERALL':<35} {bo:>10.1%}  {eo:>10.1%}  {sign}{d:>7.1%}")
    print(f"{'=' * w}")


# ---------------------------------------------------------------------------
# Pytest test class
# ---------------------------------------------------------------------------

class TestExperiment4:
    """
    Experiment 4: Expanded Voice Library + Multi-Attribute Scoring.

    Each test runs the expanded implementation and prints a comparison
    against the baseline.
    """

    def test_expanded_voice_library_size(self):
        """Verify we have 22 voices: 10 female, 12 male."""
        evm = ExpandedVoiceManager()
        females = [v for v in evm.available_voices.values() if v["gender"] == "female"]
        males = [v for v in evm.available_voices.values() if v["gender"] == "male"]
        print(f"\nVoice pool: {len(females)} female + {len(males)} male = {len(evm.available_voices)} total")
        assert len(females) == 10, f"Expected 10 female voices, got {len(females)}"
        assert len(males) == 12, f"Expected 12 male voices, got {len(males)}"
        assert len(evm.available_voices) == 22

    def test_numeric_pitch_centres(self):
        """Every voice must have a numeric pitch_center field."""
        evm = ExpandedVoiceManager()
        for vid, props in evm.available_voices.items():
            assert "pitch_center" in props, f"{vid} missing pitch_center"
            assert isinstance(props["pitch_center"], float), (
                f"{vid} pitch_center is {type(props['pitch_center'])}, expected float"
            )
        print("\nAll 22 voices have numeric pitch_center fields.")

    def test_multi_attribute_scoring(self):
        """Validate the scoring formula with known inputs."""
        props = {
            "gender": "male",
            "pitch_center": 130.0,
            "style": "energetic_young",
            "sports_compatible": True,
        }
        # Perfect gender match, exact pitch, sports style
        score = ExpandedVoiceManager._score(props, "male", 130.0)
        expected = 1.0 + 2.0 + 0.5  # pitch(1.0) + gender(2.0) + sports(0.5)
        assert abs(score - expected) < 1e-6, f"Expected {expected}, got {score}"

        # 25 Hz away -> pitch_proximity = 0.5
        score2 = ExpandedVoiceManager._score(props, "male", 155.0)
        expected2 = 0.5 + 2.0 + 0.5
        assert abs(score2 - expected2) < 1e-6, f"Expected {expected2}, got {score2}"

        # Wrong gender
        score3 = ExpandedVoiceManager._score(props, "female", 130.0)
        expected3 = 1.0 + 0.0 + 0.5  # no gender bonus
        assert abs(score3 - expected3) < 1e-6, f"Expected {expected3}, got {score3}"

        print("\nMulti-attribute scoring formula verified.")

    def test_no_reuse_while_alternatives_exist(self):
        """Voices should not be reused until all same-gender options are exhausted."""
        evm = ExpandedVoiceManager()
        # Assign 12 male voices -- all should be unique
        assigned = []
        for i in range(12):
            vid = evm._match_best_voice("male", 130.0, 20.0)
            assigned.append(vid)
        assert len(set(assigned)) == 12, (
            f"Expected 12 unique male voices, got {len(set(assigned))}"
        )

        # 13th male voice must reuse (only 12 males available)
        vid13 = evm._match_best_voice("male", 130.0, 20.0)
        assert vid13 in assigned, "13th voice should reuse an existing male voice"

        print("\nNo-reuse policy verified: 12 unique males before any reuse.")

    def test_voice_matching_benchmark(self):
        """Run voice matching benchmark: expanded vs. baseline."""
        np.random.seed(42)

        baseline = run_voice_matching_benchmark(
            _baseline_voice_match, _expanded_estimate,
        )
        expanded = run_expanded_voice_matching_benchmark(
            expanded_voice_match, _expanded_estimate,
        )

        _print_comparison(
            "VOICE DISTINCTION: Baseline vs. Expanded",
            baseline,
            expanded,
            key_fn=lambda r: r["distinction_score"],
        )

        # The expanded manager should not regress
        assert expanded["__overall__"] >= baseline["__overall__"] - 0.01, (
            f"Expanded distinction ({expanded['__overall__']}) regressed vs "
            f"baseline ({baseline['__overall__']})"
        )

    def test_continuity_benchmark(self):
        """Run continuity benchmark: expanded vs. baseline."""
        np.random.seed(42)

        baseline = run_continuity_benchmark(
            _baseline_continuity, _expanded_estimate,
        )

        np.random.seed(42)  # reset so both see identical random drifts
        expanded = run_continuity_benchmark(
            expanded_continuity, _expanded_estimate,
        )

        w = 70
        print(f"\n{'=' * w}")
        print(f"  CONTINUITY: Baseline vs. Expanded")
        print(f"{'=' * w}")
        print(f"  {'Metric':<40} {'Baseline':>8}  {'Expanded':>8}  {'Delta':>8}")
        print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*8}")
        for key in baseline:
            if key == "__overall__":
                continue
            bv = baseline[key]
            ev = expanded[key]
            delta = ev - bv
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<40} {bv:>8.1%}  {ev:>8.1%}  {sign}{delta:>7.1%}")

        bo = baseline["__overall__"]
        eo = expanded["__overall__"]
        d = eo - bo
        sign = "+" if d >= 0 else ""
        print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*8}")
        print(f"  {'OVERALL':<40} {bo:>8.1%}  {eo:>8.1%}  {sign}{d:>7.1%}")
        print(f"{'=' * w}")

        # Key metric: many_speakers_unique_voices should improve
        print(f"\n  KEY METRIC  many_speakers_unique_voices:")
        print(f"    Baseline : {baseline['many_speakers_unique_voices']:.0%}")
        print(f"    Expanded : {expanded['many_speakers_unique_voices']:.0%}")
        assert expanded["many_speakers_unique_voices"] > baseline["many_speakers_unique_voices"], (
            f"many_speakers_unique_voices did not improve: "
            f"baseline={baseline['many_speakers_unique_voices']}, "
            f"expanded={expanded['many_speakers_unique_voices']}"
        )

    def test_continuity_many_speakers_regression(self):
        """
        Focused regression: 8 speakers at distinct pitches should each get a
        unique voice with the expanded pool.
        """
        np.random.seed(42)
        session = {"speaker_voice_ids": {}, "speaker_pitches": {}}
        pitches = [100, 120, 140, 160, 180, 200, 220, 240]
        assigned = []
        for p in pitches:
            audio = make_tone(float(p), 2.0, 16000)
            est, _ = _expanded_estimate(audio, 16000)
            vid = expanded_continuity(est, session)
            assigned.append(vid)

        unique = len(set(assigned))
        pct = unique / len(pitches)
        print(f"\n  Many-speaker regression: {unique}/{len(pitches)} unique voices ({pct:.0%})")
        for i, (p, v) in enumerate(zip(pitches, assigned)):
            print(f"    {p} Hz -> {v}")
        assert unique == len(pitches), (
            f"Expected {len(pitches)} unique voices for {len(pitches)} distinct "
            f"speakers, got {unique}"
        )
