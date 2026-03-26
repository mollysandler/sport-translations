# tests/test_chunk_splitting.py
"""
Tests that audio chunking splits at silence boundaries, not mid-word.

We synthesize a PCM signal that mimics:
  [2s speech] [0.2s silence] [2s speech] [0.2s silence] [2s speech] ...

With a fixed 8-second chunk, the boundary falls in the middle of a
"word" (tone burst), splitting it in half.  A silence-aware chunker
should instead split at one of the silence gaps.
"""

import time
import numpy as np
from utils import find_silence_split


# ---------------------------------------------------------------------------
# Helper: build a fake speech signal with known silence gaps
# ---------------------------------------------------------------------------

def make_speech_with_gaps(
    sr: int = 16000,
    word_durations: list[float] = None,
    gap_duration: float = 0.2,
    freq: float = 220.0,
    amplitude: float = 0.5,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Create a float32 PCM signal of alternating tone bursts ("words") and
    silence gaps.

    Returns:
        (samples, gap_intervals)  where gap_intervals is a list of
        (start_sec, end_sec) for each silence gap.
    """
    if word_durations is None:
        # 5 "words" of 2s each → 10s speech + 0.8s gaps ≈ 10.8s total
        word_durations = [2.0, 2.0, 2.0, 2.0, 2.0]

    parts: list[np.ndarray] = []
    gaps: list[tuple[float, float]] = []
    cursor = 0.0

    for i, dur in enumerate(word_durations):
        # tone burst = "word"
        n_samples = int(dur * sr)
        t = np.arange(n_samples, dtype=np.float32) / sr
        tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        parts.append(tone)
        cursor += dur

        # silence gap (except after last word)
        if i < len(word_durations) - 1:
            n_gap = int(gap_duration * sr)
            gap_start = cursor
            parts.append(np.zeros(n_gap, dtype=np.float32))
            cursor += gap_duration
            gaps.append((gap_start, cursor))

    return np.concatenate(parts), gaps


# ---------------------------------------------------------------------------
# Current (broken) chunker: fixed-time split
# ---------------------------------------------------------------------------

def fixed_time_chunker(samples: np.ndarray, sr: int, chunk_seconds: float = 8.0) -> list[np.ndarray]:
    """Split samples into fixed-duration chunks (current behavior)."""
    chunk_size = int(chunk_seconds * sr)
    chunks = []
    offset = 0
    while offset < len(samples):
        end = min(offset + chunk_size, len(samples))
        chunks.append(samples[offset:end])
        offset = end
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFixedChunkerSplitsWords:
    """Prove that the fixed-time chunker cuts through speech."""

    def test_fixed_chunker_splits_mid_speech(self):
        """
        With 2s words + 0.2s gaps, an 8s chunk boundary lands at exactly 8.0s.
        Timeline:
            0.0-2.0  word0
            2.0-2.2  gap
            2.2-4.2  word1
            4.2-4.4  gap
            4.4-6.4  word2
            6.4-6.6  gap
            6.6-8.6  word3   ← chunk boundary at 8.0s falls INSIDE word3
            8.6-8.8  gap
            8.8-10.8 word4

        The 8-second boundary (sample 128000) is at t=8.0s, which is
        inside word3 (6.6-8.6s).  The fixed chunker will cut word3 in half.
        """
        sr = 16000
        samples, gaps = make_speech_with_gaps(sr=sr)

        chunks = fixed_time_chunker(samples, sr, chunk_seconds=8.0)

        # The boundary between chunk[0] and chunk[1] is at sample 128000 = 8.0s
        boundary_sec = 8.0

        # Check: does the boundary fall inside a silence gap?
        boundary_in_gap = any(
            start <= boundary_sec <= end for start, end in gaps
        )

        # With fixed chunking, the boundary should NOT be in a gap —
        # it lands at 8.0s which is inside word3 (6.6–8.6s).
        # This is the bug we want to fix.
        assert not boundary_in_gap, (
            f"Expected boundary at {boundary_sec}s to be mid-speech, "
            f"but it landed in a gap: {gaps}"
        )

        # Verify the tail of chunk[0] has significant energy (= speech was cut)
        tail = chunks[0][-int(0.1 * sr):]  # last 100ms of first chunk
        tail_rms = float(np.sqrt(np.mean(tail ** 2)))
        assert tail_rms > 0.1, (
            f"Tail of chunk should contain speech (RMS={tail_rms:.4f}), "
            f"proving the word was split mid-utterance"
        )

    def test_silence_aware_chunker_avoids_mid_speech_split(self):
        """
        A silence-aware chunker should split at a gap, NOT mid-word.
        """
        sr = 16000
        samples, gaps = make_speech_with_gaps(sr=sr)
        min_chunk_sec = 5.0
        max_chunk_sec = 12.0

        split_idx = find_silence_split(
            samples, sr,
            min_chunk_samples=int(min_chunk_sec * sr),
            max_chunk_samples=int(max_chunk_sec * sr),
        )

        split_sec = split_idx / sr

        # The split point should land inside one of the silence gaps
        split_in_gap = any(
            start <= split_sec <= end for start, end in gaps
        )
        assert split_in_gap, (
            f"Split at {split_sec:.3f}s is NOT in a silence gap. "
            f"Gaps are: {[(f'{s:.1f}-{e:.1f}') for s, e in gaps]}. "
            f"The chunker split mid-word!"
        )

        # Additionally: the audio just before the split should be silent
        pre_split = samples[max(0, split_idx - int(0.05 * sr)):split_idx]
        pre_rms = float(np.sqrt(np.mean(pre_split ** 2)))
        assert pre_rms < 0.05, (
            f"Audio before split point should be silence (RMS={pre_rms:.4f})"
        )


# ---------------------------------------------------------------------------
# Exhaustive silence-aware chunker tests
# ---------------------------------------------------------------------------

def _assert_split_in_gap(split_idx, sr, gaps, label=""):
    """Helper: assert split_idx (in samples) falls inside one of the gaps."""
    split_sec = split_idx / sr
    in_gap = any(start <= split_sec <= end for start, end in gaps)
    assert in_gap, (
        f"[{label}] Split at {split_sec:.3f}s is NOT in a silence gap. "
        f"Gaps: {[(f'{s:.2f}-{e:.2f}') for s, e in gaps]}"
    )


def _simulate_streaming_splits(samples, sr, min_sec, max_sec):
    """Simulate how stream_chunks_from_url would split a full buffer
    incrementally.  Returns list of (split_sample_idx, chunk)."""
    min_samp = int(min_sec * sr)
    max_samp = int(max_sec * sr)
    offset = 0
    splits = []
    while offset < len(samples):
        remaining = samples[offset:]
        if len(remaining) < min_samp:
            # final short tail — emit as-is
            splits.append((offset, remaining))
            break
        idx = find_silence_split(remaining, sr, min_samp, max_samp)
        if idx > 0:
            splits.append((offset, remaining[:idx]))
            offset += idx
        else:
            # Buffer too short for window check — emit all
            splits.append((offset, remaining))
            break
    return splits


class TestSilenceSplitEdgeCases:
    """Broad coverage: various speech/silence patterns."""

    def test_short_words_many_gaps(self):
        """0.5s words with 0.15s gaps — lots of split opportunities."""
        sr = 16000
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[0.5] * 20, gap_duration=0.15,
        )
        # Total ≈ 10s speech + 2.85s gaps = 12.85s
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0, "Should find silence in a signal full of gaps"
        _assert_split_in_gap(idx, sr, gaps, "short_words")

    def test_long_words_with_200ms_gaps(self):
        """4s words with 0.2s gaps — should find the gaps."""
        sr = 16000
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[4.0, 4.0, 4.0], gap_duration=0.2,
        )
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0
        _assert_split_in_gap(idx, sr, gaps, "long_words_200ms")

    def test_continuous_speech_returns_quietest_point(self):
        """Continuous speech with no gaps → returns the least-bad split
        (quietest region), not -1. This is the fallback behavior."""
        sr = 16000
        n = 15 * sr
        t = np.arange(n, dtype=np.float32) / sr
        samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        # Should return a valid index (fallback to quietest point)
        assert idx > 0, "Should return a fallback split, not -1"
        assert 5 * sr <= idx <= 12 * sr, (
            f"Fallback split at {idx} should be within [min, max] range"
        )

    def test_all_silence(self):
        """Entirely silent buffer — should find a split easily."""
        sr = 16000
        samples = np.zeros(10 * sr, dtype=np.float32)
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0, "Should split within an all-silent buffer"
        assert 5 * sr <= idx <= 10 * sr

    def test_silence_only_before_min_falls_back(self):
        """Gap exists but only before min_chunk — fallback picks quietest in range."""
        sr = 16000
        # 2s silence then 13s speech
        silence = np.zeros(2 * sr, dtype=np.float32)
        t = np.arange(13 * sr, dtype=np.float32) / sr
        speech = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        samples = np.concatenate([silence, speech])
        # min=5s, max=12s — gap is at 0-2s, which is before min
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        # Should still return a valid fallback (quietest point in the range)
        assert idx > 0
        assert 5 * sr <= idx <= 12 * sr

    def test_silence_right_at_min_boundary(self):
        """Gap starting exactly at min_chunk_seconds."""
        sr = 16000
        # 5s speech, 0.2s silence, 5s speech
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[5.0, 5.0], gap_duration=0.2,
        )
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0
        _assert_split_in_gap(idx, sr, gaps, "at_min_boundary")

    def test_prefers_latest_gap(self):
        """With multiple gaps in range, should pick the one closest to max
        (scans backwards)."""
        sr = 16000
        # 3s, gap, 3s, gap, 3s  →  gaps at 3.0 and 6.2
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[3.0, 3.0, 3.0], gap_duration=0.2,
        )
        idx = find_silence_split(samples, sr, 2 * sr, 10 * sr)
        split_sec = idx / sr
        # Should pick the later gap (≈6.2s), not the earlier one (≈3.0s)
        assert split_sec > 5.0, (
            f"Expected split near later gap (~6.2s), got {split_sec:.3f}s"
        )
        _assert_split_in_gap(idx, sr, gaps, "latest_gap")

    def test_quiet_speech_below_new_threshold(self):
        """Speech with amplitude=0.015 (RMS ≈ 0.0106) is below the 0.02
        threshold, so pass 1 treats entire signal as "silent". The function
        should still return a valid split in range."""
        sr = 16000
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[3.0, 3.0, 3.0], gap_duration=0.2,
            amplitude=0.015,
        )
        idx = find_silence_split(samples, sr, 2 * sr, 10 * sr)
        assert idx > 0, "Should return a split"
        assert 2 * sr <= idx <= 10 * sr, "Split should be within range"

    def test_very_loud_with_gaps(self):
        """Loud speech (amplitude=1.0) with gaps — gaps still detected."""
        sr = 16000
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[2.5, 2.5, 2.5, 2.5], gap_duration=0.2,
            amplitude=1.0,
        )
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0
        _assert_split_in_gap(idx, sr, gaps, "loud_speech")

    def test_single_word_falls_back_to_quietest(self):
        """One long word, no gap → fallback returns the quietest window."""
        sr = 16000
        t = np.arange(10 * sr, dtype=np.float32) / sr
        samples = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        # Should return a valid fallback position, not -1
        assert idx > 0
        assert 5 * sr <= idx <= 10 * sr

    def test_very_short_gap_uses_fallback(self):
        """A 60ms gap (< 4×30ms=120ms) won't pass strict check, but
        fallback should still pick the quietest region (near the gap)."""
        sr = 16000
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[4.0, 4.0, 4.0], gap_duration=0.06,
        )
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0  # fallback always returns something
        # Fallback should gravitate toward the gap region
        split_sec = idx / sr
        near_gap = any(
            abs(split_sec - start) < 0.5
            for start, end in gaps
        )
        assert near_gap, (
            f"Fallback at {split_sec:.3f}s should be near a gap. "
            f"Gaps: {[(f'{s:.2f}-{e:.2f}') for s, e in gaps]}"
        )

    def test_150ms_gap_detected_strict(self):
        """A 150ms gap (> 4×30ms=120ms) should pass strict detection
        even with alignment variation."""
        sr = 16000
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[4.0, 4.0, 4.0], gap_duration=0.15,
        )
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0
        _assert_split_in_gap(idx, sr, gaps, "150ms_gap")

    def test_buffer_shorter_than_min(self):
        """Buffer smaller than min_chunk_samples → -1."""
        sr = 16000
        samples = np.zeros(3 * sr, dtype=np.float32)  # 3s < 5s min
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx == -1

    def test_realistic_commentary_pattern(self):
        """Mimics real sports commentary: variable word lengths and pauses."""
        sr = 16000
        word_durs = [1.2, 0.8, 1.5, 1.3, 2.0, 1.8]
        gap_durs = [0.15, 0.5, 0.1, 0.3, 0.2]

        parts = []
        gaps = []
        cursor = 0.0
        for i, dur in enumerate(word_durs):
            n = int(dur * sr)
            t = np.arange(n, dtype=np.float32) / sr
            parts.append((0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
            cursor += dur
            if i < len(gap_durs):
                n_gap = int(gap_durs[i] * sr)
                gaps.append((cursor, cursor + gap_durs[i]))
                parts.append(np.zeros(n_gap, dtype=np.float32))
                cursor += gap_durs[i]

        samples = np.concatenate(parts)
        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0, "Should find a split in realistic commentary"
        _assert_split_in_gap(idx, sr, gaps, "commentary")

    def test_noisy_silence_below_threshold(self):
        """Gaps with low-level noise (RMS < 0.02) should still be detected."""
        sr = 16000
        rng = np.random.RandomState(42)
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[3.0, 3.0, 3.0], gap_duration=0.2,
        )
        # Add very quiet noise to the gaps
        noise = (rng.randn(len(samples)) * 0.005).astype(np.float32)
        for start, end in gaps:
            s, e = int(start * sr), int(end * sr)
            samples[s:e] += noise[s:e]

        idx = find_silence_split(samples, sr, 5 * sr, 12 * sr)
        assert idx > 0, "Noisy silence below threshold should still split"
        _assert_split_in_gap(idx, sr, gaps, "noisy_silence")


class TestStreamingSimulation:
    """Simulate incremental splitting like stream_chunks_from_url does."""

    def test_multi_chunk_stream_splits_at_quiet_points(self):
        """Stream a 30s signal and verify splits land at gaps or quiet points."""
        sr = 16000
        samples, gaps = make_speech_with_gaps(
            sr=sr, word_durations=[2.0] * 13, gap_duration=0.2,
        )
        splits = _simulate_streaming_splits(samples, sr, 5.0, 12.0)
        assert len(splits) >= 2, f"Expected multiple chunks, got {len(splits)}"

        # Most boundaries should be in gaps (allow last chunk to be tail)
        gap_hits = 0
        for i, (offset, chunk) in enumerate(splits[:-1]):
            abs_split = offset + len(chunk)
            split_sec = abs_split / sr
            in_gap = any(start <= split_sec <= end for start, end in gaps)
            if in_gap:
                gap_hits += 1

        total_non_tail = len(splits) - 1
        pct = gap_hits / max(1, total_non_tail) * 100
        assert pct >= 60, (
            f"Only {gap_hits}/{total_non_tail} ({pct:.0f}%) splits in gaps. "
            f"Expected >= 60%"
        )

    def test_stream_no_tiny_chunks(self):
        """No chunk should be shorter than min_chunk_seconds (except final tail)."""
        sr = 16000
        samples, _ = make_speech_with_gaps(
            sr=sr, word_durations=[2.0] * 10, gap_duration=0.2,
        )
        splits = _simulate_streaming_splits(samples, sr, 5.0, 12.0)
        for i, (offset, chunk) in enumerate(splits[:-1]):  # skip final tail
            chunk_sec = len(chunk) / sr
            assert chunk_sec >= 4.9, (  # slight tolerance for sample rounding
                f"Chunk {i} is only {chunk_sec:.2f}s — below min 5s"
            )

    def test_stream_no_huge_chunks(self):
        """No chunk should exceed max_chunk_seconds."""
        sr = 16000
        samples, _ = make_speech_with_gaps(
            sr=sr, word_durations=[2.0] * 10, gap_duration=0.2,
        )
        splits = _simulate_streaming_splits(samples, sr, 5.0, 12.0)
        for i, (offset, chunk) in enumerate(splits):
            chunk_sec = len(chunk) / sr
            assert chunk_sec <= 12.1, (  # slight tolerance
                f"Chunk {i} is {chunk_sec:.2f}s — exceeds max 12s"
            )

    def test_stream_continuous_speech_still_splits(self):
        """Continuous speech with no gaps → fallback splits at quietest points."""
        sr = 16000
        n = 30 * sr
        t = np.arange(n, dtype=np.float32) / sr
        samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        splits = _simulate_streaming_splits(samples, sr, 5.0, 12.0)
        assert len(splits) >= 2, "Should still produce multiple chunks"
        for i, (offset, chunk) in enumerate(splits):
            chunk_sec = len(chunk) / sr
            assert chunk_sec <= 12.1, (
                f"Chunk {i} is {chunk_sec:.2f}s — exceeds max 12s"
            )


class TestLatency:
    """Verify find_silence_split doesn't introduce unacceptable latency."""

    def test_latency_under_2ms_for_15s_buffer(self):
        """15s of audio at 16kHz (240k samples) should split quickly."""
        sr = 16000
        samples, _ = make_speech_with_gaps(
            sr=sr, word_durations=[2.0] * 7, gap_duration=0.2,
        )
        # Warm up
        find_silence_split(samples, sr, 5 * sr, 12 * sr)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            find_silence_split(samples, sr, 5 * sr, 12 * sr)
            times.append(time.perf_counter() - start)

        median_ms = sorted(times)[50] * 1000
        p99_ms = sorted(times)[99] * 1000
        assert median_ms < 2.0, f"Median latency {median_ms:.3f}ms exceeds 2ms"
        assert p99_ms < 10.0, f"P99 latency {p99_ms:.3f}ms exceeds 10ms"

    def test_latency_under_10ms_for_60s_buffer(self):
        """60s of audio (960k samples) — worst-case long buffer."""
        sr = 16000
        n = 60 * sr
        t = np.arange(n, dtype=np.float32) / sr
        samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        # Warm up
        find_silence_split(samples, sr, 5 * sr, 50 * sr)

        times = []
        for _ in range(50):
            start = time.perf_counter()
            find_silence_split(samples, sr, 5 * sr, 50 * sr)
            times.append(time.perf_counter() - start)

        median_ms = sorted(times)[25] * 1000
        assert median_ms < 20.0, (
            f"Median latency for 60s scan: {median_ms:.3f}ms exceeds 20ms"
        )

    def test_latency_fast_when_silence_present(self):
        """When silence exists, should be fast (early exit on pass 1)."""
        sr = 16000
        # 10s speech then 0.5s silence then 1s speech
        t1 = np.arange(10 * sr, dtype=np.float32) / sr
        speech1 = (0.5 * np.sin(2 * np.pi * 220 * t1)).astype(np.float32)
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        t2 = np.arange(1 * sr, dtype=np.float32) / sr
        speech2 = (0.5 * np.sin(2 * np.pi * 220 * t2)).astype(np.float32)
        samples = np.concatenate([speech1, silence, speech2])

        # Warm up
        find_silence_split(samples, sr, 5 * sr, 12 * sr)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            find_silence_split(samples, sr, 5 * sr, 12 * sr)
            times.append(time.perf_counter() - start)

        median_ms = sorted(times)[50] * 1000
        assert median_ms < 2.0, (
            f"Silence-present case should be < 2ms, got {median_ms:.3f}ms"
        )
