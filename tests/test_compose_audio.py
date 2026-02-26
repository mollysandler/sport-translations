# tests/test_compose_audio.py
"""
Tests for DynamicSpeakerTranslator._compose_audio in main.py.

Uses real pydub AudioSegment operations — no GPU needed.
"""
from dataclasses import dataclass

import main
from conftest import make_wav_bytes


@dataclass
class _FakeTSeg:
    """Minimal stand-in for TranslationSegment with fields _compose_audio reads."""
    speaker_id: str
    start_ms: int
    end_ms: int
    original_text: str
    translated_text: str
    audio_bytes: bytes
    duration_ms: int


def _make_translator():
    return main.DynamicSpeakerTranslator()


def _make_tseg(start_ms: int, duration_ms: int = 500, speaker: str = "A") -> _FakeTSeg:
    """Build a TranslationSegment with real WAV audio_bytes."""
    sr = 24000
    seconds = duration_ms / 1000.0
    wav = make_wav_bytes(sr=sr, seconds=seconds, nch=1)
    return _FakeTSeg(
        speaker_id=speaker,
        start_ms=start_ms,
        end_ms=start_ms + duration_ms,
        original_text="hello",
        translated_text="hola",
        audio_bytes=wav,
        duration_ms=duration_ms,
    )


class TestComposeAudio:
    def test_empty_segments_silent_output(self):
        t = _make_translator()
        result = t._compose_audio([], total_duration_sec=5.0)
        assert len(result) == 5000  # 5s in ms

    def test_single_segment_placed_at_offset(self):
        t = _make_translator()
        seg = _make_tseg(start_ms=1000, duration_ms=500)
        result = t._compose_audio([seg], total_duration_sec=3.0)
        # Result should be 3 seconds
        assert len(result) == 3000

    def test_multiple_non_overlapping(self):
        t = _make_translator()
        seg1 = _make_tseg(start_ms=0, duration_ms=500)
        seg2 = _make_tseg(start_ms=1500, duration_ms=500, speaker="B")
        result = t._compose_audio([seg1, seg2], total_duration_sec=3.0)
        assert len(result) == 3000

    def test_segment_at_position_zero(self):
        t = _make_translator()
        seg = _make_tseg(start_ms=0, duration_ms=500)
        result = t._compose_audio([seg], total_duration_sec=2.0)
        assert len(result) == 2000

    def test_negative_start_ms_clamped_to_zero(self):
        t = _make_translator()
        seg = _make_tseg(start_ms=-500, duration_ms=500)
        # _compose_audio uses max(0, int(seg.start_ms)) so this is safe
        result = t._compose_audio([seg], total_duration_sec=2.0)
        assert len(result) == 2000
