# tests/test_diarizer.py
"""
Tests for the pure-logic segment processing helpers in diarizer.py.

We import the real diarizer module (not the conftest stub) and bypass
the heavy __init__ by using object.__new__() to create a bare instance.
"""
import sys
import os

# Ensure we can import the real diarizer module (not the conftest stub)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Remove the conftest stub so we get the real module
_had_stub = "diarizer" in sys.modules
_saved = sys.modules.pop("diarizer", None)

# We need numpy but NOT torch/pyannote at module level in the real diarizer.
# Patch out heavy imports before importing diarizer.
import types as _types

# Create lightweight stubs for heavy deps that diarizer.py imports at top level
for _mod_name in ("torch", "pyannote", "pyannote.audio", "librosa", "huggingface_hub"):
    if _mod_name not in sys.modules:
        _stub = _types.ModuleType(_mod_name)
        if _mod_name == "torch":
            # diarizer uses torch.Tensor type hints and torch.device
            class _FakeTensor:
                pass
            _stub.Tensor = _FakeTensor
            _stub.device = lambda *a, **k: None
            _stub.no_grad = lambda: _types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *a: None)
            class _FakeBackends:
                class mps:
                    @staticmethod
                    def is_available():
                        return False
            _stub.backends = _FakeBackends()
            @staticmethod
            def _cuda_avail():
                return False
            _stub.cuda = _types.SimpleNamespace(is_available=_cuda_avail)
            _stub.cat = lambda tensors, dim=0: None
        elif _mod_name == "pyannote.audio":
            _stub.Pipeline = type("Pipeline", (), {"from_pretrained": classmethod(lambda cls, *a, **k: None)})
        elif _mod_name == "librosa":
            pass
        elif _mod_name == "huggingface_hub":
            _stub.login = lambda **k: None
        sys.modules[_mod_name] = _stub

# Now we need utils to be importable too
if "utils" not in sys.modules:
    sys.path.insert(0, _project_root)

import importlib
_real_diarizer = importlib.import_module("diarizer")

# Restore conftest stub for other test files
if _had_stub and _saved is not None:
    sys.modules["diarizer"] = _saved

import numpy as np
import pytest

from utils import SpeakerMergeConfig

# Import real classes/functions from the real diarizer module
_cosine_sim = _real_diarizer._cosine_sim
RealSpeakerSegment = _real_diarizer.SpeakerSegment
RealSpeakerDiarizer = _real_diarizer.SpeakerDiarizer


def _seg(speaker: str, start_ms: int, end_ms: int) -> RealSpeakerSegment:
    return RealSpeakerSegment(
        speaker_id=speaker,
        start_ms=start_ms,
        end_ms=end_ms,
        start_sec=start_ms / 1000.0,
        end_sec=end_ms / 1000.0,
    )


def _make_diarizer(merge_config=None):
    """Create a bare SpeakerDiarizer without calling __init__."""
    d = object.__new__(RealSpeakerDiarizer)
    d.merge_config = merge_config or SpeakerMergeConfig()
    d._spkrec = None
    return d


# ═══════════════════════════════════════════
# _cosine_sim
# ═══════════════════════════════════════════

class TestCosineSim:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_sim(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        # Should not crash (eps protects division); result is ~0
        result = _cosine_sim(a, b)
        assert abs(result) < 0.01


# ═══════════════════════════════════════════
# _filter_short_segments (MIN_SEGMENT_DURATION_MS = 300)
# ═══════════════════════════════════════════

class TestFilterShortSegments:
    def test_removes_below_300ms(self):
        d = _make_diarizer()
        segs = [_seg("A", 0, 200), _seg("B", 500, 1500)]
        result = d._filter_short_segments(segs)
        assert len(result) == 1
        assert result[0].speaker_id == "B"

    def test_empty_input(self):
        d = _make_diarizer()
        assert d._filter_short_segments([]) == []

    def test_all_short(self):
        d = _make_diarizer()
        segs = [_seg("A", 0, 100), _seg("B", 200, 400)]
        result = d._filter_short_segments(segs)
        assert len(result) == 0


# ═══════════════════════════════════════════
# _merge_close_segments (SPEAKER_MERGE_GAP_SECONDS = 0.3)
# ═══════════════════════════════════════════

class TestMergeCloseSegments:
    def test_same_speaker_small_gap_merges(self):
        d = _make_diarizer()
        segs = [_seg("A", 0, 1000), _seg("A", 1200, 2000)]  # 200ms gap
        result = d._merge_close_segments(segs)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000

    def test_same_speaker_large_gap_no_merge(self):
        d = _make_diarizer()
        segs = [_seg("A", 0, 1000), _seg("A", 1500, 2000)]  # 500ms gap
        result = d._merge_close_segments(segs)
        assert len(result) == 2

    def test_different_speakers_no_merge(self):
        d = _make_diarizer()
        segs = [_seg("A", 0, 1000), _seg("B", 1100, 2000)]  # 100ms gap, diff speaker
        result = d._merge_close_segments(segs)
        assert len(result) == 2

    def test_chain_merge(self):
        d = _make_diarizer()
        segs = [
            _seg("A", 0, 1000),
            _seg("A", 1100, 2000),   # 100ms gap
            _seg("A", 2200, 3000),   # 200ms gap
        ]
        result = d._merge_close_segments(segs)
        assert len(result) == 1
        assert result[0].end_ms == 3000

    def test_unsorted_input(self):
        d = _make_diarizer()
        segs = [_seg("A", 1200, 2000), _seg("A", 0, 1000)]  # reversed
        result = d._merge_close_segments(segs)
        assert len(result) == 1  # sorted then merged

    def test_empty(self):
        d = _make_diarizer()
        assert d._merge_close_segments([]) == []


# ═══════════════════════════════════════════
# _split_long_segments
# ═══════════════════════════════════════════

class TestSplitLongSegments:
    def test_no_split_needed(self):
        d = _make_diarizer()
        segs = [_seg("A", 0, 5000)]
        result = d._split_long_segments(segs, max_duration_sec=10.0)
        assert len(result) == 1

    def test_even_split(self):
        d = _make_diarizer()
        # 20s segment, max 10s → 2 pieces
        segs = [_seg("A", 0, 20000)]
        result = d._split_long_segments(segs, max_duration_sec=10.0)
        assert len(result) == 2
        assert result[0].start_ms == 0
        assert result[0].end_ms == 10000
        assert result[1].start_ms == 10000
        assert result[1].end_ms == 20000

    def test_remainder(self):
        d = _make_diarizer()
        # 25s segment, max 10s → 3 pieces (10+10+5)
        segs = [_seg("A", 0, 25000)]
        result = d._split_long_segments(segs, max_duration_sec=10.0)
        assert len(result) == 3
        assert result[2].duration_ms == 5000

    def test_mixed_short_and_long(self):
        d = _make_diarizer()
        segs = [_seg("A", 0, 5000), _seg("B", 5000, 25000)]
        result = d._split_long_segments(segs, max_duration_sec=10.0)
        # 1 short (5s) + 2 splits of 20s (10+10) = 3 total
        assert len(result) == 3


# ═══════════════════════════════════════════
# _collect_ref_audio
# ═══════════════════════════════════════════

def _has_real_torch():
    """Check if real torch (not our stub) is available."""
    try:
        import torch
        return hasattr(torch, "randn") and callable(getattr(torch.randn(1), "shape").__class__.__getitem__)
    except Exception:
        return False

_real_torch_available = _has_real_torch()


@pytest.mark.skipif(not _real_torch_available, reason="real torch not installed")
class TestCollectRefAudio:
    def test_basic_concat(self):
        import torch
        d = _make_diarizer()
        waveform = torch.randn(1, 48000)  # 3s at 16kHz
        segs = [_seg("A", 0, 1500), _seg("A", 2000, 3000)]
        result = d._collect_ref_audio(waveform, 16000, segs, target_sec=12.0)
        assert result is not None
        assert result.shape[0] == 1  # mono

    def test_skips_short_chunks(self):
        import torch
        d = _make_diarizer(SpeakerMergeConfig(emb_min_chunk_ms=500))
        waveform = torch.randn(1, 48000)
        # 200ms segment is too short (min_chunk_ms=500)
        segs = [_seg("A", 0, 200)]
        result = d._collect_ref_audio(waveform, 16000, segs, target_sec=12.0)
        assert result is None

    def test_respects_target_sec(self):
        import torch
        d = _make_diarizer()
        waveform = torch.randn(1, 160000)  # 10s at 16kHz
        segs = [_seg("A", 0, 5000), _seg("A", 5000, 10000)]
        result = d._collect_ref_audio(waveform, 16000, segs, target_sec=3.0)
        assert result is not None
        # Should have collected roughly up to 3s (may include first full segment)
        collected_sec = result.shape[1] / 16000
        assert collected_sec <= 6.0  # at most one full 5s segment
