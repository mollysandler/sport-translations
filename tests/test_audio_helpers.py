# tests/test_audio_helpers.py
"""
Tests for _wav_bytes_from_audio_np, _audio_np_from_wav_bytes, and
DynamicSpeakerTranslator._tts_bytes_to_wav in main.py.

_wav_bytes_from_audio_np uses scipy (available). _audio_np_from_wav_bytes uses
torchaudio.load which is stubbed — we monkeypatch it with a scipy-based
decoder for these tests. _tts_bytes_to_wav uses pydub (available with ffmpeg).
"""
import io
import wave

import numpy as np
import pytest

import main
from conftest import make_wav_bytes


def _scipy_wav_load(file_or_path):
    """Stand-in for torchaudio.load using scipy + wave module.

    Returns (tensor, sr) where tensor is a real torch.Tensor if torch is
    available, otherwise a lightweight _NpTensor wrapper.
    """
    if isinstance(file_or_path, (str, bytes)):
        f = open(file_or_path, "rb")
    else:
        f = file_or_path

    try:
        with wave.open(f, "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if nch > 1:
                pcm = pcm.reshape(-1, nch).T  # (nch, samples)
            else:
                pcm = pcm.reshape(1, -1)
    finally:
        if isinstance(file_or_path, (str, bytes)):
            f.close()

    # Use real torch.Tensor if available so torch.mean() works
    try:
        import torch as _torch
        return _torch.from_numpy(pcm.copy()), sr
    except (ImportError, Exception):
        pass

    # Fallback: lightweight wrapper
    class _NpTensor:
        def __init__(self, data):
            self._data = data
            self.shape = data.shape
        def __getitem__(self, key):
            return _NpTensor(self._data[key])
        def squeeze(self, dim=None):
            return _NpTensor(self._data.squeeze(axis=dim) if dim is not None else self._data.squeeze())
        def cpu(self):
            return self
        def numpy(self):
            return self._data
        def mean(self, dim=None, keepdim=False):
            return _NpTensor(np.mean(self._data, axis=dim, keepdims=keepdim))

    return _NpTensor(pcm), sr


# ═══════════════════════════════════════════
# _wav_bytes_from_audio_np
# ═══════════════════════════════════════════

class TestWavBytesFromAudioNp:
    def test_roundtrip_float32(self):
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        wav_bytes = main._wav_bytes_from_audio_np(audio, sr)
        # Should produce valid WAV
        assert wav_bytes[:4] == b"RIFF"
        assert b"WAVE" in wav_bytes[:32]

    def test_empty_array(self):
        sr = 16000
        audio = np.zeros(0, dtype=np.float32)
        wav_bytes = main._wav_bytes_from_audio_np(audio, sr)
        assert wav_bytes[:4] == b"RIFF"

    def test_2d_input_squeeze(self):
        sr = 16000
        audio = np.random.randn(1, 1600).astype(np.float32)
        wav_bytes = main._wav_bytes_from_audio_np(audio, sr)
        assert wav_bytes[:4] == b"RIFF"

    def test_int16_passthrough(self):
        sr = 16000
        audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        wav_bytes = main._wav_bytes_from_audio_np(audio, sr)
        assert wav_bytes[:4] == b"RIFF"


# ═══════════════════════════════════════════
# _audio_np_from_wav_bytes
# ═══════════════════════════════════════════

class TestAudioNpFromWavBytes:
    """Tests for _audio_np_from_wav_bytes.

    We always patch torchaudio.load with a scipy-based decoder so tests
    work regardless of whether torchcodec is installed.
    """

    def test_empty_bytes(self):
        audio, sr = main._audio_np_from_wav_bytes(b"")
        assert len(audio) == 0
        assert sr == 0

    def test_valid_wav(self, monkeypatch):
        monkeypatch.setattr(main.torchaudio, "load", _scipy_wav_load)
        wav_data = make_wav_bytes(sr=16000, seconds=0.5, nch=1)
        audio, sr = main._audio_np_from_wav_bytes(wav_data)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_stereo_to_mono(self, monkeypatch):
        monkeypatch.setattr(main.torchaudio, "load", _scipy_wav_load)
        wav_data = make_wav_bytes(sr=16000, seconds=0.5, nch=2)
        audio, sr = main._audio_np_from_wav_bytes(wav_data)
        assert sr == 16000
        assert audio.ndim == 1  # mono output


# ═══════════════════════════════════════════
# _tts_bytes_to_wav
# ═══════════════════════════════════════════

class TestTtsBytesToWav:
    def _make_translator(self):
        return main.DynamicSpeakerTranslator()

    def test_valid_wav_input(self):
        t = self._make_translator()
        wav_data = make_wav_bytes(sr=24000, seconds=0.5, nch=1)
        result_bytes, duration_ms = t._tts_bytes_to_wav(wav_data, output_sr=24000)
        assert result_bytes[:4] == b"RIFF"
        assert duration_ms > 0

    def test_empty_raises_value_error(self):
        t = self._make_translator()
        with pytest.raises(ValueError, match="empty"):
            t._tts_bytes_to_wav(b"", output_sr=24000)

    def test_resamples_to_output_sr(self):
        t = self._make_translator()
        # Input at 8000 Hz, output at 24000 Hz
        wav_data = make_wav_bytes(sr=8000, seconds=0.5, nch=1)
        result_bytes, duration_ms = t._tts_bytes_to_wav(wav_data, output_sr=24000)
        assert result_bytes[:4] == b"RIFF"
        # Duration should be approximately the same (~500ms)
        assert 400 <= duration_ms <= 600
