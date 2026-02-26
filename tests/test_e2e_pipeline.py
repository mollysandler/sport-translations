# tests/test_e2e_pipeline.py
"""
End-to-end tests for translate_audio_file_no_playback with a fully mocked
service chain:  diarizer → whisper → translator → elevenlabs TTS → compose.

All heavy deps are stubbed so tests run fast with no real model inference.
"""
import io
import types

import numpy as np
import pytest

import main
from conftest import seg, make_wav_bytes


# ---------------------------------------------------------------------------
# Numpy-backed tensor that quacks enough like torch.Tensor for main.py
# ---------------------------------------------------------------------------

class _NpTensor:
    """Lightweight tensor-like wrapper around a numpy array."""
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)
        self.shape = self._data.shape

    def __getitem__(self, key):
        return _NpTensor(self._data[key])

    def squeeze(self, dim=None):
        return _NpTensor(self._data.squeeze(axis=dim) if dim is not None else self._data.squeeze())

    def numpy(self):
        return self._data

    def mean(self, dim=None, keepdim=False):
        return _NpTensor(np.mean(self._data, axis=dim, keepdims=keepdim))

    def to(self, *a, **k):
        return self

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


# ---------------------------------------------------------------------------
# Fake service stubs
# ---------------------------------------------------------------------------

class FakeWhisperSegment:
    """Mimics faster_whisper segment objects."""
    def __init__(self, text: str):
        self.text = text


class FakeWhisper:
    """Fake faster-whisper model."""
    def __init__(self, texts=None):
        # texts: list of lists — one inner list per call
        self._texts = texts or [["Hello world"]]
        self._call_idx = 0

    def transcribe(self, audio_np, **kwargs):
        idx = min(self._call_idx, len(self._texts) - 1)
        segs = [FakeWhisperSegment(t) for t in self._texts[idx]]
        self._call_idx += 1
        return iter(segs), None


class FakeTranslator:
    """Fake deep_translator GoogleTranslator."""
    def __init__(self, prefix="translated: "):
        self._prefix = prefix

    def translate(self, text: str) -> str:
        return self._prefix + text


class FakeElevenLabs:
    """Fake ElevenLabs client that returns valid WAV bytes from text_to_speech.convert."""
    def __init__(self):
        self.text_to_speech = types.SimpleNamespace(convert=self._convert)

    def _convert(self, **kwargs):
        wav = make_wav_bytes(sr=24000, seconds=0.3, nch=1)
        yield wav


class FakeVoiceSettings:
    stability = 0.5
    similarity_boost = 0.75
    style = 0.0
    use_speaker_boost = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_translator(
    segments=None,
    whisper_texts=None,
    voice_ids=None,
):
    """
    Build a DynamicSpeakerTranslator with all services replaced by fakes.
    """
    t = main.DynamicSpeakerTranslator()

    # Services
    t.whisper = FakeWhisper(texts=whisper_texts)
    t.translator = FakeTranslator()
    t.use_local_translation = False
    t.elevenlabs = FakeElevenLabs()
    t.voice_settings = FakeVoiceSettings()
    t.use_voice_cloning = False

    # Diarizer segments — bypass the real _get_speaker_segments
    _segments = segments if segments is not None else [seg("SPEAKER_00", 0, 2000)]
    t._get_speaker_segments = lambda audio, path: _segments

    # Voice assignment
    unique = sorted(set(s.speaker_id for s in _segments))
    default_ids = voice_ids or {
        spk: list(t.voice_manager.available_voices.keys())[i % 10]
        for i, spk in enumerate(unique)
    }
    t.voice_manager.assign_voices = lambda audio, segs, sr: default_ids

    return t


def _run_pipeline(t, duration_sec=3.0):
    """
    Run translate_audio_file_no_playback with a fake wav file.
    We monkeypatch torchaudio.load to return a numpy-backed tensor.
    """
    samples = int(duration_sec * t.sample_rate)
    fake_waveform = _NpTensor(np.zeros((1, samples), dtype=np.float32))

    original_load = main.torchaudio.load

    def fake_load(path):
        return fake_waveform, t.sample_rate

    main.torchaudio.load = fake_load
    try:
        return t.translate_audio_file_no_playback("/tmp/fake.wav")
    finally:
        main.torchaudio.load = original_load


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2EPipeline:
    def test_happy_path_produces_mp3_and_captions(self):
        segs = [seg("SPEAKER_00", 0, 2000), seg("SPEAKER_01", 2500, 4000)]
        t = _build_translator(
            segments=segs,
            whisper_texts=[["Hello"], ["World"]],
        )
        mp3_bytes, captions = _run_pipeline(t, duration_sec=5.0)

        assert isinstance(mp3_bytes, bytes)
        assert len(mp3_bytes) > 0
        assert isinstance(captions, list)
        assert len(captions) == 2

    def test_captions_match_segments(self):
        segs = [seg("SPEAKER_00", 1000, 3000)]
        t = _build_translator(
            segments=segs,
            whisper_texts=[["Test sentence"]],
        )
        mp3_bytes, captions = _run_pipeline(t, duration_sec=4.0)

        assert len(captions) == 1
        c = captions[0]
        assert c["speaker"] == "SPEAKER_00"
        assert c["original"] == "Test sentence"
        assert c["translated"].startswith("translated: ")

    def test_caption_timing_matches_boundaries(self):
        segs = [seg("A", 500, 2500)]
        t = _build_translator(segments=segs, whisper_texts=[["Hello"]])
        _, captions = _run_pipeline(t, duration_sec=3.0)

        assert len(captions) == 1
        assert captions[0]["startTime"] == pytest.approx(0.5)
        assert captions[0]["endTime"] == pytest.approx(2.5)

    def test_empty_diarization_returns_silent_and_empty_captions(self):
        t = _build_translator(segments=[])
        mp3_bytes, captions = _run_pipeline(t, duration_sec=2.0)

        assert isinstance(mp3_bytes, bytes)
        assert len(mp3_bytes) > 0  # even silence produces an mp3
        assert captions == []

    def test_single_speaker(self):
        segs = [seg("SPEAKER_00", 0, 1500)]
        t = _build_translator(segments=segs, whisper_texts=[["Solo"]])
        mp3_bytes, captions = _run_pipeline(t, duration_sec=2.0)

        assert len(captions) == 1
        assert captions[0]["speaker"] == "SPEAKER_00"

    def test_empty_asr_text_skips_segment(self):
        segs = [seg("SPEAKER_00", 0, 2000)]
        t = _build_translator(segments=segs, whisper_texts=[[""]])
        mp3_bytes, captions = _run_pipeline(t, duration_sec=3.0)

        assert len(mp3_bytes) > 0
        assert captions == []
