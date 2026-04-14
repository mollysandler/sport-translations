"""Tests for speaker_manager.py."""

import asyncio
import struct
import pytest
from unittest.mock import AsyncMock, MagicMock

from speaker_manager import SpeakerManager, _pcm16_to_wav
from tts_client import TTSClient


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestPcm16ToWav:
    def test_wav_header(self):
        pcm = b"\x00\x01" * 100  # 200 bytes of PCM
        wav = _pcm16_to_wav(pcm, sample_rate=16000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "

    def test_data_preserved(self):
        pcm = b"\x00\x01\x02\x03"
        wav = _pcm16_to_wav(pcm)
        # Data starts after 44-byte header
        assert wav[44:] == pcm

    def test_correct_file_size(self):
        pcm = b"\x00" * 320
        wav = _pcm16_to_wav(pcm)
        # RIFF size = file_size - 8
        riff_size = struct.unpack("<I", wav[4:8])[0]
        assert riff_size == len(wav) - 8


class TestSpeakerManager:
    def _make_manager(self, enable_cloning=False):
        tts = MagicMock(spec=TTSClient)
        tts.assign_stock_voice = MagicMock(return_value="voice-abc")
        tts.clone_voice = AsyncMock(return_value="cloned-xyz")
        tts.delete_voice = AsyncMock()
        mgr = SpeakerManager(tts, enable_cloning=enable_cloning)
        return mgr, tts

    def test_get_voice_id_assigns_stock(self):
        mgr, tts = self._make_manager()
        voice = mgr.get_voice_id(0)
        assert voice == "voice-abc"
        tts.assign_stock_voice.assert_called_once()

    def test_get_voice_id_reuses_existing(self):
        mgr, tts = self._make_manager()
        v1 = mgr.get_voice_id(0)
        v2 = mgr.get_voice_id(0)
        assert v1 == v2
        # Should only assign once
        assert tts.assign_stock_voice.call_count == 1

    def test_multiple_speakers_get_different_calls(self):
        mgr, tts = self._make_manager()
        mgr.get_voice_id(0)
        mgr.get_voice_id(1)
        assert tts.assign_stock_voice.call_count == 2

    def test_add_audio_accumulates(self):
        mgr, _ = self._make_manager()
        mgr.get_voice_id(0)
        pcm = b"\x00\x01" * 1600  # 0.1s at 16kHz
        mgr.add_audio(0, pcm)
        info = mgr._speakers[0]
        assert len(info.audio_samples) == len(pcm)
        assert info.audio_sec == pytest.approx(0.1, abs=0.01)

    def test_no_clone_when_disabled(self):
        mgr, tts = self._make_manager(enable_cloning=False)
        mgr.get_voice_id(0)
        # Add enough audio to trigger cloning if enabled (15 seconds)
        pcm = b"\x00\x01" * (16000 * 15)  # 15 seconds
        mgr.add_audio(0, pcm)
        # No clone task should be created
        assert len(mgr._clone_tasks) == 0

    def test_cleanup(self, event_loop):
        mgr, tts = self._make_manager()
        mgr._cloned_voice_ids = ["v1", "v2"]
        event_loop.run_until_complete(mgr.cleanup())
        assert tts.delete_voice.call_count == 2
        assert len(mgr._speakers) == 0
