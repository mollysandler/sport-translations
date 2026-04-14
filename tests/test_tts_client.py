"""Tests for tts_client.py stock voice assignment."""

from tts_client import TTSClient, STOCK_VOICES


class TestStockVoiceAssignment:
    def test_male_low_pitch(self):
        client = TTSClient.__new__(TTSClient)
        client._api_key = "test"
        client._used_stock_voices = set()
        voice_id = client.assign_stock_voice(gender="male", avg_pitch=120.0)
        assert voice_id in STOCK_VOICES
        assert STOCK_VOICES[voice_id]["gender"] == "male"

    def test_female_high_pitch(self):
        client = TTSClient.__new__(TTSClient)
        client._api_key = "test"
        client._used_stock_voices = set()
        voice_id = client.assign_stock_voice(gender="female", avg_pitch=200.0)
        assert voice_id in STOCK_VOICES
        assert STOCK_VOICES[voice_id]["gender"] == "female"

    def test_avoids_reuse(self):
        client = TTSClient.__new__(TTSClient)
        client._api_key = "test"
        client._used_stock_voices = set()
        v1 = client.assign_stock_voice(gender="male", avg_pitch=150.0)
        v2 = client.assign_stock_voice(gender="male", avg_pitch=150.0)
        assert v1 != v2  # should pick different voices

    def test_all_stock_voices_have_required_fields(self):
        for vid, props in STOCK_VOICES.items():
            assert "gender" in props
            assert "pitch" in props
            assert "style" in props
            assert props["gender"] in ("male", "female")

    def test_exhausted_voices_allows_reuse(self):
        client = TTSClient.__new__(TTSClient)
        client._api_key = "test"
        # Mark all male voices as used
        client._used_stock_voices = {
            vid for vid, p in STOCK_VOICES.items() if p["gender"] == "male"
        }
        # Should still return a voice (from female pool or reuse)
        voice_id = client.assign_stock_voice(gender="male", avg_pitch=150.0)
        assert voice_id in STOCK_VOICES
