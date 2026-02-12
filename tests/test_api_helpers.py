import os
import io
import wave
import numpy as np

import api_server
import base64
from fastapi.testclient import TestClient
import api_server

class DummyUpload:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.file = io.BytesIO(data)


def _make_wav_bytes(sr=8000, seconds=0.1, nch=2):
    # simple PCM int16 WAV
    n = int(sr * seconds)
    x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype(np.float32)
    pcm = (x * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # duplicate channels if stereo
        if nch == 2:
            interleaved = np.column_stack([pcm, pcm]).ravel().astype(np.int16)
            wf.writeframes(interleaved.tobytes())
        else:
            wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_save_upload_to_temp_preserves_extension(tmp_path, monkeypatch):
    data = b"abc123"
    up = DummyUpload("clip.m4a", data)

    path = api_server._save_upload_to_temp(up)
    try:
        assert path.endswith(".m4a")
        with open(path, "rb") as f:
            assert f.read() == data
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_save_upload_to_temp_defaults_to_wav_if_no_name():
    data = b"abc123"
    up = DummyUpload("", data)

    path = api_server._save_upload_to_temp(up)
    try:
        assert path.endswith(".wav")
    finally:
        if os.path.exists(path):
            os.remove(path)

def test_translate_audio_wires_configs_and_converts_bools(monkeypatch):
    # skip conversion
    monkeypatch.setattr(api_server, "_ensure_wav_16k_mono", lambda p: p)
    monkeypatch.setattr(api_server, "_save_upload_to_temp", lambda upload: "/tmp/in.wav")

    captured = {}

    class SpyTranslator:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def translate_audio_file_no_playback(self, wav_path: str):
            return b"mp3", [{"start": 0.0, "end": 0.1, "text": "x"}]

    monkeypatch.setattr(api_server, "DynamicSpeakerTranslator", SpyTranslator)

    client = TestClient(api_server.app, raise_server_exceptions=False)

    files = {"audio": ("x.wav", b"fake", "audio/wav")}
    data = {
        "source_lang": "en",
        "target_lang": "hi",
        "max_workers": "7",
        "use_voice_cloning": "false",     # FastAPI will coerce to bool
        "qwen_tts_enable": "0",           # you convert via bool(int)
        "xtts_enable": "1",
        "speaker_merge_enable": "0",
        "speaker_merge_sim": "0.8",
    }

    r = client.post("/translate-audio", files=files, data=data)
    assert r.status_code == 200

    kwargs = captured["kwargs"]
    assert kwargs["source_lang"] == "en"
    assert kwargs["target_lang"] == "hi"
    assert kwargs["max_workers"] == 7
    assert kwargs["use_voice_cloning"] is False

    tts_cfg = kwargs["tts_config"]
    assert tts_cfg.qwen_enable is False
    assert tts_cfg.xtts_enable is True

    sp_cfg = kwargs["speaker_merge"]
    assert sp_cfg.merge_enable is False
    assert sp_cfg.merge_sim == 0.8

    payload = r.json()
    assert base64.b64decode(payload["audio_base64"]) == b"mp3"
    assert isinstance(payload["captions"], list)


def test_translate_audio_defaults_stable(monkeypatch):
    monkeypatch.setattr(api_server, "_ensure_wav_16k_mono", lambda p: p)
    monkeypatch.setattr(api_server, "_save_upload_to_temp", lambda upload: "/tmp/in.wav")

    captured = {}

    class SpyTranslator:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def translate_audio_file_no_playback(self, wav_path: str):
            return b"mp3", []

    monkeypatch.setattr(api_server, "DynamicSpeakerTranslator", SpyTranslator)

    client = TestClient(api_server.app, raise_server_exceptions=False)
    files = {"audio": ("x.wav", b"fake", "audio/wav")}
    r = client.post("/translate-audio", files=files, data={})
    assert r.status_code == 200

    kwargs = captured["kwargs"]
    assert kwargs["source_lang"] == "en"
    assert kwargs["target_lang"] == "hi"
    assert kwargs["buffer_duration_sec"] == 30
    assert kwargs["max_workers"] == 3
    assert kwargs["use_voice_cloning"] is True


def test_translate_audio_conversion_failure_returns_500(monkeypatch):
    def boom(_):
        raise RuntimeError("convert failed")

    monkeypatch.setattr(api_server, "_ensure_wav_16k_mono", boom)
    monkeypatch.setattr(api_server, "_save_upload_to_temp", lambda upload: "/tmp/in.wav")

    client = TestClient(api_server.app, raise_server_exceptions=False)
    files = {"audio": ("x.wav", b"fake", "audio/wav")}
    r = client.post("/translate-audio", files=files, data={})
    assert r.status_code == 500


def test_translate_audio_translator_failure_returns_500(monkeypatch):
    monkeypatch.setattr(api_server, "_ensure_wav_16k_mono", lambda p: p)
    monkeypatch.setattr(api_server, "_save_upload_to_temp", lambda upload: "/tmp/in.wav")

    class BadTranslator:
        def __init__(self, **kwargs):
            pass
        def translate_audio_file_no_playback(self, wav_path: str):
            raise RuntimeError("translator failed")

    monkeypatch.setattr(api_server, "DynamicSpeakerTranslator", BadTranslator)

    client = TestClient(api_server.app, raise_server_exceptions=False)
    files = {"audio": ("x.wav", b"fake", "audio/wav")}
    r = client.post("/translate-audio", files=files, data={})
    assert r.status_code == 500


def test_cors_allows_vite_origin():

    client = TestClient(api_server.app, raise_server_exceptions=False)
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    r = client.options("/translate-audio", headers=headers)
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
