import base64
import io
import wave
import numpy as np
from fastapi.testclient import TestClient

import api_server


def make_wav_bytes(sr=16000, seconds=0.25, freq=440.0):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes((x * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def test_translate_audio_happy_path(monkeypatch):
    # 1) Don’t run pydub conversion during tests
    monkeypatch.setattr(api_server, "_ensure_wav_16k_mono", lambda p: p)

    # 2) Mock translator class used by the endpoint
    class FakeTranslator:
        def __init__(self, **kwargs):
            # keep kwargs if you want to assert config wiring later
            self.kwargs = kwargs

        def translate_audio_file_no_playback(self, wav_path: str):
            mp3_bytes = b"fake-mp3-bytes"
            captions = [{"start": 0.0, "end": 1.0, "text": "hello"}]
            return mp3_bytes, captions

    monkeypatch.setattr(api_server, "DynamicSpeakerTranslator", FakeTranslator)

    client = TestClient(api_server.app)

    wav_bytes = make_wav_bytes()
    files = {"audio": ("test.wav", wav_bytes, "audio/wav")}
    data = {"source_lang": "en", "target_lang": "hi"}

    r = client.post("/translate-audio", files=files, data=data)
    assert r.status_code == 200

    payload = r.json()
    assert "audio_base64" in payload
    assert "captions" in payload

    # base64 decodes to our fake bytes
    assert base64.b64decode(payload["audio_base64"]) == b"fake-mp3-bytes"
    assert payload["captions"][0]["text"] == "hello"
