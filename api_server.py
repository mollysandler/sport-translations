import os, tempfile, traceback, base64
import io
import torch, torchaudio
from typing import Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from fastapi.responses import Response

def wav_bytes_16k_mono(upload: UploadFile) -> bytes:
    data = upload.file.read()
    wav, sr = torchaudio.load(io.BytesIO(data))  # supports wav/mp3 if ffmpeg backend; wav always ok
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    out = io.BytesIO()
    torchaudio.save(out, wav, 16000, format="wav")
    return out.getvalue()

def _running_on_modal() -> bool:
    # Modal sets task-related env vars inside containers.
    return bool(os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IS_REMOTE"))

def make_app(service: Any) -> FastAPI:
    app = FastAPI(redirect_slashes=False)

    # IMPORTANT: for local dev (localhost:5173) use explicit origins.
    # "*" + allow_credentials=True is invalid per CORS spec and can cause missing headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    @app.post("/translate-audio")
    def translate_audio(
        audio: UploadFile = File(...),
        source_lang: str = Form("en"),
        target_lang: str = Form("hi"),
    ):

        wav_bytes = wav_bytes_16k_mono(audio)
        if _running_on_modal():
            mp3_bytes, captions = service.translate_wav_bytes_local(wav_bytes, source_lang, target_lang)
        else:
            mp3_bytes, captions = service.translate_wav_bytes.remote(wav_bytes, source_lang, target_lang)


        return {
            "audio_base64": base64.b64encode(mp3_bytes).decode("utf-8"),
            "captions": captions,
        }

    return app
