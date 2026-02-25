# api_server.py
import os, base64, io, json
from typing import Any, Iterator

import torch, torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


def wav_bytes_16k_mono(upload: UploadFile) -> bytes:
    data = upload.file.read()
    wav, sr = torchaudio.load(io.BytesIO(data))  # wav always ok; mp3 depends on ffmpeg backend
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    out = io.BytesIO()
    torchaudio.save(out, wav, 16000, format="wav")
    return out.getvalue()


def _running_on_modal() -> bool:
    return bool(os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IS_REMOTE"))


def make_app(service: Any) -> FastAPI:
    app = FastAPI(redirect_slashes=False)

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

    @app.post("/translate-audio-stream")
    def translate_audio_stream(
        audio: UploadFile = File(...),
        source_lang: str = Form("en"),
        target_lang: str = Form("hi"),
        buffer_sec: float = Form(120.0),
    ):
        wav_bytes = wav_bytes_16k_mono(audio)

        # This endpoint is designed to stream while running INSIDE the Modal container.
        if not _running_on_modal():
            raise HTTPException(
                status_code=400,
                detail="Streaming endpoint only supported on Modal deployment. Use /translate-audio locally.",
            )

        def ndjson() -> Iterator[bytes]:
            try:
                # IMPORTANT: this yields dict events from your TranslatorService generator
                for event in service.translate_wav_bytes_stream_local(
                    wav_bytes, source_lang, target_lang, buffer_sec
                ):
                    yield (json.dumps(event) + "\n").encode("utf-8")
            except Exception as e:
                yield (json.dumps({"type": "error", "message": str(e)}) + "\n").encode("utf-8")

        return StreamingResponse(
            ndjson(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return app
