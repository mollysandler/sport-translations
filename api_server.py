import os, tempfile, traceback, base64, json
import io
from typing import Any
import torch, torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydub import AudioSegment

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

    @app.post("/session/start")
    def session_start():
        return {"session_id": service.create_session()}

    @app.post("/session/end")
    def session_end(session_id: str = Form(...)):
        service.delete_session(session_id)
        return {"ok": True}

    @app.post("/translate-live")
    def translate_live(
        audio: UploadFile = File(...),
        session_id: str = Form(""),
        source_lang: str = Form("en"),
        target_lang: str = Form("hi"),
    ):
        wav_bytes = wav_bytes_16k_mono(audio)

        def event_stream():
            try:
                for item in service.process_live_chunk(wav_bytes, session_id, source_lang, target_lang):
                    yield f"data: {json.dumps(item)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
