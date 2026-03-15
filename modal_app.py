# modal_app.py
import modal
import os
import tempfile
import torch

from utils import TTSConfig, SpeakerMergeConfig
from main import QwenLocalVoiceCloner, XTTSLocalVoiceCloner, DynamicSpeakerTranslator

app = modal.App("sports-translation-api")

HF_CACHE = modal.Volume.from_name("sports-hf-cache", create_if_missing=True)
TORCH_CACHE = modal.Volume.from_name("sports-torch-cache", create_if_missing=True)
FEEDBACK_VOLUME = modal.Volume.from_name("sports-feedback", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "sox", "libsndfile1")
    .env({
        "HF_HOME": "/cache/hf",
        "TRANSFORMERS_CACHE": "/cache/hf",
        "TORCH_HOME": "/cache/torch",
        "TRANSLATION_BACKEND": os.getenv("TRANSLATION_BACKEND", "local"),
        "COQUI_TOS_AGREED": "1",
        "HF_HUB_DISABLE_XET": "1",
        "USE_VOICE_CLONING": "0",

    })
    .pip_install_from_requirements("requirements.txt")
    .pip_install_from_requirements("requirements_gpu.txt")
    .add_local_dir(".", remote_path="/root", copy=True)
)

secrets = [modal.Secret.from_name("sports-secrets")]

@app.cls(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    secrets=secrets,
    max_containers=1,
    volumes={"/cache/hf": HF_CACHE, "/cache/torch": TORCH_CACHE, "/feedback": FEEDBACK_VOLUME},
    enable_memory_snapshot=True
)
class TranslatorService:
    @modal.enter()
    def load_models(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.use_voice_cloning = os.getenv("USE_VOICE_CLONING", "0") == "1"

        self._translator_cache = {}
        self._sessions: dict = {}

        self._tts_config = TTSConfig(
            qwen_enable=True,
            qwen_device="cuda",
            xtts_enable=True,
            xtts_device="cuda",
        )
        self._speaker_merge = SpeakerMergeConfig()

        self._qwen = None
        self._xtts = None

        if self.use_voice_cloning:
            self._qwen = QwenLocalVoiceCloner(model_id=self._tts_config.qwen_model_id, device="cuda")
            self._xtts = XTTSLocalVoiceCloner(model_name=self._tts_config.xtts_model_id, device="cuda")

    def _get_translator(self, source_lang: str, target_lang: str):
        key = (source_lang, target_lang)
        if key in self._translator_cache:
            return self._translator_cache[key]
        
        translator = DynamicSpeakerTranslator(
            source_lang=source_lang,
            target_lang=target_lang,
            use_voice_cloning=self.use_voice_cloning,
            max_workers=3,
            tts_config=self._tts_config,
            speaker_merge=self._speaker_merge,
            qwen_cloner=self._qwen,
            xtts_cloner=self._xtts,
        )
        self._translator_cache[key] = translator
        return translator

    def _translate_wav_bytes_impl(self, wav_bytes: bytes, source_lang: str, target_lang: str):
        translator = self._get_translator(source_lang, target_lang)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(wav_bytes)
            result = translator.translate_audio_file_no_playback(path)
            # Returns (mp3_bytes, captions, detected_language)
            return result
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def translate_wav_bytes_stream_local(self, wav_bytes: bytes, source_lang: str, target_lang: str, buffer_sec: float = 120.0):
        translator = self._get_translator(source_lang, target_lang)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(wav_bytes)
            # This yields dict events
            yield from translator.translate_audio_file_stream(path, buffer_duration_sec=buffer_sec)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass 

    @modal.method()
    def translate_wav_bytes(self, wav_bytes: bytes, source_lang: str, target_lang: str):
        return self._translate_wav_bytes_impl(wav_bytes, source_lang, target_lang)

    def translate_wav_bytes_local(self, wav_bytes: bytes, source_lang: str, target_lang: str):
        return self._translate_wav_bytes_impl(wav_bytes, source_lang, target_lang)

    def create_session(self) -> str:
        import uuid
        sid = str(uuid.uuid4())
        self._sessions[sid] = {"speaker_voice_ids": {}, "speaker_pitches": {}, "chunk_offset_ms": 0}
        return sid

    def delete_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def process_live_chunk(self, wav_bytes: bytes, session_id: str, source_lang: str, target_lang: str, overlap_duration_sec: float = 0.0):
        session = self._sessions.get(session_id)
        if session is None:
            session = {"speaker_voice_ids": {}, "speaker_pitches": {}, "chunk_offset_ms": 0}
            self._sessions[session_id or "default"] = session
        translator = self._get_translator(source_lang, target_lang)
        yield from translator.translate_chunk_stream(wav_bytes, session, overlap_duration_sec=overlap_duration_sec)

    def store_feedback(self, data: dict):
        import json, datetime
        data["timestamp"] = datetime.datetime.utcnow().isoformat()
        with open("/feedback/feedback.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
        FEEDBACK_VOLUME.commit()

    @modal.asgi_app()
    def fastapi_app(self):
        from api_server import make_app
        api = make_app(self)

        return api
