# modal_app.py
import modal

app = modal.App("sports-translation-api")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "sox", "libsndfile1")
    .pip_install_from_requirements("requirements.txt")
    .pip_install_from_requirements("requirements_gpu.txt")
    # Your codebase is copied into the container so imports like `from main import ...` work
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
    enable_memory_snapshot=True,
    max_containers=1,  # cheapest / simplest; increase only if you need more throughput
)
class TranslatorService:
    @modal.enter(snap=True)
    def load_models(self):
        # GPU perf toggles (fine to keep)
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Cache translators by language pair so we don't rebuild each request.
        self._translator_cache = {}

        # If these configs are static, build once and reuse.
        from utils import TTSConfig, SpeakerMergeConfig
        self._tts_config = TTSConfig(xtts_enable=False)
        self._speaker_merge = SpeakerMergeConfig()

    def _get_translator(self, source_lang: str, target_lang: str):
        key = (source_lang, target_lang)
        if key in self._translator_cache:
            return self._translator_cache[key]

        from main import DynamicSpeakerTranslator
        translator = DynamicSpeakerTranslator(
            source_lang=source_lang,
            target_lang=target_lang,
            use_voice_cloning=True,
            max_workers=3,
            tts_config=self._tts_config,
            speaker_merge=self._speaker_merge,
        )
        self._translator_cache[key] = translator
        return translator

    def translate_wav(self, wav_path: str, source_lang: str, target_lang: str):
        translator = self._get_translator(source_lang, target_lang)
        return translator.translate_audio_file_no_playback(wav_path)

    @modal.asgi_app()
    def fastapi_app(self):
        # Important: this runs *inside* the Modal container.
        from api_server import make_app
        return make_app(self)

