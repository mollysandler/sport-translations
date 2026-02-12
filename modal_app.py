import modal

app = modal.App("sports-translation-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # you use pydub / audio ops
    .pip_install(
        "fastapi[standard]",
        "torch",
        "torchaudio",
        "pydub",
        "librosa",
        "numpy",
        "scipy",
        "python-dotenv",
        "faster-whisper",
        "pyannote.audio",
        "huggingface_hub",
        "deep-translator",
        "elevenlabs",
    )
)

secrets = [modal.Secret.from_name("sports-secrets")]

@app.function(
    image=image,
    gpu="A10G",          # good price/perf for whisper + pyannote
    cpu=4,
    memory=16384,
    secrets=secrets,
    timeout=60 * 60,     # large files
)
@modal.asgi_app()
def fastapi_app():
    from api_server import app as api
    return api
