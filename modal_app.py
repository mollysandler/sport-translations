import modal

app = modal.App("sports-translation-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
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
    # Explicitly ship your local python modules to the container:
    .add_local_python_source("api_server", "main", "diarizer", "utils")
)

secrets = [modal.Secret.from_name("sports-secrets")]

@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    secrets=secrets,
)
@modal.asgi_app()
def fastapi_app():
    from api_server import app as api
    return api
