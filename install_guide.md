# Installation Fix Guide

The dependency resolution issue you're experiencing is common with complex ML packages. Here's how to fix it:

## Solution 1: Install in Stages (Recommended)

This avoids the dependency resolution problem by installing in the correct order:

### Step 1: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 2: Core dependencies

```bash
pip install numpy==1.26.4 scipy==1.13.1
```

### Step 3: PyTorch (Mac M1/M2)

```bash
pip install torch==2.4.0 torchaudio==2.4.0
```

**If you have Intel Mac or Linux:**

```bash
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Audio processing

```bash
pip install soundfile==0.12.1 audioread==3.0.1 pydub==0.25.1
```

### Step 5: Librosa (often problematic)

```bash
pip install librosa==0.10.2.post1
```

### Step 6: HuggingFace ecosystem

```bash
pip install transformers==4.45.2 sentencepiece==0.2.0 huggingface-hub==0.25.2 accelerate==0.34.2
```

### Step 7: Faster Whisper

```bash
pip install faster-whisper==1.0.3
```

### Step 8: Coqui TTS (large download ~2GB)

```bash
pip install TTS==0.22.0
```

### Step 9: SpeechBrain

```bash
pip install speechbrain==1.0.0
```

### Step 10: PyAnnote

```bash
pip install pyannote.audio==3.3.2
```

### Step 11: Video processing

```bash
pip install moviepy==1.0.3 imageio==2.35.1 imageio-ffmpeg==0.5.1
```

### Step 12: Utilities

```bash
pip install python-dotenv==1.0.1
```

### Step 13: (Optional) Google Cloud

```bash
pip install google-cloud-translate==3.17.0
```

---

## Solution 2: Use the Install Script

Save the install script I created as `install_hybrid.sh` and run:

```bash
chmod +x install_hybrid.sh
./install_hybrid.sh
```

---

## Solution 3: Minimal Installation (Start Simple)

If you want to test the system quickly, install only the essentials:

```bash
# Upgrade pip
pip install --upgrade pip

# Essential only
pip install torch torchaudio
pip install faster-whisper
pip install transformers sentencepiece
pip install TTS
pip install pyannote.audio
pip install moviepy pydub
pip install python-dotenv
```

Then install others as needed when you encounter import errors.

---

## Solution 4: Create Fresh Environment

If all else fails, start completely fresh:

```bash
# Remove old environment
rm -rf venv4

# Create new environment with Python 3.10 (more stable)
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Then follow Solution 1 steps
```

---

## Common Issues & Fixes

### Issue: "No matching distribution found for torch"

**Fix**: Install PyTorch separately first:

```bash
# For Mac M1/M2
pip install torch torchaudio

# For Intel/Linux
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "librosa requires numba"

**Fix**: Install numba first:

```bash
pip install numba==0.59.1
pip install librosa==0.10.2.post1
```

### Issue: "pyannote.audio requires torch>=2.0"

**Fix**: Ensure PyTorch is installed first (see Step 3 above)

### Issue: "TTS requires espeak-ng"

**Fix** (Mac):

```bash
brew install espeak-ng
```

**Fix** (Ubuntu):

```bash
sudo apt-get install espeak-ng
```

### Issue: "moviepy requires imageio-ffmpeg"

**Fix**: Install imageio-ffmpeg first:

```bash
pip install imageio-ffmpeg==0.5.1
pip install moviepy==1.0.3
```

---

## Verification

After installation, verify everything works:

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import faster_whisper; print('Faster-Whisper: OK')"
python3 -c "import TTS; print('TTS: OK')"
python3 -c "import pyannote.audio; print('PyAnnote: OK')"
python3 -c "import speechbrain; print('SpeechBrain: OK')"
```

---

## Troubleshooting Checklist

- [ ] Python 3.8-3.11 (not 3.12)
- [ ] pip upgraded to latest version
- [ ] Virtual environment activated
- [ ] FFmpeg installed system-wide
- [ ] Enough disk space (~5GB for models)
- [ ] Stable internet connection (for model downloads)

---

## Alternative: Use Docker (Advanced)

If you continue having issues, I can provide a Dockerfile that pre-configures everything:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements_hybrid.txt .
RUN pip install --no-cache-dir -r requirements_hybrid.txt

# Your app
WORKDIR /app
COPY . .

CMD ["python", "hybrid_system.py"]
```

Let me know if you need the full Docker setup!

---

## Quick Test

Once installed, test with this simple script:

```python
# test_install.py
import sys

packages = {
    'torch': 'PyTorch',
    'faster_whisper': 'Faster-Whisper',
    'TTS': 'Coqui TTS',
    'pyannote.audio': 'PyAnnote',
    'speechbrain': 'SpeechBrain',
    'transformers': 'Transformers',
    'moviepy.editor': 'MoviePy',
}

print("Checking installations...\n")
failed = []

for module, name in packages.items():
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError as e:
        print(f"❌ {name}: {e}")
        failed.append(name)

if failed:
    print(f"\n⚠️  Failed: {', '.join(failed)}")
    print("Install missing packages and try again.")
else:
    print("\n🎉 All packages installed successfully!")
```

Run: `python test_install.py`
