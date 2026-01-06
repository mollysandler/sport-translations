#!/bin/bash
# install_hybrid.sh
# Step-by-step installation script for hybrid translation system

echo "=================================================="
echo "Hybrid Translation System - Installation Script"
echo "=================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

# Upgrade pip first
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install in stages to avoid conflicts
echo ""
echo "=================================================="
echo "Stage 1: Core Dependencies"
echo "=================================================="
pip install numpy==1.26.4
pip install scipy==1.13.1

echo ""
echo "=================================================="
echo "Stage 2: PyTorch (this may take a few minutes)"
echo "=================================================="
# For Mac
pip install torch==2.4.0 torchaudio==2.4.0

# If the above fails, try:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=================================================="
echo "Stage 3: Audio Processing"
echo "=================================================="
pip install soundfile==0.12.1
pip install audioread==3.0.1
pip install soxr==0.3.7
pip install librosa==0.10.2.post1
pip install pydub==0.25.1

echo ""
echo "=================================================="
echo "Stage 4: ML Models"
echo "=================================================="
pip install transformers==4.45.2
pip install sentencepiece==0.2.0
pip install tokenizers==0.20.3
pip install huggingface-hub==0.25.2
pip install accelerate==0.34.2

echo ""
echo "=================================================="
echo "Stage 5: Whisper"
echo "=================================================="
pip install faster-whisper==1.0.3

echo ""
echo "=================================================="
echo "Stage 6: TTS (this will download ~2GB of models)"
echo "=================================================="
pip install TTS==0.22.0

echo ""
echo "=================================================="
echo "Stage 7: Speaker Recognition"
echo "=================================================="
pip install speechbrain==1.0.0

echo ""
echo "=================================================="
echo "Stage 8: Diarization"
echo "=================================================="
pip install pyannote.audio==3.3.2

echo ""
echo "=================================================="
echo "Stage 9: Video Processing"
echo "=================================================="
pip install imageio==2.35.1
pip install imageio-ffmpeg==0.5.1
pip install moviepy==1.0.3

echo ""
echo "=================================================="
echo "Stage 10: Utilities"
echo "=================================================="
pip install python-dotenv==1.0.1
pip install protobuf==4.25.5
pip install sounddevice==0.5.1

echo ""
echo "=================================================="
echo "Stage 11: Optional - Google Cloud"
echo "=================================================="
read -p "Install Google Cloud Translate? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install google-cloud-translate==3.17.0
fi

echo ""
echo "=================================================="
echo "✅ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Create .env file with your HUGGING_FACE_TOKEN"
echo "2. Accept PyAnnote licenses (see setup guide)"
echo "3. Test with: python hybrid_system.py test_video.mp4"
echo ""