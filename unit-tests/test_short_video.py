# test_short_video.py
import os
from dotenv import load_dotenv
from diarizer import SpeakerDiarizer
from moviepy.editor import VideoFileClip
import tempfile

load_dotenv()

# Extract audio
video = VideoFileClip("sample-videos/short.mp4")
temp_audio = "temp_test_audio.wav"
video.audio.write_audiofile(temp_audio, codec='pcm_s16le', fps=16000, logger=None)

# Diarize
hf_token = os.getenv("HUGGING_FACE_TOKEN")
diarizer = SpeakerDiarizer(hf_token)
segments = diarizer.diarize(temp_audio)

# Analyze results
print("\n" + "="*60)
print("DIARIZATION ANALYSIS")
print("="*60)
print(f"Total segments: {len(segments)}")
print(f"Speakers: {set(s.speaker_id for s in segments)}")
print("\nSegment details:")
for i, seg in enumerate(segments, 1):
    duration = seg.duration_ms / 1000.0
    status = "✅" if duration >= 1.5 else "⚠️"
    print(f"{status} [{i}] {seg.speaker_id}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s ({duration:.2f}s)")

os.remove(temp_audio)