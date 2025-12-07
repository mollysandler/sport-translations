# test_microphone.py
import pyaudio
import time

# Use the same audio configuration as your main app
RATE = 16000
CHUNK = int(RATE / 10)
FORMAT = pyaudio.paInt16
CHANNELS = 1

print("--- Microphone Test ---")
print("This test will try to capture audio for 3 seconds.")

try:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("✅ Microphone stream opened successfully.")
    print("Capturing audio...")

    # Read from the stream for a few seconds
    for i in range(0, int(RATE / CHUNK * 3)):
        data = stream.read(CHUNK)
        print(f"Read chunk {i+1}...", end='\r')
    
    print("\n✅ Audio captured successfully for 3 seconds.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("✅ Microphone stream closed properly.")
    print("\n--- Test Passed! ---")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("--- Test Failed! ---")
    print("Troubleshooting: Ensure your microphone is connected and not in use by another app.")
    print("On macOS, make sure your terminal has microphone permissions in System Settings > Privacy & Security.")