# app.py

import os
import io
import tempfile
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv
load_dotenv()

from translator import SpeechTranslator
from config import TARGET_LANGUAGE, STT_SAMPLE_RATE

# 1. --- SETUP FLASK APP ---
app = Flask(__name__)
# Allow requests from your React app's origin (e.g., http://localhost:5173)
CORS(app)

# 2. --- LOAD MODELS ONCE AT STARTUP ---
print("LOADING: Speaker diarization pipeline...")
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN not found in .env file.")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
pipeline.to(device)
print("✅ Diarization pipeline loaded.")

translator = SpeechTranslator(target_language=TARGET_LANGUAGE)
print("✅ SpeechTranslator initialized.")


# 3. --- CREATE THE API ENDPOINT ---
@app.route('/translate-audio', methods=['POST'])
def translate_audio_endpoint():
    # Check if an audio file was sent
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir = tempfile.mkdtemp()
    try:
        temp_audio_path = os.path.join(temp_dir, "uploaded_audio.wav")
        audio = AudioSegment.from_file(file)

        audio.export(temp_audio_path,
                    format="wav",
                    parameters=["-ar", str(STT_SAMPLE_RATE), "-ac", "1"])

        # --- THIS IS YOUR CORE LOGIC FROM main.py ---
        print("DIARIZING: Identifying speakers in the audio...")
        diarization = pipeline(temp_audio_path)
        print("✅ Diarization complete.")

        full_original_audio = AudioSegment.from_file(temp_audio_path)
        final_composition = AudioSegment.silent(duration=len(full_original_audio))
        
        commentary_data = [] # To store text commentary for the frontend

        print("\n--- Starting translation of speaker segments ---")
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            
            print(f"\nProcessing segment for {speaker} from {start_ms}ms to {end_ms}ms")
            
            segment_audio = full_original_audio[start_ms:end_ms]
            segment_wav_data = segment_audio.export(format="wav").read()
            
            # We assume translate_audio_chunk now returns both audio and text
            # You might need to adjust your SpeechTranslator for this
            result = translator.translate_audio_chunk(
                audio_chunk_data=segment_wav_data,
                speaker_id=speaker
            )

            # Let's assume the translator returns a dictionary:
            # { "audio_data": <mp3_bytes>, "original_text": "...", "translated_text": "..." }
            # For now, we'll mock the text part if it's not there.
            
            translated_audio_data = result.get("audio_data") if isinstance(result, dict) else result
            
            if translated_audio_data:
                translated_segment = AudioSegment.from_file(io.BytesIO(translated_audio_data), format="mp3")
                final_composition = final_composition.overlay(translated_segment, position=start_ms)
                
                commentary_data.append({
                    "time": turn.start,
                    "original": result.get("original_text", "Original text..."),
                    "translated": result.get("translated_text", "Translated text...")
                })


        print("\n✅ All segments processed.")
        
        # Instead of playing the file, we send it back
        output_buffer = io.BytesIO()
        final_composition.export(output_buffer, format="mp3")
        output_buffer.seek(0)
        
        # For now, we'll just send the audio. In a real app you might send JSON with a URL to the audio.
        return send_file(
            output_buffer,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="translated_commentary.mp3"
        )
        # --- END OF CORE LOGIC ---

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        print("🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

# To run the server
if __name__ == '__main__':
    app.run(debug=True, port=5000)