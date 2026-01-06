import os
import io
import tempfile
import shutil
import base64 
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv
load_dotenv()

from v3.translator import SpeechTranslator
# --- CHANGE 1: Update Imports to match new config.py ---
from hybrid_config import STT_SAMPLE_RATE, LANGUAGE_MAPPING, DEFAULT_SOURCE, DEFAULT_TARGET

# 1. --- SETUP FLASK APP ---
app = Flask(__name__)
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

# 3. --- CREATE THE API ENDPOINT ---
@app.route('/translate-audio', methods=['POST'])
def translate_audio_endpoint():
    # Check if an audio file was sent
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    file = request.files['audio']
    
    raw_source = request.form.get('source_lang', 'en')
    raw_target = request.form.get('target_lang', 'hi')

    # Map to BCP-47 codes (e.g., "en" -> "en-US")
    source_lang = LANGUAGE_MAPPING.get(raw_source, DEFAULT_SOURCE)
    target_lang = LANGUAGE_MAPPING.get(raw_target, DEFAULT_TARGET)

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    print(f"🌍 Request: Translating {source_lang} -> {target_lang}")

    temp_dir = tempfile.mkdtemp()
    try:
        temp_audio_path = os.path.join(temp_dir, "uploaded_audio.wav")
        audio = AudioSegment.from_file(file)

        audio.export(temp_audio_path,
                    format="wav",
                    parameters=["-ar", str(STT_SAMPLE_RATE), "-ac", "1"])

        
        translator = SpeechTranslator(source_language=source_lang, target_language=target_lang)

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

            # SKIPPING SHORT SEGMENTS, COULD BE A PROBLEM LATER
            if (end_ms - start_ms) < 500:
                print(f"   » Skipping short segment for {speaker} ({end_ms - start_ms}ms)")
                continue
            
            # Extract segment
            segment_audio = full_original_audio[start_ms:end_ms]
            segment_wav_data = segment_audio.export(format="wav").read()
            
            # Translate
            result = translator.translate_audio_chunk(
                audio_chunk_data=segment_wav_data,
                speaker_id=speaker
            )

            if result and result.get("audio_data"):
                # Overlay audio
                translated_segment = AudioSegment.from_file(io.BytesIO(result["audio_data"]), format="mp3")
                final_composition = final_composition.overlay(translated_segment, position=start_ms)
                
                # Append to commentary data list
                # WE KEEP TIMESTAMPS IN SECONDS FOR THE FRONTEND
                commentary_data.append({
                    "startTime": turn.start, 
                    "endTime": turn.end,
                    "speaker": speaker,
                    "original": result.get("original_text", ""),
                    "translated": result.get("translated_text", "")
                })

        print("\n✅ All segments processed.")
        
        # Export final audio to bytes
        output_buffer = io.BytesIO()
        final_composition.export(output_buffer, format="mp3")
        output_buffer.seek(0)
        
        # Encode audio to Base64 string
        audio_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')
        
        # Return JSON containing BOTH audio and captions
        return jsonify({
            "audio_base64": audio_base64,
            "captions": commentary_data
        })

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