# translator.py
import io
import torch
import torchaudio
from typing import Dict, Optional
from transformers import SeamlessM4Tv2ForSpeechToSpeech, AutoProcessor
import numpy as np

# IMPORT SEAMLESS_MODEL HERE
from config import DEVICE, SAMPLE_RATE, SEAMLESS_MODEL 

class SeamlessSpeechTranslator:
    def __init__(self, source_language: str, target_language: str):
        self.source_language = source_language
        self.target_language = target_language
        
        print(f"🔄 Initializing SeamlessM4T translator ({source_language} → {target_language})...")
        
        # FIX: Use the model defined in config.py, not hardcoded large model
        self.processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL)
        self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(SEAMLESS_MODEL)
        
        device = torch.device(DEVICE)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        
        self.speaker_voice_map = {}
        self.speaker_count = 0
        
        print(f"   Model loaded on: {device}")
        print("✅ Translator initialized successfully")
    
    def _assign_speaker_id(self, speaker_id: str) -> int:
        if speaker_id not in self.speaker_voice_map:
            self.speaker_voice_map[speaker_id] = self.speaker_count
            self.speaker_count += 1
            print(f"   🎤 Assigned ID {self.speaker_voice_map[speaker_id]} to {speaker_id}")
        return self.speaker_voice_map[speaker_id]
    
    def translate_segment(self, audio_chunk_data: bytes, speaker_id: str) -> Optional[Dict]:
        try:
            # Load audio using torchaudio
            audio_buffer = io.BytesIO(audio_chunk_data)
            waveform, sample_rate = torchaudio.load(audio_buffer)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to numpy for processor
            audio_array = waveform.squeeze().numpy()
            
            speaker_id_num = self._assign_speaker_id(speaker_id)
            
            # Process audio
            inputs = self.processor(
                audio=audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                audio_array_from_audio = self.model.generate(
                    **inputs,
                    tgt_lang=self.target_language,
                    speaker_id=speaker_id_num % 6,
                )[0].cpu()
            
            # Clear cache
            if self.device.type == "mps":
                torch.mps.empty_cache()
            
            # FIX: Squeeze the array to remove batch/channel dimensions
            # Converts shape (1, N) -> (N,)
            audio_np = audio_array_from_audio.numpy().squeeze()
            
            # Normalize
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val * 0.9
            
            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Save as WAV
            output_buffer = io.BytesIO()
            import scipy.io.wavfile as wavfile
            
            # Write explicitly as mono
            wavfile.write(output_buffer, 16000, audio_int16)
            output_buffer.seek(0)
            
            return {
                'audio_data': output_buffer.read(),
                'translated_text': "",
                'speaker_id': speaker_id
            }
                
        except Exception as e:
            print(f"   ❌ [{speaker_id}] Translation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_speaker_count(self) -> int:
        return len(self.speaker_voice_map)