# v3_test.py
# Comprehensive testing suite for the translation system

import os
import sys
import io
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import Sine

from config import SAMPLE_RATE, SEAMLESS_LANGUAGE_MAPPING
from translator import SeamlessSpeechTranslator
from diarizer import SpeakerDiarizer

load_dotenv()


class TestAudioGenerator:
    """Generate synthetic test audio files."""
    
    @staticmethod
    def generate_test_audio(
        duration_ms: int = 3000,
        frequency: int = 440,
        output_path: str = "test_audio.wav"
    ) -> str:
        """
        Generate a simple sine wave audio file for testing.
        
        Args:
            duration_ms: Duration in milliseconds
            frequency: Frequency in Hz
            output_path: Path to save the audio file
            
        Returns:
            Path to the generated audio file
        """
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        tone.export(output_path, format="wav")
        return output_path
    
    @staticmethod
    def generate_multi_speaker_audio(
        output_path: str = "test_multi_speaker.wav",
        speakers: int = 2
    ) -> str:
        """
        Generate audio with multiple speakers (different frequencies).
        
        Args:
            output_path: Path to save the audio file
            speakers: Number of speakers to simulate
            
        Returns:
            Path to the generated audio file
        """
        frequencies = [440, 880, 1320, 1760]  # A4, A5, E6, A6
        
        combined = AudioSegment.silent(duration=0)
        
        for i in range(speakers):
            freq = frequencies[i % len(frequencies)]
            segment = Sine(freq).to_audio_segment(duration=2000)
            combined += segment
            
            # Add short pause between speakers
            if i < speakers - 1:
                combined += AudioSegment.silent(duration=500)
        
        combined.export(output_path, format="wav")
        return output_path


class TranslationTester:
    """Test suite for the translation system."""
    
    def __init__(self):
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGING_FACE_TOKEN not found in .env file")
        
        self.test_results = []
    
    def test_translator_initialization(self):
        """Test 1: Can we initialize the translator?"""
        print("\n" + "="*60)
        print("TEST 1: Translator Initialization")
        print("="*60)
        
        try:
            start_time = time.time()
            translator = SeamlessSpeechTranslator("eng", "hin")
            elapsed = time.time() - start_time
            
            print(f"✅ PASS: Translator initialized in {elapsed:.2f}s")
            self.test_results.append(("Initialization", True, elapsed))
            return translator
        except Exception as e:
            print(f"❌ FAIL: {e}")
            self.test_results.append(("Initialization", False, 0))
            import traceback
            traceback.print_exc()
            return None
    
    def test_diarizer_initialization(self):
        """Test 2: Can we initialize the diarizer?"""
        print("\n" + "="*60)
        print("TEST 2: Diarizer Initialization")
        print("="*60)
        
        try:
            start_time = time.time()
            diarizer = SpeakerDiarizer(self.hf_token)
            elapsed = time.time() - start_time
            
            print(f"✅ PASS: Diarizer initialized in {elapsed:.2f}s")
            self.test_results.append(("Diarizer Init", True, elapsed))
            return diarizer
        except Exception as e:
            print(f"❌ FAIL: {e}")
            self.test_results.append(("Diarizer Init", False, 0))
            import traceback
            traceback.print_exc()
            return None
    
    def test_simple_translation(self, translator):
        """Test 3: Can we translate a simple audio segment?"""
        print("\n" + "="*60)
        print("TEST 3: Simple Translation")
        print("="*60)
        
        if not translator:
            print("⚠️ SKIP: No translator available")
            return False
        
        try:
            # Generate test audio
            print("Generating test audio...")
            test_audio_path = TestAudioGenerator.generate_test_audio(
                duration_ms=2000,
                output_path="test_simple.wav"
            )
            
            # Load as bytes
            with open(test_audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Translate
            print("Translating audio segment...")
            start_time = time.time()
            result = translator.translate_segment(
                audio_chunk_data=audio_bytes,
                speaker_id="TEST_SPEAKER"
            )
            elapsed = time.time() - start_time
            
            # Check result
            if result and result.get('audio_data'):
                print(f"✅ PASS: Translation completed in {elapsed:.2f}s")
                print(f"   Translated text: {result.get('translated_text', 'N/A')}")
                self.test_results.append(("Simple Translation", True, elapsed))
                
                # Save output for manual inspection
                with open("test_output_simple.wav", 'wb') as f:
                    f.write(result['audio_data'])
                print("   Saved output to: test_output_simple.wav")
                
                return True
            else:
                print("❌ FAIL: No audio data in result")
                self.test_results.append(("Simple Translation", False, elapsed))
                return False
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            self.test_results.append(("Simple Translation", False, 0))
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            if os.path.exists("test_simple.wav"):
                os.remove("test_simple.wav")
    
    def test_voice_consistency(self, translator):
        """Test 4: Do the same speakers get consistent voices?"""
        print("\n" + "="*60)
        print("TEST 4: Voice Consistency")
        print("="*60)
        
        if not translator:
            print("⚠️ SKIP: No translator available")
            return False
        
        try:
            # Generate test audio
            test_audio_path = TestAudioGenerator.generate_test_audio(
                duration_ms=1500,
                output_path="test_consistency.wav"
            )
            
            with open(test_audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Translate same audio with same speaker twice
            print("Translating segment 1 for SPEAKER_A...")
            result1 = translator.translate_segment(audio_bytes, "SPEAKER_A")
            
            print("Translating segment 2 for SPEAKER_A...")
            result2 = translator.translate_segment(audio_bytes, "SPEAKER_A")
            
            # Translate with different speaker
            print("Translating segment 3 for SPEAKER_B...")
            result3 = translator.translate_segment(audio_bytes, "SPEAKER_B")
            
            # Check all succeeded
            if not (result1 and result2 and result3):
                print("❌ FAIL: One or more translations failed")
                self.test_results.append(("Voice Consistency", False, 0))
                return False
            
            # Check voice seeds
            seed_a = translator.speaker_voice_map.get("SPEAKER_A")
            seed_b = translator.speaker_voice_map.get("SPEAKER_B")
            
            print(f"\nVoice seeds:")
            print(f"   SPEAKER_A: {seed_a}")
            print(f"   SPEAKER_B: {seed_b}")
            
            if seed_a == seed_b:
                print("❌ FAIL: Different speakers got same voice seed")
                self.test_results.append(("Voice Consistency", False, 0))
                return False
            
            # Manual inspection needed
            print("\n✅ PASS: Voice seeds are consistent per speaker")
            print("   Manual verification needed: Listen to outputs")
            
            # Save outputs for manual inspection
            with open("test_speaker_a1.wav", 'wb') as f:
                f.write(result1['audio_data'])
            with open("test_speaker_a2.wav", 'wb') as f:
                f.write(result2['audio_data'])
            with open("test_speaker_b.wav", 'wb') as f:
                f.write(result3['audio_data'])
            
            print("   Saved: test_speaker_a1.wav, test_speaker_a2.wav, test_speaker_b.wav")
            
            self.test_results.append(("Voice Consistency", True, 0))
            return True
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            self.test_results.append(("Voice Consistency", False, 0))
            import traceback
            traceback.print_exc()
            return False
        finally:
            if os.path.exists("test_consistency.wav"):
                os.remove("test_consistency.wav")
    
    def test_diarization(self, diarizer):
        """Test 5: Can we diarize a multi-speaker audio?"""
        print("\n" + "="*60)
        print("TEST 5: Speaker Diarization")
        print("="*60)
        
        if not diarizer:
            print("⚠️ SKIP: No diarizer available")
            return False
        
        try:
            # Generate multi-speaker test audio
            print("Generating multi-speaker test audio...")
            test_audio_path = TestAudioGenerator.generate_multi_speaker_audio(
                output_path="test_diarization.wav",
                speakers=2
            )
            
            # Diarize
            print("Running diarization...")
            start_time = time.time()
            segments = diarizer.diarize(test_audio_path)
            elapsed = time.time() - start_time
            
            # Check results
            if not segments:
                print("❌ FAIL: No segments found")
                self.test_results.append(("Diarization", False, elapsed))
                return False
            
            print(f"\n✅ PASS: Found {len(segments)} segments in {elapsed:.2f}s")
            
            # Show segments
            print("\nSegments:")
            for seg in segments:
                print(f"   {seg.speaker_id}: {seg.start_sec:.2f}s - {seg.end_sec:.2f}s")
            
            self.test_results.append(("Diarization", True, elapsed))
            return True
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            self.test_results.append(("Diarization", False, 0))
            import traceback
            traceback.print_exc()
            return False
        finally:
            if os.path.exists("test_diarization.wav"):
                os.remove("test_diarization.wav")
    
    def test_language_support(self, translator):
        """Test 6: Can we translate to different languages?"""
        print("\n" + "="*60)
        print("TEST 6: Multi-Language Support")
        print("="*60)
        
        test_languages = ["spa", "fra", "deu"]  # Spanish, French, German
        
        try:
            test_audio_path = TestAudioGenerator.generate_test_audio(
                duration_ms=1500,
                output_path="test_multilang.wav"
            )
            
            with open(test_audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            success_count = 0
            for lang in test_languages:
                print(f"\nTesting translation to {lang}...")
                
                try:
                    lang_translator = SeamlessSpeechTranslator("eng", lang)
                    result = lang_translator.translate_segment(
                        audio_bytes,
                        "TEST_SPEAKER"
                    )
                    
                    if result and result.get('audio_data'):
                        print(f"   ✅ {lang}: Success")
                        success_count += 1
                    else:
                        print(f"   ❌ {lang}: Failed")
                        
                except Exception as e:
                    print(f"   ❌ {lang}: Error - {e}")
            
            if success_count == len(test_languages):
                print(f"\n✅ PASS: All {len(test_languages)} languages worked")
                self.test_results.append(("Multi-Language", True, 0))
                return True
            else:
                print(f"\n⚠️ PARTIAL: {success_count}/{len(test_languages)} languages worked")
                self.test_results.append(("Multi-Language", False, 0))
                return False
                
        except Exception as e:
            print(f"❌ FAIL: {e}")
            self.test_results.append(("Multi-Language", False, 0))
            import traceback
            traceback.print_exc()
            return False
        finally:
            if os.path.exists("test_multilang.wav"):
                os.remove("test_multilang.wav")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        print(f"\nResults: {passed}/{total} tests passed\n")
        
        for test_name, success, elapsed in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            time_str = f"({elapsed:.2f}s)" if elapsed > 0 else ""
            print(f"  {status} {test_name} {time_str}")
        
        print("\n" + "="*60)
        
        if passed == total:
            print("🎉 All tests passed!")
        elif passed > 0:
            print(f"⚠️ {total - passed} test(s) failed")
        else:
            print("❌ All tests failed")
        
        print("="*60 + "\n")

    def test_with_real_video(self):
        """Test with actual video file."""
        print("\n" + "="*60)
        print("TEST 7: Real Video Translation")
        print("="*60)
        
        if not os.path.exists("short.mp4"):
            print("⚠️ SKIP: short.mp4 not found")
            return False
        
        try:
            from main import translate_video
            translate_video("short.mp4", "eng", "hin", play_output=False)
            print("✅ PASS: Real video translated")
            return True
        except Exception as e:
            print(f"❌ FAIL: {e}")
            return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 TRANSLATION SYSTEM TEST SUITE")
    print("="*60)
    print("\nThis will test all components of the translation system.")
    print("Some tests generate audio files for manual verification.\n")
    
    input("Press Enter to begin tests...")
    
    tester = TranslationTester()
    
    # Run tests
    translator = tester.test_translator_initialization()
    diarizer = tester.test_diarizer_initialization()
    
    tester.test_simple_translation(translator)
    tester.test_voice_consistency(translator)
    tester.test_diarization(diarizer)
    tester.test_language_support(translator)
    tester.test_with_real_video()
    
    # Print summary
    tester.print_summary()
    
    # Cleanup test files
    print("🧹 Cleaning up test files...")
    test_files = [
        "test_output_simple.wav",
        "test_speaker_a1.wav",
        "test_speaker_a2.wav",
        "test_speaker_b.wav"
    ]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
    
    print("✅ Cleanup complete\n")


if __name__ == "__main__":
    main()