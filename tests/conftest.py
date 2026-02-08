# tests/conftest.py
import sys
import types
from dataclasses import dataclass

# --- Provide a lightweight fake "diarizer" module so importing main/api_server works
fake_diarizer = types.ModuleType("diarizer")

@dataclass
class SpeakerSegment:
    speaker_id: str
    start_ms: int
    end_ms: int
    start_sec: float
    end_sec: float

class SpeakerDiarizer:
    """Test stub. Real implementation lives in diarizer.py."""
    pass

fake_diarizer.SpeakerSegment = SpeakerSegment
fake_diarizer.SpeakerDiarizer = SpeakerDiarizer

# Only install the stub if the real module isn't already imported
sys.modules.setdefault("diarizer", fake_diarizer)


# --- Prevent heavy model/service init when constructing DynamicSpeakerTranslator in tests
import pytest

@pytest.fixture(autouse=True)
def _disable_heavy_initialization(monkeypatch):
    import main
    monkeypatch.setattr(main.DynamicSpeakerTranslator, "_initialize_services", lambda self: None)
