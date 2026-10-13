"""Optional real-model test. Skipped unless everything is available.

Run locally:
    export ALMUS_XTTS_MODEL_DIR=/path/to/XTTS-v2
    export ALMUS_REFERENCE_WAV=/path/to/consented_reference.wav
    pytest -m integration -v
"""

import io
import os
import wave
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

MODEL_DIR = os.environ.get("ALMUS_XTTS_MODEL_DIR")
REFERENCE = os.environ.get("ALMUS_REFERENCE_WAV")

if not MODEL_DIR or not REFERENCE:
    pytest.skip("ALMUS_XTTS_MODEL_DIR / ALMUS_REFERENCE_WAV not set", allow_module_level=True)
pytest.importorskip("torch")
pytest.importorskip("TTS")

from app.services.xtts_engine import XttsEngine  # noqa: E402


def test_real_xtts_generates_audio() -> None:
    engine = XttsEngine(MODEL_DIR, os.environ.get("ALMUS_XTTS_DEVICE", "auto"))
    assert engine.is_configured
    result = engine.synthesize(
        "Hello, welcome to Almus AI.", "en", "integration", reference_audio=Path(REFERENCE)
    )
    with wave.open(io.BytesIO(result.data), "rb") as w:
        assert w.getnframes() / w.getframerate() > 0.5
    # Passing proves only that valid audio was produced. It does NOT prove the voice
    # resembles the speaker: that needs human listening evaluation.
