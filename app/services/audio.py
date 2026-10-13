"""WAV encoding helper (standard library only)."""

import io
import math
import sys
import wave
from array import array
from typing import Any


class AudioEncodingError(Exception):
    """Raised when model output cannot be turned into valid audio."""


def encode_wav_pcm16(samples: Any, sample_rate: int) -> bytes:
    """Encode mono float samples in [-1, 1] as a 16-bit PCM WAV file.

    Refuses empty, non-finite or entirely silent output so a failed inference can
    never be returned as a "successful" audio file.
    """
    if hasattr(samples, "tolist"):
        samples = samples.tolist()
    if not samples:
        raise AudioEncodingError("The engine produced no audio samples.")
    if isinstance(samples[0], (list, tuple)):
        raise AudioEncodingError("Expected mono audio samples.")

    pcm = array("h")
    peak = 0.0
    for raw in samples:
        value = float(raw)
        if not math.isfinite(value):
            raise AudioEncodingError("The engine produced invalid (non-finite) audio.")
        value = max(-1.0, min(1.0, value))
        peak = max(peak, abs(value))
        pcm.append(int(value * 32767))
    if peak < 1e-4:
        raise AudioEncodingError("The engine produced silent audio.")
    if sys.byteorder == "big":
        pcm.byteswap()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()
