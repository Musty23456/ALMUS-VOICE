import io
import math
import struct
import wave


def make_wav(seconds: float = 6.0, rate: int = 16000, channels: int = 1) -> bytes:
    """Build a small valid PCM WAV (quiet tone) for tests."""
    frames = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        data = bytearray()
        for i in range(frames):
            sample = int(3000 * math.sin(2 * math.pi * 220 * i / rate))
            data += struct.pack("<h", sample) * channels
        w.writeframes(bytes(data))
    return buf.getvalue()
