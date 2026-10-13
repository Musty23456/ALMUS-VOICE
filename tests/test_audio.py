import io
import wave

import pytest

from app.services.audio import AudioEncodingError, encode_wav_pcm16


def test_encodes_valid_wav() -> None:
    data = encode_wav_pcm16([0.0, 0.5, -0.5, 0.25] * 100, 24000)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 400


@pytest.mark.parametrize(
    "samples",
    [[], [0.0] * 100, [0.5, float("nan")], [0.5, float("inf")], [[0.1, 0.2]]],
)
def test_rejects_bad_audio(samples) -> None:
    with pytest.raises(AudioEncodingError):
        encode_wav_pcm16(samples, 24000)
