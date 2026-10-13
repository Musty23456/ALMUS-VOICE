"""XttsEngine tests with a mocked model. No torch, no downloads, no GPU."""

import io
import wave

import pytest

from app.services.voice_service import VoiceEngineError, VoiceEngineLoadError
from app.services.xtts_engine import REQUIRED_FILES, XTTS_LANGUAGES, XttsEngine
from tests.helpers import make_wav


class FakeModel:
    def __init__(self, wav=None, fail=False) -> None:
        self.cond_calls = 0
        self.infer_calls = 0
        self._wav = wav if wav is not None else [0.1, -0.2, 0.3] * 200
        self._fail = fail

    def get_conditioning_latents(self, audio_path, **kwargs):
        self.cond_calls += 1
        return ("gpt", "spk")

    def inference(self, text, language, gpt_cond_latent, speaker_embedding, **kwargs):
        self.infer_calls += 1
        if self._fail:
            raise RuntimeError("boom with secret text " + text)
        return {"wav": self._wav}


@pytest.fixture
def model_dir(tmp_path):
    for name in REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


@pytest.fixture
def ref(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(make_wav(6))
    return p


def make_engine(model_dir, model, loads=None):
    def loader(path, device):
        if loads is not None:
            loads.append(device)
        return model

    return XttsEngine(str(model_dir), "cpu", model_loader=loader)


def test_not_configured_without_files(tmp_path) -> None:
    assert XttsEngine("", "cpu").is_configured is False
    assert XttsEngine(str(tmp_path), "cpu", model_loader=lambda *a: None).is_configured is False


def test_languages_exclude_hausa() -> None:
    assert "en" in XTTS_LANGUAGES and "ha" not in XTTS_LANGUAGES


def test_synthesize_returns_valid_wav(model_dir, ref) -> None:
    engine = make_engine(model_dir, FakeModel())
    result = engine.synthesize("Hello", "en", "v", reference_audio=ref)
    assert result.media_type == "audio/wav"
    with wave.open(io.BytesIO(result.data), "rb") as w:
        assert w.getframerate() == 24000


def test_model_loaded_once_and_latents_cached(model_dir, ref) -> None:
    model, loads = FakeModel(), []
    engine = make_engine(model_dir, model, loads)
    engine.synthesize("one", "en", "v", reference_audio=ref)
    engine.synthesize("two", "en", "v", reference_audio=ref)
    assert loads == ["cpu"]
    assert model.cond_calls == 1
    assert model.infer_calls == 2


def test_requires_reference(model_dir) -> None:
    engine = make_engine(model_dir, FakeModel())
    with pytest.raises(VoiceEngineError):
        engine.synthesize("Hello", "en", "v")


def test_inference_failure_does_not_leak_text(model_dir, ref) -> None:
    engine = make_engine(model_dir, FakeModel(fail=True))
    with pytest.raises(VoiceEngineError) as err:
        engine.synthesize("my private sentence", "en", "v", reference_audio=ref)
    assert "private" not in str(err.value)


@pytest.mark.parametrize("bad", [[], [0.0] * 50, [float("nan")] * 50])
def test_bad_model_output_is_an_error_not_audio(model_dir, ref, bad) -> None:
    engine = make_engine(model_dir, FakeModel(wav=bad))
    with pytest.raises(VoiceEngineError):
        engine.synthesize("Hello", "en", "v", reference_audio=ref)


def test_load_failure_raises_load_error(model_dir, ref) -> None:
    def loader(path, device):
        raise OSError("corrupt checkpoint")

    engine = XttsEngine(str(model_dir), "cpu", model_loader=loader)
    with pytest.raises(VoiceEngineLoadError):
        engine.synthesize("Hello", "en", "v", reference_audio=ref)


def test_invalid_device_setting(model_dir, ref) -> None:
    engine = XttsEngine(str(model_dir), "tpu", model_loader=lambda *a: FakeModel())
    with pytest.raises(VoiceEngineLoadError):
        engine.synthesize("Hello", "en", "v", reference_audio=ref)
