"""Route behaviour with a mocked engine and a real VoiceStore."""

import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_voice_service
from app.services.audio import encode_wav_pcm16
from app.services.voice_service import (
    AudioResult,
    VoiceEngine,
    VoiceEngineError,
    VoiceEngineLoadError,
    VoiceService,
)
from tests.helpers import make_wav

URL = "/v1/voice/generate"
BODY = {"text": "Hello, welcome to Almus AI.", "language": "en", "voice_id": "almus_owner"}


class MockEngine(VoiceEngine):
    name = "mock"

    def __init__(self, error=None, languages=frozenset({"en", "fr"})) -> None:
        self.error = error
        self.languages = languages
        self.last_reference = None

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def supported_languages(self):
        return self.languages

    def synthesize(self, text, language, voice_id, reference_audio=None):
        self.last_reference = reference_audio
        if self.error:
            raise self.error
        return AudioResult(data=encode_wav_pcm16([0.2, -0.2] * 500, 24000))


def use(client, store, engine):
    client.app.dependency_overrides[get_voice_service] = lambda: VoiceService(engine, store)


def test_success_returns_wav_using_stored_reference(client: TestClient, store, auth) -> None:
    store.save("almus_owner", make_wav(6))
    engine = MockEngine()
    use(client, store, engine)
    r = client.post(URL, json=BODY, headers=auth)
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content), "rb") as w:
        assert w.getnframes() == 1000
    assert engine.last_reference == store.get_path("almus_owner")


def test_unknown_voice_is_404(client: TestClient, store, auth) -> None:
    use(client, store, MockEngine())
    r = client.post(URL, json=BODY, headers=auth)
    assert r.status_code == 404


def test_language_not_supported_by_engine_is_422(client: TestClient, store, auth) -> None:
    store.save("almus_owner", make_wav(6))
    use(client, store, MockEngine())
    r = client.post(URL, json={**BODY, "language": "ha"}, headers=auth)
    assert r.status_code == 422
    assert "not supported by the active voice engine" in r.json()["detail"]


@pytest.mark.parametrize(
    ("error", "status"),
    [(VoiceEngineError("x"), 502), (VoiceEngineLoadError("x"), 503)],
)
def test_engine_failures_never_return_audio(client: TestClient, store, auth, error, status) -> None:
    store.save("almus_owner", make_wav(6))
    use(client, store, MockEngine(error=error))
    r = client.post(URL, json=BODY, headers=auth)
    assert r.status_code == status
    assert r.headers["content-type"].startswith("application/json")
    assert "Traceback" not in r.text


def test_phase1_behaviour_preserved_when_no_engine(client: TestClient, auth) -> None:
    r = client.post(URL, json=BODY, headers=auth)
    assert r.status_code == 501
