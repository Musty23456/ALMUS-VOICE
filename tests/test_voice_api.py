from fastapi.testclient import TestClient

from app.api.routes import get_voice_service
from app.services.voice_service import (
    AudioResult,
    VoiceEngine,
    VoiceService,
)

URL = "/v1/voice/generate"
VALID = {"text": "Hello world", "language": "en", "voice_id": "my_voice"}


def test_requires_api_key(client: TestClient) -> None:
    assert client.post(URL, json=VALID).status_code == 401


def test_rejects_wrong_api_key(client: TestClient) -> None:
    r = client.post(URL, json=VALID, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_fails_closed_when_no_keys_configured(client: TestClient, settings) -> None:
    settings.api_keys = ""
    r = client.post(URL, json=VALID, headers={"X-API-Key": "anything"})
    assert r.status_code == 503


def test_unconfigured_engine_returns_501(client: TestClient, auth) -> None:
    r = client.post(URL, json=VALID, headers=auth)
    assert r.status_code == 501
    assert "not configured" in r.json()["detail"].lower()
    assert r.headers["content-type"].startswith("application/json")


def test_hausa_accepted_but_engine_unconfigured(client: TestClient, auth) -> None:
    body = {**VALID, "language": "ha"}
    assert client.post(URL, json=body, headers=auth).status_code == 501


def test_rejects_empty_text(client: TestClient, auth) -> None:
    r = client.post(URL, json={**VALID, "text": "   "}, headers=auth)
    assert r.status_code == 422


def test_rejects_too_long_text(client: TestClient, auth) -> None:
    r = client.post(URL, json={**VALID, "text": "a" * 51}, headers=auth)
    assert r.status_code == 422
    assert "maximum length" in r.json()["detail"]


def test_rejects_unsupported_language(client: TestClient, auth) -> None:
    r = client.post(URL, json={**VALID, "language": "xx"}, headers=auth)
    assert r.status_code == 422
    assert "Unsupported language" in r.json()["detail"]


def test_rejects_missing_fields(client: TestClient, auth) -> None:
    assert client.post(URL, json={"text": "hi"}, headers=auth).status_code == 422


def test_rejects_bad_voice_id(client: TestClient, auth) -> None:
    r = client.post(URL, json={**VALID, "voice_id": "../etc"}, headers=auth)
    assert r.status_code == 422


class _FakeEngine(VoiceEngine):
    """Test double used ONLY to verify the response contract. Not a real engine."""

    name = "fake-test-double"

    @property
    def is_configured(self) -> bool:
        return True

    def synthesize(self, text: str, language: str, voice_id: str) -> AudioResult:
        return AudioResult(data=b"RIFF....WAVE")


def test_wav_response_contract_with_test_double(client: TestClient, auth) -> None:
    client.app.dependency_overrides[get_voice_service] = lambda: VoiceService(_FakeEngine())
    r = client.post(URL, json=VALID, headers=auth)
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert "attachment" in r.headers["content-disposition"]
    assert r.content == b"RIFF....WAVE"
