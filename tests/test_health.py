from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "almus-voice-ai",
        "version": "0.1.0",
    }


def test_api_info(client: TestClient) -> None:
    response = client.get("/v1")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v1"
    assert body["supported_languages"] == ["en", "ha", "fr"]
    assert body["max_text_length"] == 50
    assert body["voice_engine_configured"] is False
