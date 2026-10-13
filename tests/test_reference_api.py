from fastapi.testclient import TestClient

from tests.helpers import make_wav

WAV = {"Content-Type": "audio/wav"}


def put(client, voice_id, data, headers, **params):
    return client.put(f"/v1/voices/{voice_id}", content=data, headers=headers, params=params)


def test_upload_requires_admin_key(client: TestClient, auth) -> None:
    assert put(client, "v", make_wav(6), WAV).status_code == 401
    # a generation key must not work for voice management
    assert put(client, "v", make_wav(6), {**WAV, **auth}).status_code == 401


def test_upload_fails_closed_without_admin_keys(client: TestClient, settings, admin) -> None:
    settings.admin_api_keys = ""
    assert put(client, "v", make_wav(6), {**WAV, **admin}).status_code == 503


def test_upload_list_delete(client: TestClient, admin) -> None:
    r = put(client, "almus_owner", make_wav(6), {**WAV, **admin})
    assert r.status_code == 201
    body = r.json()
    assert body["voice_id"] == "almus_owner" and body["sample_rate"] == 16000
    assert "path" not in body
    assert client.get("/v1/voices", headers=admin).json() == {"voices": ["almus_owner"]}
    assert client.delete("/v1/voices/almus_owner", headers=admin).status_code == 204
    assert client.delete("/v1/voices/almus_owner", headers=admin).status_code == 404


def test_upload_conflict_and_overwrite(client: TestClient, admin) -> None:
    h = {**WAV, **admin}
    assert put(client, "v", make_wav(6), h).status_code == 201
    assert put(client, "v", make_wav(6), h).status_code == 409
    assert put(client, "v", make_wav(6), h, overwrite="true").status_code == 201


def test_upload_rejects_wrong_content_type(client: TestClient, admin) -> None:
    r = put(client, "v", make_wav(6), {"Content-Type": "application/json", **admin})
    assert r.status_code == 415


def test_upload_rejects_garbage_and_short(client: TestClient, admin) -> None:
    h = {**WAV, **admin}
    assert put(client, "v", b"not audio", h).status_code == 422
    assert put(client, "v", make_wav(1), h).status_code == 422


def test_upload_rejects_oversize(client: TestClient, store, admin) -> None:
    store.max_bytes = 1_000
    r = put(client, "v", make_wav(6), {**WAV, **admin})
    assert r.status_code == 413


def test_upload_rejects_unsafe_ids(client: TestClient, admin) -> None:
    h = {**WAV, **admin}
    assert put(client, "bad.id", make_wav(6), h).status_code == 422
    assert put(client, "..%2Fevil", make_wav(6), h).status_code in (404, 422)
