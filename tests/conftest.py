import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_voice_service, get_voice_store
from app.config import Settings, get_settings
from app.main import create_app
from app.services.voice_store import VoiceStore

TEST_KEY = "test-key"
ADMIN_KEY = "admin-key"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        api_keys=TEST_KEY,
        admin_api_keys=ADMIN_KEY,
        max_text_length=50,
        supported_languages="en,ha,fr",
        reference_voice_dir=str(tmp_path / "voices"),
        reference_max_bytes=2_000_000,
    )


@pytest.fixture
def store(settings: Settings) -> VoiceStore:
    return VoiceStore.from_settings(settings)


@pytest.fixture
def client(settings: Settings, store: VoiceStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_voice_store] = lambda: store
    app.dependency_overrides.pop(get_voice_service, None)
    return TestClient(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": TEST_KEY}


@pytest.fixture
def admin() -> dict[str, str]:
    return {"X-Admin-Key": ADMIN_KEY}
