"""Application configuration, loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Values come from the environment or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "almus-voice-ai"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    # Comma-separated list of accepted API keys. Empty means generation is disabled.
    api_keys: str = ""

    max_text_length: int = Field(default=500, ge=1, le=10_000)

    # Comma-separated language identifiers accepted by the API.
    supported_languages: str = "en,ha"

    # Voice engine: "none" (Phase 1 behaviour, returns 501) or "xtts_v2".
    voice_engine: str = "none"

    # --- XTTS-v2 engine (Phase 2) ---
    # Directory containing the pre-downloaded XTTS-v2 files (config.json, model.pth, vocab.json).
    xtts_model_dir: str = ""
    # "auto" (cuda if available, else cpu), "cpu" or "cuda". An explicit "cuda" never falls back silently.
    xtts_device: str = "auto"
    # Load the model at server start instead of on the first request.
    xtts_preload: bool = False

    # --- Reference voice storage ---
    # Keep this directory OUTSIDE any public web root and never commit it.
    reference_voice_dir: str = "data/voices"
    reference_min_seconds: float = Field(default=5.0, gt=0)
    reference_max_seconds: float = Field(default=60.0, gt=0)
    reference_max_bytes: int = Field(default=10_000_000, ge=1_000, le=100_000_000)
    reference_min_sample_rate: int = Field(default=16_000, ge=8_000)

    # Comma-separated admin keys for managing reference voices. Separate from API_KEYS.
    admin_api_keys: str = ""

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def admin_key_list(self) -> list[str]:
        return [k.strip() for k in self.admin_api_keys.split(",") if k.strip()]

    @property
    def language_list(self) -> list[str]:
        return [
            lang.strip().lower()
            for lang in self.supported_languages.split(",")
            if lang.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
