"""Pydantic request and response models."""

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ApiInfoResponse(BaseModel):
    service: str
    version: str
    api_version: str
    supported_languages: list[str]
    max_text_length: int
    voice_engine_configured: bool
    endpoints: list[str]
    voice_engine: str = "none"
    engine_languages: list[str] | None = None


class VoiceGenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise.")
    language: str = Field(..., description="Language identifier, e.g. 'en' or 'ha'.")
    voice_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Identifier of an authorised voice.",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        return value

    @field_validator("language")
    @classmethod
    def normalise_language(cls, value: str) -> str:
        return value.strip().lower()


class ErrorResponse(BaseModel):
    detail: str


class VoiceInfo(BaseModel):
    voice_id: str
    duration_seconds: float
    sample_rate: int
    channels: int
    size_bytes: int


class VoiceListResponse(BaseModel):
    voices: list[str]
