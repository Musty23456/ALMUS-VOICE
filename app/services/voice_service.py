"""Voice service layer.

``VoiceEngine`` is the interface every engine implements. ``NullVoiceEngine`` (the
Phase 1 default) reports that no engine exists and never returns audio.
``XttsEngine`` (Phase 2) lives in ``xtts_engine.py`` and is built on demand.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.services.voice_store import VoiceStore

NOT_CONFIGURED_MESSAGE = (
    "No voice engine is configured. Voice generation is not available yet."
)


class VoiceEngineNotConfiguredError(Exception):
    """Raised when audio is requested but no real voice engine is available."""


class VoiceEngineError(Exception):
    """Raised when a configured engine fails while generating audio."""


class VoiceEngineLoadError(VoiceEngineError):
    """Raised when the engine or its model cannot be loaded."""


@dataclass(frozen=True)
class AudioResult:
    """Generated audio. ``data`` must be complete WAV bytes."""

    data: bytes
    media_type: str = "audio/wav"
    file_extension: str = "wav"


class VoiceEngine(ABC):
    """Interface every concrete voice engine must implement."""

    name: str

    @property
    @abstractmethod
    def is_configured(self) -> bool: ...

    @property
    def supported_languages(self) -> frozenset[str] | None:
        """Languages the engine can really speak. ``None`` means no engine-level limit."""
        return None

    def load(self) -> None:
        """Load heavy resources (no-op by default)."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: str,
        voice_id: str,
        reference_audio: Path | None = None,
    ) -> AudioResult: ...


class NullVoiceEngine(VoiceEngine):
    """Placeholder that honestly reports no engine exists. Produces no audio."""

    name = "none"

    @property
    def is_configured(self) -> bool:
        return False

    def synthesize(
        self,
        text: str,
        language: str,
        voice_id: str,
        reference_audio: Path | None = None,
    ) -> AudioResult:
        raise VoiceEngineNotConfiguredError(NOT_CONFIGURED_MESSAGE)


class VoiceService:
    def __init__(self, engine: VoiceEngine, store: VoiceStore | None = None) -> None:
        self._engine = engine
        self._store = store

    @property
    def is_configured(self) -> bool:
        return self._engine.is_configured

    @property
    def engine_name(self) -> str:
        return self._engine.name

    @property
    def supported_languages(self) -> frozenset[str] | None:
        return self._engine.supported_languages

    def preload(self) -> None:
        self._engine.load()

    def generate(self, text: str, language: str, voice_id: str) -> AudioResult:
        if not self._engine.is_configured:
            raise VoiceEngineNotConfiguredError(NOT_CONFIGURED_MESSAGE)
        kwargs: dict[str, Path] = {}
        if self._store is not None:
            # Raises VoiceNotFoundError / InvalidVoiceIdError for unknown or unsafe ids.
            kwargs["reference_audio"] = self._store.get_path(voice_id)
        return self._engine.synthesize(text=text, language=language, voice_id=voice_id, **kwargs)


def build_voice_service(
    engine_name: str = "none",
    settings=None,  # noqa: ANN001
    store: VoiceStore | None = None,
) -> VoiceService:
    """Select an engine by name: 'none' (Phase 1) or 'xtts_v2' (Phase 2)."""
    name = engine_name.lower()
    if name == "none":
        return VoiceService(NullVoiceEngine(), store)
    if name == "xtts_v2":
        if settings is None:
            raise ValueError("settings are required for the xtts_v2 engine")
        from app.services.xtts_engine import XttsEngine  # lazy: keeps torch out of import time

        return VoiceService(XttsEngine(settings.xtts_model_dir, settings.xtts_device), store)
    raise ValueError(f"Unknown voice engine: {engine_name!r}")
