"""Secure storage for reference-voice recordings.

Rules enforced here:
- voice ids are restricted to ``[A-Za-z0-9_-]{1,64}`` (no dots, slashes, spaces);
- the final path is resolved and must stay inside the storage root;
- only PCM WAV data is accepted, with size, duration and sample-rate limits;
- files are written atomically with owner-only permissions;
- no remote URLs are ever fetched: audio arrives only as an upload body.
"""

import io
import os
import re
import secrets
import wave
from dataclasses import dataclass
from pathlib import Path

VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class VoiceStoreError(Exception):
    """Base class for store errors."""


class InvalidVoiceIdError(VoiceStoreError):
    pass


class InvalidReferenceAudioError(VoiceStoreError):
    pass


class VoiceNotFoundError(VoiceStoreError):
    pass


class VoiceExistsError(VoiceStoreError):
    pass


@dataclass(frozen=True)
class ReferenceInfo:
    voice_id: str
    duration_seconds: float
    sample_rate: int
    channels: int
    size_bytes: int


class VoiceStore:
    def __init__(
        self,
        root: Path,
        *,
        min_seconds: float,
        max_seconds: float,
        max_bytes: int,
        min_sample_rate: int,
    ) -> None:
        self._root = Path(root)
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.max_bytes = max_bytes
        self.min_sample_rate = min_sample_rate

    @classmethod
    def from_settings(cls, settings) -> "VoiceStore":  # noqa: ANN001
        return cls(
            Path(settings.reference_voice_dir),
            min_seconds=settings.reference_min_seconds,
            max_seconds=settings.reference_max_seconds,
            max_bytes=settings.reference_max_bytes,
            min_sample_rate=settings.reference_min_sample_rate,
        )

    # -- paths -----------------------------------------------------------
    def path_for(self, voice_id: str) -> Path:
        if not isinstance(voice_id, str) or not VOICE_ID_RE.fullmatch(voice_id):
            raise InvalidVoiceIdError(
                "voice_id must be 1-64 characters: letters, digits, '_' or '-'."
            )
        root = self._root.resolve()
        candidate = (root / f"{voice_id}.wav").resolve()
        if candidate.parent != root:  # defence in depth against traversal / symlinks
            raise InvalidVoiceIdError("Invalid voice_id.")
        return candidate

    # -- validation ------------------------------------------------------
    def validate_wav(self, voice_id: str, data: bytes) -> ReferenceInfo:
        if len(data) > self.max_bytes:
            raise InvalidReferenceAudioError(
                f"File is too large (maximum {self.max_bytes} bytes)."
            )
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.getnframes()
        except (wave.Error, EOFError):
            raise InvalidReferenceAudioError(
                "Not a valid uncompressed PCM WAV file."
            ) from None
        if channels not in (1, 2):
            raise InvalidReferenceAudioError("Audio must be mono or stereo.")
        if rate < self.min_sample_rate:
            raise InvalidReferenceAudioError(
                f"Sample rate must be at least {self.min_sample_rate} Hz."
            )
        if frames * channels * width > len(data):
            raise InvalidReferenceAudioError("WAV header does not match the file size.")
        duration = frames / rate
        if duration < self.min_seconds:
            raise InvalidReferenceAudioError(
                f"Recording is too short (minimum {self.min_seconds:g} seconds)."
            )
        if duration > self.max_seconds:
            raise InvalidReferenceAudioError(
                f"Recording is too long (maximum {self.max_seconds:g} seconds)."
            )
        return ReferenceInfo(voice_id, duration, rate, channels, len(data))

    # -- operations ------------------------------------------------------
    def save(self, voice_id: str, data: bytes, *, overwrite: bool = False) -> ReferenceInfo:
        target = self.path_for(voice_id)
        info = self.validate_wav(voice_id, data)
        if target.exists() and not overwrite:
            raise VoiceExistsError(f"Voice '{voice_id}' already exists.")
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = target.parent / f".tmp-{secrets.token_hex(8)}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()
        return info

    def get_path(self, voice_id: str) -> Path:
        path = self.path_for(voice_id)
        if not path.is_file():
            raise VoiceNotFoundError(f"Reference voice '{voice_id}' was not found.")
        return path

    def delete(self, voice_id: str) -> None:
        path = self.get_path(voice_id)
        path.unlink()

    def list_ids(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(p.stem for p in self._root.glob("*.wav") if VOICE_ID_RE.fullmatch(p.stem))
