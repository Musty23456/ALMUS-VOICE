"""XTTS-v2 voice engine (Coqui, via the maintained ``coqui-tts`` package).

- No automatic downloads: the model directory must already exist (see README and
  ``scripts/download_xtts.py``), so licence acceptance is explicit.
- Heavy imports (torch, TTS) happen lazily on first load, never at import time.
- The model is loaded once and reused; inference is serialised with a lock.
- Speaker conditioning latents are cached per reference file.
- Any failure raises an exception. Audio is never fabricated.
"""

import importlib.util
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.services.audio import AudioEncodingError, encode_wav_pcm16
from app.services.voice_service import (
    AudioResult,
    VoiceEngine,
    VoiceEngineError,
    VoiceEngineLoadError,
)

logger = logging.getLogger("almus.xtts")

# Languages documented for XTTS-v2. Hausa ("ha") is NOT among them.
XTTS_LANGUAGES = frozenset(
    {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
     "zh-cn", "ja", "hu", "ko", "hi"}
)
REQUIRED_FILES = ("config.json", "model.pth", "vocab.json")
_LATENT_CACHE_SIZE = 16


class XttsEngine(VoiceEngine):
    name = "xtts_v2"

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        model_loader: Callable[[Path, str], Any] | None = None,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else None
        self._device_setting = device.lower()
        self._model_loader = model_loader  # injectable for tests
        self._model: Any = None
        self._lock = threading.RLock()
        self._latents: OrderedDict[tuple[str, int, int], tuple[Any, Any]] = OrderedDict()

    # -- interface -------------------------------------------------------
    @property
    def supported_languages(self) -> frozenset[str]:
        return XTTS_LANGUAGES

    @property
    def is_configured(self) -> bool:
        if self._model_dir is None:
            return False
        if not all((self._model_dir / f).is_file() for f in REQUIRED_FILES):
            return False
        if self._model_loader is None:
            return all(importlib.util.find_spec(m) is not None for m in ("torch", "TTS"))
        return True

    def load(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def synthesize(
        self,
        text: str,
        language: str,
        voice_id: str,
        reference_audio: Path | None = None,
    ) -> AudioResult:
        if reference_audio is None:
            raise VoiceEngineError("A reference voice recording is required.")
        with self._lock:
            self._ensure_loaded()
            try:
                gpt_cond_latent, speaker_embedding = self._conditioning(reference_audio)
                output = self._model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    enable_text_splitting=True,
                )
                samples = output["wav"]
                rate = self._sample_rate()
                data = encode_wav_pcm16(samples, rate)
            except AudioEncodingError as exc:
                logger.error("xtts output rejected", extra={"fields": {"reason": str(exc)}})
                raise VoiceEngineError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - never leak internals or user data
                logger.error(
                    "xtts inference failed",
                    extra={"fields": {"error_type": type(exc).__name__}},
                )
                raise VoiceEngineError("Speech inference failed.") from exc
        return AudioResult(data=data)

    # -- internals -------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.is_configured:
            raise VoiceEngineLoadError(
                "XTTS-v2 is not available: check XTTS_MODEL_DIR and that torch and "
                "coqui-tts are installed."
            )
        assert self._model_dir is not None
        device = self._resolve_device()
        try:
            loader = self._model_loader or self._default_loader
            self._model = loader(self._model_dir, device)
        except VoiceEngineLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "xtts load failed", extra={"fields": {"error_type": type(exc).__name__}}
            )
            raise VoiceEngineLoadError("The XTTS-v2 model failed to load.") from exc
        logger.info("xtts model loaded", extra={"fields": {"device": device}})

    def _resolve_device(self) -> str:
        if self._device_setting == "cpu":
            return "cpu"
        if self._device_setting not in ("auto", "cuda"):
            raise VoiceEngineLoadError("XTTS_DEVICE must be 'auto', 'cpu' or 'cuda'.")
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise VoiceEngineLoadError("PyTorch is not installed.") from exc
        cuda = torch.cuda.is_available()
        if self._device_setting == "cuda" and not cuda:
            raise VoiceEngineLoadError("XTTS_DEVICE=cuda but no CUDA device is available.")
        return "cuda" if cuda else "cpu"

    @staticmethod
    def _default_loader(model_dir: Path, device: str) -> Any:
        from TTS.tts.configs.xtts_config import XttsConfig  # noqa: PLC0415
        from TTS.tts.models.xtts import Xtts  # noqa: PLC0415

        config = XttsConfig()
        config.load_json(str(model_dir / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
        model.to(device)
        return model

    def _conditioning(self, reference_audio: Path) -> tuple[Any, Any]:
        stat = reference_audio.stat()
        key = (str(reference_audio), stat.st_mtime_ns, stat.st_size)
        cached = self._latents.get(key)
        if cached is not None:
            self._latents.move_to_end(key)
            return cached
        cfg = getattr(self._model, "config", None)
        kwargs: dict[str, Any] = {}
        for arg, attr in (
            ("gpt_cond_len", "gpt_cond_len"),
            ("gpt_cond_chunk_len", "gpt_cond_chunk_len"),
            ("max_ref_length", "max_ref_len"),
            ("sound_norm_refs", "sound_norm_refs"),
        ):
            if cfg is not None and hasattr(cfg, attr):
                kwargs[arg] = getattr(cfg, attr)
        latents = self._model.get_conditioning_latents(audio_path=[str(reference_audio)], **kwargs)
        self._latents[key] = latents
        while len(self._latents) > _LATENT_CACHE_SIZE:
            self._latents.popitem(last=False)
        return latents

    def _sample_rate(self) -> int:
        audio_cfg = getattr(getattr(self._model, "config", None), "audio", None)
        return int(getattr(audio_cfg, "output_sample_rate", 24000))
