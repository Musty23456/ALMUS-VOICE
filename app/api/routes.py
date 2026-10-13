"""HTTP routes."""

import hmac
import logging
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status

from app.config import Settings, get_settings
from app.schemas import (
    ApiInfoResponse,
    ErrorResponse,
    HealthResponse,
    VoiceGenerateRequest,
    VoiceInfo,
    VoiceListResponse,
)
from app.services.voice_service import (
    VoiceEngineError,
    VoiceEngineLoadError,
    VoiceEngineNotConfiguredError,
    VoiceService,
    build_voice_service,
)
from app.services.voice_store import (
    InvalidReferenceAudioError,
    InvalidVoiceIdError,
    VoiceExistsError,
    VoiceNotFoundError,
    VoiceStore,
)

logger = logging.getLogger("almus.api")

router = APIRouter()
v1 = APIRouter(prefix="/v1")

ALLOWED_WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


@lru_cache
def get_voice_store() -> VoiceStore:
    return VoiceStore.from_settings(get_settings())


@lru_cache
def get_voice_service() -> VoiceService:
    settings = get_settings()
    return build_voice_service(settings.voice_engine, settings, get_voice_store())


def _check_key(keys: list[str], supplied: str | None, what: str) -> None:
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{what} authentication is not configured on the server.",
        )
    if not supplied or not any(hmac.compare_digest(supplied, k) for k in keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Protect expensive endpoints. Fails closed if no keys are configured."""
    _check_key(settings.api_key_list, x_api_key, "API")


def require_admin_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> None:
    """Protect reference-voice management. Separate from the generation keys."""
    _check_key(settings.admin_key_list, x_admin_key, "Admin")


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok", service=settings.app_name, version=settings.app_version
    )


@v1.get("", response_model=ApiInfoResponse, tags=["system"])
def api_info(
    settings: Annotated[Settings, Depends(get_settings)],
    voice: Annotated[VoiceService, Depends(get_voice_service)],
) -> ApiInfoResponse:
    engine_langs = voice.supported_languages
    return ApiInfoResponse(
        service=settings.app_name,
        version=settings.app_version,
        api_version="v1",
        supported_languages=settings.language_list,
        max_text_length=settings.max_text_length,
        voice_engine_configured=voice.is_configured,
        endpoints=[
            "GET /health",
            "GET /v1",
            "POST /v1/voice/generate",
            "GET /v1/voices",
            "PUT /v1/voices/{voice_id}",
            "DELETE /v1/voices/{voice_id}",
        ],
        voice_engine=voice.engine_name,
        engine_languages=sorted(engine_langs) if engine_langs is not None else None,
    )


@v1.post(
    "/voice/generate",
    tags=["voice"],
    dependencies=[Depends(require_api_key)],
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Generated WAV audio."},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def generate_voice(
    payload: VoiceGenerateRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    voice: Annotated[VoiceService, Depends(get_voice_service)],
) -> Response:
    if len(payload.text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"text exceeds the maximum length of {settings.max_text_length} characters.",
        )
    if payload.language not in settings.language_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported language '{payload.language}'. "
                f"Supported: {', '.join(settings.language_list)}."
            ),
        )
    engine_langs = voice.supported_languages
    if voice.is_configured and engine_langs is not None and payload.language not in engine_langs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Language '{payload.language}' is not supported by the active voice engine "
                f"({voice.engine_name}). Supported: {', '.join(sorted(engine_langs))}."
            ),
        )

    # Log metadata only, never the text itself.
    logger.info(
        "voice generation requested",
        extra={"fields": {"language": payload.language, "text_length": len(payload.text)}},
    )

    try:
        audio = voice.generate(payload.text, payload.language, payload.voice_id)
    except VoiceEngineNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)
        ) from exc
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidVoiceIdError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except VoiceEngineLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The voice engine is unavailable.",
        ) from exc
    except VoiceEngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The voice engine failed to generate audio.",
        ) from exc

    return Response(
        content=audio.data,
        media_type=audio.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{audio.file_extension}"'
        },
    )


# ---------------------------------------------------------------- reference voices
@v1.get(
    "/voices",
    response_model=VoiceListResponse,
    tags=["voices"],
    dependencies=[Depends(require_admin_key)],
)
def list_voices(store: Annotated[VoiceStore, Depends(get_voice_store)]) -> VoiceListResponse:
    return VoiceListResponse(voices=store.list_ids())


@v1.put(
    "/voices/{voice_id}",
    response_model=VoiceInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["voices"],
    dependencies=[Depends(require_admin_key)],
    responses={
        409: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def upload_voice(
    voice_id: str,
    request: Request,
    store: Annotated[VoiceStore, Depends(get_voice_store)],
    overwrite: bool = False,
) -> VoiceInfo:
    """Upload a reference recording as the raw request body (Content-Type: audio/wav).

    Only the voice owner's consented recording should be uploaded. Remote URLs are
    never fetched.
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_WAV_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be audio/wav (uncompressed PCM WAV).",
        )
    too_large = HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"File is too large (maximum {store.max_bytes} bytes).",
    )
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > store.max_bytes:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > store.max_bytes:
            raise too_large
    try:
        info = store.save(voice_id, bytes(body), overwrite=overwrite)
    except (InvalidVoiceIdError, InvalidReferenceAudioError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except VoiceExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    logger.info("reference voice stored", extra={"fields": {"size_bytes": info.size_bytes}})
    return VoiceInfo(
        voice_id=info.voice_id,
        duration_seconds=round(info.duration_seconds, 2),
        sample_rate=info.sample_rate,
        channels=info.channels,
        size_bytes=info.size_bytes,
    )


@v1.delete(
    "/voices/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["voices"],
    dependencies=[Depends(require_admin_key)],
)
def delete_voice(
    voice_id: str, store: Annotated[VoiceStore, Depends(get_voice_store)]
) -> Response:
    try:
        store.delete(voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidVoiceIdError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


router.include_router(v1)
