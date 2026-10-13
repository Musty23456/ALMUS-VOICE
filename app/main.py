"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import get_voice_service, router
from app.config import get_settings
from app.core.logging_config import configure_logging

logger = logging.getLogger("almus.app")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # noqa: ANN202
        if settings.xtts_preload and settings.voice_engine.lower() != "none":
            get_voice_service().preload()  # fail fast if the model cannot load
        yield

    app = FastAPI(
        title="Almus Voice AI",
        version=settings.app_version,
        description="Voice-generation API. Engine: none (501) or XTTS-v2 with a reference voice.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled error",
            extra={"fields": {"path": request.url.path, "error_type": type(exc).__name__}},
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()
