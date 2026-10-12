"""Unified FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.api import import_api, query_api
from knowledge.core.config import get_settings
from knowledge.core.exceptions import register_exception_handlers
from knowledge.core.logging import configure_logging
from knowledge.core.paths import get_front_page_dir


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        description="掌柜智库导入与查询 API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/front", StaticFiles(directory=get_front_page_dir()), name="front")

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "app": settings.app_name,
            "env": settings.app_env,
        }

    import_api.register_router(app)
    query_api.register_router(app)
    register_exception_handlers(app)

    return app


app = create_app()
