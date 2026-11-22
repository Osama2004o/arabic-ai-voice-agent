"""FastAPI entrypoint: wires the voice pipeline, AI agent, integrations, and frontend."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.notify import router as notify_router
from app.core.config import get_settings
from app.integrations import register_agent_integrations
from app.speech.router import router as speech_router
from app.speech.tts import warmup_tts_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    register_agent_integrations()
    if settings.app_env == "production" and not settings.api_key:
        logger.warning("API_KEY is not set: /send-email and /send-telegram will refuse requests.")
    try:
        await warmup_tts_cache()
    except Exception:
        logger.warning("TTS cache warm-up failed; continuing without cache.", exc_info=True)
    yield


settings = get_settings()

app = FastAPI(
    title="Arabic AI Voice Agent",
    description="Arabic customer-service voice agent: STT, LangGraph agent, TTS, Email and Telegram tools.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Credentials cannot be combined with a wildcard origin.
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(notify_router)
app.include_router(speech_router)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
