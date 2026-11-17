import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.speech.router import router as speech_router
from app.speech.tts import warmup_tts_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up TTS cache asynchronously on startup
    try:
        await warmup_tts_cache()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Arabic AI Voice Agent - Member 1 Server",
    description="Speech processing pipeline, STT, TTS, and frontend for Arabic AI Voice Agent",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS setup for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include speech router
app.include_router(speech_router)

# Mount frontend directory for static serving
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "arabic-voice-pipeline",
        "stt_target_latency": "< 3.0s",
    }


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

