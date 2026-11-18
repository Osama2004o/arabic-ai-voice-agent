import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent import agent, arun_agent
from app.speech.config import DEFAULT_VOICE, MAX_AUDIO_BYTES
from app.speech.pipeline import run_voice_turn
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


class ChatTextRequest(BaseModel):
    text: str
    session_id: str = "default_session"


@app.post("/chat/text")
async def chat_text(req: ChatTextRequest):
    """Text-based turn against the LangGraph Conversational AI Agent."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")
    return await arun_agent(session_id=req.session_id, text=req.text)


@app.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    voice: str = Form(DEFAULT_VOICE),
    session_id: str = Form("default_session"),
):
    """Full voice turn connecting the Speech Pipeline (STT/TTS) with the Conversational AI Agent."""
    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds size limit.")
    mime_type = audio.content_type or "audio/webm"
    return await run_voice_turn(
        audio_bytes=content,
        mime_type=mime_type,
        voice=voice,
        session_id=session_id,
        agent_fn=agent.respond,
    )

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

