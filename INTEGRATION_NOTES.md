# Integration Notes: Member 1 (Voice Pipeline & UI)

These interface contracts and integration instructions are prepared for **Member 2 (LLM / Agent)** and **Member 3 (FastAPI Deployment & Orchestration)**.

> **Status (backend integrated):** `/chat/text` and `/chat/voice` are implemented in `app/api/chat.py`,
> the frontend now calls `/chat/voice` (real agent), and Email/Telegram senders are attached to the agent's
> tools at startup by `app/integrations/register_agent_integrations()`. Sections 2B and 2C below are kept
> for reference. See the README for the API table, Docker, and deployment.

---

## 1. Contract with Member 2 (Conversational Agent)

Member 1 provides the end-to-end voice turn orchestrator `run_voice_turn`. Member 2 must provide an asynchronous callable:

```python
from typing import Awaitable, Callable

# Agent function signature:
AgentFn = Callable[[str, str], Awaitable[str]]
# async def respond(text: str, session_id: str) -> str
```

### Critical Requirements for Member 2's Agent:
1. **Response Length:** To satisfy the **< 3.0s total latency budget**, the agent's reply **must be 1–3 short sentences** (maximum 60 words). Long paragraphs significantly increase TTS synthesis and playback delays.
2. **Plain Text Only:** The reply must be pure Arabic text. **No markdown (`**`, `#`, backticks), no bullet points, no numbered lists, and no emojis**. (Member 1's `clean_text_for_tts` will strip them if present, but clean text generation keeps LLM generation time under 1.0s).
3. **Multi-Turn State:** Member 2 should manage conversation memory using `session_id`. Member 1 generates and persists a unique `session_id` per client session.

---

## 2. Contract with Member 3 (FastAPI & Server Integration)

Member 3 owns the main application server and production deployment. Member 1 provides:

### A. Including the Speech Router
```python
from app.speech.router import router as speech_router

# In Member 3's main.py:
app.include_router(speech_router)
```

This immediately exposes:
- `POST /speech/stt` (Speech-to-text upload)
- `POST /speech/tts` (Text-to-speech synthesis)
- `GET /speech/voices` (Available Arabic voices)
- `POST /speech/demo-turn` (Test turn with stub LLM)

### B. Integrating the Main Production Voice Route (`POST /chat/voice`)
In Member 3's API, implement `/chat/voice` using Member 1's `run_voice_turn` and Member 2's `agent.respond`:

```python
from fastapi import APIRouter, File, Form, UploadFile
from app.speech.pipeline import run_voice_turn
from app.agent import agent  # Member 2's agent instance

router = APIRouter()

@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    voice: str = Form("female"),
    session_id: str = Form("default"),
):
    audio_bytes = await audio.read()
    mime_type = audio.content_type or "audio/webm"

    result = await run_voice_turn(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        voice=voice,
        session_id=session_id,
        agent_fn=agent.respond,
    )
    return result
```

### C. Static Files & Frontend
Member 1 provides the complete static RTL Arabic frontend in `frontend/`. Member 3 can mount it as:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

---

## 3. Environment Variables
Add these to `.env`:
```bash
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
STT_PROVIDER=groq
GROQ_STT_MODEL=whisper-large-v3-turbo
LOCAL_WHISPER_MODEL=small
TTS_PROVIDER=edge
DEFAULT_VOICE=female
```
