---
title: Arabic AI Voice Agent
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Arabic customer-service voice agent (STT, LLM, TTS)
---

# Arabic AI Voice Agent (وكيل الذكاء الاصطناعي الصوتي العربي)
**Cellula Technologies — NLP Internship**

An intelligent, real-time Arabic voice assistant capable of understanding spoken Arabic (Modern Standard Arabic and Egyptian Colloquial Arabic), reasoning through a customer service agent, and responding naturally via male or female synthetic Arabic speech with **< 3.0 seconds end-to-end latency**.

---

## 🚀 Voice Pipeline Architecture (Member 1)

```
[Customer Mic] 
      │ (WebM/Opus Audio)
      ▼
[STT Engine: Groq Whisper Large-v3-Turbo] (≤ 1.0s)
      │ (Biased with Egyptian customer-service prompt)
      ▼
[Agent Reasoning: LangGraph / LLM Stub] (≤ 1.0s)
      │ (Strict store guardrails, 1-3 short plain Arabic sentences)
      ▼
[TTS Engine: ElevenLabs / Edge-TTS / Hugging Face CPU] (≤ 0.8s)
      │ (ElevenLabs: Sarah/Adam | Edge: Salma/Shakir | HF: TinyVoca)
      ▼
[Browser Speaker + Live Latency Display] (< 3.0s Total)
```

### Key Technical Highlights:
- **Ultra-Fast STT:** Primary speech-to-text powered by Groq's LPU-accelerated `whisper-large-v3-turbo` with Egyptian vocabulary biasing and an automatic local `faster-whisper` (`small`, int8 CPU) fallback.
- **Silence & Hallucination Guard:** Rejects silence and suppresses common Whisper hallucinations (`no_speech_prob > 0.6` filtering), safely responding with a polite retry prompt without overwhelming the LLM agent.
- **Multi-Provider Arabic TTS:** High-fidelity speech synthesis supporting **ElevenLabs** (`eleven_multilingual_v2`), **Microsoft Edge-TTS** (`ar-EG-SalmaNeural` & `ar-EG-ShakirNeural`), and offline CPU **Hugging Face** (`oddadmix/tiny-voca-arabic`). Features automatic fallback to Edge-TTS if quotas are exhausted.
- **Strict Domain Guardrails:** LangGraph agent strictly stays in-character for e-commerce customer service and politely deflects general-knowledge or out-of-domain queries.
- **In-Memory Audio Caching:** Pre-caches recurring system phrases (greetings, retry prompts) for sub-millisecond audio delivery.
- **Real-Time RTL Web UI:** Built with vanilla HTML5/CSS3/JS with dynamic 3-engine TTS switching (ElevenLabs / Edge-TTS / Hugging Face) and voice selection. Measures and reports real client-side end-to-end latency clock ($t_{\text{playing}} - t_{\text{stop recording}}$) with a live visual badge against the 3.0s target.

---

## 📦 Project Structure

```text
arabic-ai-voice-agent/
├── app/
│   ├── main.py                   # FastAPI entrypoint (routers, CORS, frontend, startup wiring)
│   ├── api/
│   │   ├── chat.py               # POST /chat/text, POST /chat/voice
│   │   ├── notify.py             # POST /send-email, POST /send-telegram (API key + rate limit)
│   │   └── health.py             # GET /health
│   ├── core/
│   │   ├── config.py             # Environment-based settings
│   │   └── security.py           # X-API-Key check, rate limiter
│   ├── integrations/
│   │   ├── email_client.py       # SMTP (Gmail etc.) or Resend API
│   │   ├── telegram_client.py    # Telegram Bot API
│   │   └── __init__.py           # Connects real senders to the agent's tools
│   ├── models/schemas.py         # Request/response models
│   ├── agent/                    # LangGraph customer-service agent (Member 2)
│   └── speech/                   # STT, TTS, voice pipeline, /speech/* routes (Member 1)
├── frontend/                     # RTL Arabic web UI (index.html, style.css, app.js)
├── benchmarks/                   # STT accuracy and latency benchmarks + results
├── tests/                        # pytest suites (speech, agent, backend)
├── report/                       # LaTeX report sections and figures
├── Dockerfile
├── docker-compose.yml
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # + tests, benchmarks, plots
├── .env.example                  # Environment variables template
├── INTEGRATION_NOTES.md
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/cellula-technologies/arabic-ai-voice-agent.git
cd arabic-ai-voice-agent

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements-dev.txt   # or requirements.txt for runtime only
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
```
Fill in at least `GROQ_API_KEY` / `OPENROUTER_API_KEY`.
- **TTS Engine:** Choose `TTS_PROVIDER=huggingface` (offline CPU), `TTS_PROVIDER=edge` (free, zero setup), or `TTS_PROVIDER=elevenlabs` (with `ELEVENLABS_API_KEY`).
- **Email and Telegram:** Optional. When not configured, the agent runs in "simulated" mode and logs sends. See [Email and Telegram setup](#-email-and-telegram-setup).

---

## 🚀 Running the Application

### Launch Standalone Server:
```bash
uvicorn app.main:app --reload --port 8000
```
Open your browser at **`http://localhost:8000`**.

1. Click the **Microphone** button to speak in Arabic (Egyptian colloquial or MSA).
2. Click again to stop.
3. The assistant will transcribe your speech, generate an answer, and speak it aloud immediately.
4. Select between **Hugging Face CPU**, **ElevenLabs**, and **Edge-TTS** engines, and choose your preferred voice (**Female** or **Male**) in the top header.
5. Inspect the live **Latency Panel** on the right side to verify stage-by-stage timings and whether the turn met the **< 3.0s** standard.

---

## 🔌 Backend API

Interactive docs: **`http://localhost:8000/docs`**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status, version, and which integrations are live |
| `POST` | `/chat/text` | `{"text": "...", "session_id": "...", "tts_provider": "..."}` → agent reply, action, confirmation flags |
| `POST` | `/chat/voice` | multipart `audio`, `voice` (`female`/`male`), `tts_provider` (`edge`/`elevenlabs`/`huggingface`), `session_id` → transcript, reply, base64 audio, timings |
| `POST` | `/send-email` | `{"to", "subject", "body"}` — requires `X-API-Key` |
| `POST` | `/send-telegram` | `{"message"}` — requires `X-API-Key` |
| `POST` | `/speech/stt` | Speech-to-text stage |
| `POST` | `/speech/tts` | Text-to-speech stage with `provider` (`edge` or `elevenlabs`) and `voice` |
| `GET` | `/speech/voices` | Available voices; accepts `?provider=...` query param and returns list of supported providers |
| `POST` | `/speech/demo-turn` | Voice turn with the stub LLM (testing only) |

```bash
curl -X POST http://localhost:8000/chat/text -H "Content-Type: application/json" \
     -d '{"text": "عايز أعرف حالة طلبي", "session_id": "demo1"}'

curl -X POST http://localhost:8000/send-telegram -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" -d '{"message": "اختبار من وكيل خدمة العملاء"}'
```

**How the agent sends messages:** the agent never sends anything directly. When a customer asks for
an email or Telegram message, it prepares the message and asks for confirmation. Only after the customer
confirms ("نعم", "ابعت", …) does the backend send it through the integrations below. If delivery fails,
the agent tells the customer instead of claiming success.

**Security:** `/send-email` and `/send-telegram` require the `X-API-Key` header when `API_KEY` is set
(in `APP_ENV=production` they refuse every request until it is set). All sends, including those triggered by
the agent, are rate-limited (`NOTIFY_RATE_LIMIT_PER_MIN`). `EMAIL_ALLOWED_DOMAINS` can restrict recipients.

---

## 🔊 Text-to-Speech (TTS) Setup

The application supports three distinct TTS engines, configurable via `.env` or switched dynamically live from the web UI:

### 1. Microsoft Edge-TTS (Free & Unlimited)
- **Zero API keys required.** High-quality neural Arabic voices streamed over WebSocket.
- Voices:
  - Female: `ar-EG-SalmaNeural` (Egyptian)
  - Male: `ar-EG-ShakirNeural` (Egyptian)
- In `.env`:
  ```ini
  TTS_PROVIDER=edge
  DEFAULT_VOICE=female
  ```

### 2. ElevenLabs (High-Fidelity Neural)
- Ultra-natural synthetic voices using the `eleven_multilingual_v2` model.
- **Free Tier Requirement:** Free accounts must use pre-made default voices. Community/Library voices are restricted by ElevenLabs to paid plans (HTTP 402).
- Supported pre-made voices:
  - Female: **Sarah** (`EXAVITQu4vr4xnSDxMaL`)
  - Male: **Adam** (`pNInz6obpgDQGcFmaJgB`)
- In `.env`:
  ```ini
  TTS_PROVIDER=elevenlabs
  ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
  ELEVENLABS_MODEL_ID=eleven_multilingual_v2
  ELEVENLABS_VOICE_FEMALE=EXAVITQu4vr4xnSDxMaL
  ELEVENLABS_VOICE_MALE=pNInz6obpgDQGcFmaJgB
  ```
- **Automatic Quota Protection & Fallback:** If `ELEVENLABS_API_KEY` is missing or character quotas are exceeded, the pipeline automatically and transparently falls back to Edge-TTS. In addition, startup audio cache warmup uses Edge-TTS to conserve ElevenLabs quota.

### 3. Hugging Face CPU (TinyVoca Arabic)
- **100% Offline & Free:** Ultra-compact 4.5M parameter Modern Standard Arabic TTS model (`oddadmix/tiny-voca-arabic`) running completely on local CPU via ONNX Runtime.
- Built-in CATT diacritizer transformer and source-filter waveform synthesizer: zero external API quota, zero network latency.
- Outputs 24 kHz mono 16-bit WAV audio in ~200-500ms on standard CPU.
- In `.env`:
  ```ini
  TTS_PROVIDER=huggingface
  HF_TTS_MODEL=oddadmix/tiny-voca-arabic
  DEFAULT_VOICE=female
  ```

---

## 📨 Email and Telegram Setup

### Email via Gmail (SMTP)
1. Enable 2-Step Verification on the Google account.
2. Create an **App Password**: Google Account → Security → App passwords.
3. In `.env`:
   ```ini
   EMAIL_PROVIDER=smtp
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=you@gmail.com
   SMTP_PASSWORD=<16-character app password>
   EMAIL_FROM=you@gmail.com
   ```

### Email via Resend (if your host blocks SMTP)
Many free hosting tiers block outbound SMTP ports. Resend uses HTTPS instead:
```ini
EMAIL_PROVIDER=resend
RESEND_API_KEY=re_...
EMAIL_FROM=onboarding@resend.dev   # or an address on your verified domain
```

### Telegram
1. In Telegram, message **@BotFather** → `/newbot` → copy the token.
2. Send any message to your new bot.
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `result[0].message.chat.id`.
4. In `.env`:
   ```ini
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=123456789
   ```

Check `GET /health`: `"integrations": {"email": true, "telegram": true}` means both are live.

---

## 🐳 Docker

```bash
docker compose up --build
```
Open `http://localhost:8000`. The container reads `.env`, runs as a non-root user, and has a health check on `/health`.

Without Compose:
```bash
docker build -t arabic-ai-voice-agent .
docker run --env-file .env -p 8000:8000 arabic-ai-voice-agent
```

---

## ☁️ Deployment

Set `APP_ENV=production`, a strong `API_KEY`, and the other variables from `.env.example` as the platform's
environment variables / secrets (never commit `.env`). Use `STT_PROVIDER=groq`: free CPU instances are too slow for local Whisper.

**Render**
1. New → Web Service → connect the GitHub repo → Runtime: **Docker**.
2. Add the environment variables. Health check path: `/health`.
3. Render sets `$PORT` automatically. The public URL is `https://<name>.onrender.com`.
4. Free instances sleep when idle (the first request takes ~1 minute) and block SMTP → use `EMAIL_PROVIDER=resend`.

**Railway**
1. New Project → Deploy from GitHub repo (the Dockerfile is detected).
2. Add the variables, then Settings → Networking → Generate Domain.

**Hugging Face Spaces** (recommended free option: 2 vCPU / 16 GB RAM, enough for local Whisper)
1. Create a Space: **New → Space**, SDK **Docker**, hardware **CPU basic (free)**.
2. Add every variable from `.env` under **Settings → Variables and secrets** (as secrets). Do not upload `.env`.
3. The Space settings are already in the front matter at the top of this README (`sdk: docker`, `app_port: 8000`).
4. Deploy (Git Bash):
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>   # once
   bash scripts/deploy_hf.sh
   ```
   Use a Hugging Face **write** token (huggingface.co/settings/tokens) as the password.
   The script pushes a clean snapshot of the current commit **without binary files** (benchmark audio,
   report figures), because Hugging Face rejects binaries that are not in Git LFS and this repo's history contains some.
5. Open `https://<username>-<space-name>.hf.space` directly (the microphone may not work inside the huggingface.co page frame).
6. Spaces block SMTP → use `EMAIL_PROVIDER=resend` for email. Free Spaces sleep after ~48 h without visits.

Microphone recording in the browser requires **HTTPS**, which all three platforms provide.

---

## 🧪 Testing & Benchmarks

### 1. Run Unit Tests
```bash
pytest -v
```

### 2. Run STT Accuracy Benchmark (Word Error Rate)
Transcribes all 30 benchmark clips across Groq Whisper Large-v3-Turbo, Groq Whisper Large-v3, and local Faster-Whisper Small:
```bash
python benchmarks/run_stt_accuracy.py
```
Outputs: `benchmarks/results/stt_accuracy.csv`.

### 3. Run Latency Benchmark
Executes 32 full conversational voice turns, logs per-stage latency, and computes P95 and percentage meeting $< 3.0$ seconds:
```bash
python benchmarks/run_latency.py
```
Outputs: `benchmarks/results/latency.csv` and `benchmarks/results/tts_latency.csv`.

### 4. Generate Report Plots
```bash
python report/plot_benchmarks.py
```
Outputs publication figures in `report/`:
- `latency_breakdown.png`
- `latency_distribution.png`
- `wer_comparison.png`

---

## 👥 Team Integration Notes
See [INTEGRATION_NOTES.md](INTEGRATION_NOTES.md) for the exact async signature contracts with **Member 2 (Conversational Agent)** and **Member 3 (FastAPI Deployment)**.
