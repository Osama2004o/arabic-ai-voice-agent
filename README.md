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
      │ (1-3 short plain Arabic sentences)
      ▼
[TTS Engine: Edge-TTS Neural Egyptian Voice] (≤ 0.8s)
      │ (Salma / Shakir MP3 Audio)
      ▼
[Browser Speaker + Live Latency Display] (< 3.0s Total)
```

### Key Technical Highlights:
- **Ultra-Fast STT:** Primary speech-to-text powered by Groq's LPU-accelerated `whisper-large-v3-turbo` with Egyptian vocabulary biasing and an automatic local `faster-whisper` (`small`, int8 CPU) fallback.
- **Silence & Hallucination Guard:** Rejects silence and suppress common Whisper hallucinations (`no_speech_prob > 0.6` filtering), safely responding with a polite retry prompt without overwhelming the LLM agent.
- **Natural Arabic TTS:** High-fidelity Egyptian voices (`ar-EG-SalmaNeural` for female, `ar-EG-ShakirNeural` for male) streamed via `edge-tts` with markdown stripping, emoji removal, and numeral spacing for clear reading.
- **In-Memory Audio Caching:** Pre-caches recurring system phrases (greetings, retry prompts) for sub-millisecond audio delivery.
- **Real-Time RTL Web UI:** Built with pure vanilla HTML5/CSS3/JS (no heavy framework build step). Measures and reports real client-side end-to-end latency clock ($t_{\text{playing}} - t_{\text{stop recording}}$) with a live visual badge against the 3.0s target.

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
Fill in at least `GROQ_API_KEY` / `OPENROUTER_API_KEY`. Email and Telegram are optional:
when they are not configured the agent still works and only *logs* the send ("simulated" mode).
See [Email and Telegram setup](#-email-and-telegram-setup).

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
4. Select between **Female (Salma)** and **Male (Shakir)** voices at the top.
5. Inspect the live **Latency Panel** on the right side to verify stage-by-stage timings and whether the turn met the **< 3.0s** standard.

---

## 🔌 Backend API

Interactive docs: **`http://localhost:8000/docs`**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status, version, and which integrations are live |
| `POST` | `/chat/text` | `{"text": "...", "session_id": "..."}` → agent reply, action, confirmation flags |
| `POST` | `/chat/voice` | multipart `audio`, `voice` (`female`/`male`/`female_msa`/`male_msa`), `session_id` → transcript, reply, base64 MP3, timings |
| `POST` | `/send-email` | `{"to", "subject", "body"}` — requires `X-API-Key` |
| `POST` | `/send-telegram` | `{"message"}` — requires `X-API-Key` |
| `POST` | `/speech/stt`, `/speech/tts` | Individual speech stages |
| `GET` | `/speech/voices` | Available voices |
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
