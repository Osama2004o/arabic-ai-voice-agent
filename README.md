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
│   ├── main.py                   # Standalone FastAPI server & static file host
│   └── speech/
│       ├── __init__.py           # Package exports
│       ├── config.py             # System configuration, voice mappings, constants
│       ├── text_utils.py         # TTS text sanitization & Arabic WER normalization
│       ├── latency.py            # LatencyTracker and statistical summarizers
│       ├── stt.py                # Groq Whisper primary + local CPU fallback
│       ├── tts.py                # Edge-TTS streaming, voice selection & in-memory cache
│       ├── stub_agent.py         # LangChain / OpenRouter LLM stub
│       ├── pipeline.py           # run_voice_turn end-to-end coordinator
│       └── router.py             # FastAPI APIRouter (/speech/* endpoints)
├── frontend/
│   ├── index.html                # RTL Arabic web interface
│   ├── style.css                 # Dark responsive styles & recording pulse
│   └── app.js                   # MediaRecorder, player & client latency clock
├── benchmarks/
│   ├── test_audio/               # 30 benchmark clips across 7 speech categories
│   │   └── references.json       # Benchmark metadata and ground truth text
│   ├── generate_synthetic_audio.py # Script to create evaluation clips
│   ├── run_stt_accuracy.py       # STT benchmark (computes WER using jiwer)
│   ├── run_latency.py            # End-to-end voice latency benchmark (>= 30 turns)
│   └── results/                  # Generated CSV benchmark results
├── tests/
│   └── test_speech_unit.py       # Comprehensive pytest unit test suite
├── report/
│   ├── section_speech.tex        # Complete LaTeX report section
│   └── plot_benchmarks.py        # Generates publication plots from benchmark CSVs
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── INTEGRATION_NOTES.md          # Interface contract for Members 2 and 3
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
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Contents of `.env`:
```ini
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
STT_PROVIDER=groq
GROQ_STT_MODEL=whisper-large-v3-turbo
LOCAL_WHISPER_MODEL=small
TTS_PROVIDER=edge
DEFAULT_VOICE=female
LLM_STUB_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

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

## 🧪 Testing & Benchmarks

### 1. Run Unit Tests
```bash
pytest tests/test_speech_unit.py -v
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
See [INTEGRATION_NOTES.md](file:///d:/Courses/nlp%20cellula/arabic-ai-voice-agent/INTEGRATION_NOTES.md) for the exact async signature contracts with **Member 2 (Conversational Agent)** and **Member 3 (FastAPI Deployment)**.
