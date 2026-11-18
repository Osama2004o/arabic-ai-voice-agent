// Voice Pipeline & UI - Member 1
// Target Latency: < 3.0s (End of Speech -> Beginning of Audio Response)

const VOICE_ENDPOINT = "/chat/voice"; // real LangGraph agent (stub test route: /speech/demo-turn)
const TTS_ENDPOINT = "/speech/tts";
const TEXT_CHAT_ENDPOINT = "/chat/text";

// Session persistence across turns
const sessionId = (typeof crypto !== "undefined" && crypto.randomUUID)
  ? crypto.randomUUID()
  : "session_" + Math.random().toString(36).substring(2, 11);

// DOM Elements
const micBtn = document.getElementById("micBtn");
const micHint = document.getElementById("micHint");
const statusIndicator = document.querySelector(".status-indicator");
const statusText = document.getElementById("statusText");
const voiceSelect = document.getElementById("voiceSelect");
const chatMessages = document.getElementById("chatMessages");
const replyAudio = document.getElementById("replyAudio");
const replayBtn = document.getElementById("replayBtn");
const textForm = document.getElementById("textForm");
const textInput = document.getElementById("textInput");

// Metric Elements
const clientLatencyVal = document.getElementById("clientLatencyVal");
const targetStatus = document.getElementById("targetStatus");
const sttVal = document.getElementById("sttVal");
const llmVal = document.getElementById("llmVal");
const ttsVal = document.getElementById("ttsVal");
const serverTotalVal = document.getElementById("serverTotalVal");

const sttBar = document.getElementById("sttBar");
const llmBar = document.getElementById("llmBar");
const ttsBar = document.getElementById("ttsBar");
const serverTotalBar = document.getElementById("serverTotalBar");

const statTurns = document.getElementById("statTurns");
const statAvg = document.getElementById("statAvg");
const statMin = document.getElementById("statMin");
const statMax = document.getElementById("statMax");

// State Variables
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let stopRecordingTimestamp = 0; // t0 for client-measured latency
let currentMimeType = "audio/webm";

// Session statistics
const sessionStats = {
  turns: 0,
  clientLatencies: [],
};

// Select supported MIME type
function getSupportedMimeType() {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return "audio/webm";
}

// UI State Updater
function setUIState(state, message) {
  statusIndicator.className = "status-indicator " + state;
  statusText.textContent = message;

  if (state === "idle") {
    micBtn.className = "mic-button idle";
    micHint.textContent = "اضغط لبدء التحدث";
  } else if (state === "recording") {
    micBtn.className = "mic-button recording";
    micHint.textContent = "اضغط لإنهاء التحدث وإرسال الصوت";
  } else if (state === "processing") {
    micBtn.className = "mic-button processing";
    micHint.textContent = "جاري المعالجة السريعة...";
  } else if (state === "speaking") {
    micBtn.className = "mic-button idle";
    micHint.textContent = "جاري تشغيل الرد الصوتي...";
  } else if (state === "error") {
    micBtn.className = "mic-button idle";
    micHint.textContent = "اضغط للمحاولة مرة أخرى";
  }
}

// Add message bubble to chat
function appendMessage(sender, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}-message`;

  const bubbleDiv = document.createElement("div");
  bubbleDiv.className = "bubble";
  bubbleDiv.textContent = text;

  const timeSpan = document.createElement("span");
  timeSpan.className = "message-time";
  timeSpan.textContent = sender === "user" ? "أنت" : "المساعد الذكي";

  msgDiv.appendChild(bubbleDiv);
  msgDiv.appendChild(timeSpan);
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update Latency & Session Statistics
function updateMetrics(serverTimings, clientDurationSec) {
  if (clientDurationSec !== null) {
    clientLatencyVal.textContent = `${clientDurationSec.toFixed(2)} ثانية`;

    if (clientDurationSec <= 3.0) {
      targetStatus.className = "target-indicator success";
      targetStatus.textContent = `✓ محقق للهدف: ${clientDurationSec.toFixed(2)}s (أقل من 3 ثوانٍ)`;
    } else {
      targetStatus.className = "target-indicator fail";
      targetStatus.textContent = `✗ متجاوز للهدف: ${clientDurationSec.toFixed(2)}s (أكثر من 3 ثوانٍ)`;
    }

    sessionStats.turns += 1;
    sessionStats.clientLatencies.push(clientDurationSec);

    const sum = sessionStats.clientLatencies.reduce((a, b) => a + b, 0);
    const avg = sum / sessionStats.clientLatencies.length;
    const min = Math.min(...sessionStats.clientLatencies);
    const max = Math.max(...sessionStats.clientLatencies);

    statTurns.textContent = sessionStats.turns;
    statAvg.textContent = `${avg.toFixed(2)}s`;
    statMin.textContent = `${min.toFixed(2)}s`;
    statMax.textContent = `${max.toFixed(2)}s`;
  }

  if (serverTimings) {
    sttVal.textContent = `${serverTimings.stt || 0} ms`;
    llmVal.textContent = `${serverTimings.llm || 0} ms`;
    ttsVal.textContent = `${serverTimings.tts || 0} ms`;
    serverTotalVal.textContent = `${serverTimings.server_total || 0} ms`;

    // Maximum expected budget is 3000ms for scale calculation
    const maxScale = 3000;
    sttBar.style.width = `${Math.min(100, ((serverTimings.stt || 0) / maxScale) * 100)}%`;
    llmBar.style.width = `${Math.min(100, ((serverTimings.llm || 0) / maxScale) * 100)}%`;
    ttsBar.style.width = `${Math.min(100, ((serverTimings.tts || 0) / maxScale) * 100)}%`;
    serverTotalBar.style.width = `${Math.min(100, ((serverTimings.server_total || 0) / maxScale) * 100)}%`;
  }
}

// Audio Event Listeners for Client Latency Clock
replyAudio.addEventListener("playing", () => {
  if (stopRecordingTimestamp > 0) {
    const playTimestamp = performance.now();
    const durationSeconds = (playTimestamp - stopRecordingTimestamp) / 1000.0;
    updateMetrics(null, durationSeconds);
    stopRecordingTimestamp = 0; // reset
  }
  setUIState("speaking", "جاري التحدث...");
  replayBtn.style.display = "inline-flex";
});

replyAudio.addEventListener("ended", () => {
  setUIState("idle", "جاهز للاستماع");
});

replayBtn.addEventListener("click", () => {
  if (replyAudio.src) {
    replyAudio.currentTime = 0;
    replyAudio.play().catch(e => console.error("Playback error:", e));
  }
});

// Initialize Microphone Stream
async function initMicStream() {
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  }
  return mediaStream;
}

// Start Recording
async function startRecording() {
  try {
    const stream = await initMicStream();
    currentMimeType = getSupportedMimeType();

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: currentMimeType });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stopRecordingTimestamp = performance.now(); // Mark t0
      setUIState("processing", "جاري تحليل الصوت وتوليد الرد...");

      const audioBlob = new Blob(audioChunks, { type: currentMimeType });
      await sendVoiceTurn(audioBlob, currentMimeType);
    };

    mediaRecorder.start();
    isRecording = true;
    setUIState("recording", "جاري التسجيل... اضغط مرة أخرى للإنهاء");
  } catch (err) {
    console.error("Microphone access error:", err);
    setUIState("error", "فشل الوصول إلى الميكروفون. يرجى السماح بالصلاحيات.");
    alert("تعذر الوصول إلى الميكروفون. يرجى التأكد من منح الإذن للمتصفح.");
  }
}

// Stop Recording
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    isRecording = false;
  }
}

// Toggle Mic Click
micBtn.addEventListener("click", () => {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

// Send Voice Turn to Backend
async function sendVoiceTurn(audioBlob, mimeType) {
  try {
    const formData = new FormData();
    const ext = mimeType.includes("mp4") ? "mp4" : "webm";
    formData.append("audio", audioBlob, `recording.${ext}`);
    formData.append("voice", voiceSelect.value);
    formData.append("session_id", sessionId);

    const response = await fetch(VOICE_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server returned error: ${response.status}`);
    }

    const data = await response.json();

    // Render User Message
    if (data.transcript && data.transcript.trim()) {
      appendMessage("user", data.transcript);
    } else {
      appendMessage("user", "[صوت غير واضح أو صمت]");
    }

    // Render Agent Message
    if (data.reply_text) {
      appendMessage("agent", data.reply_text);
    }

    // Update Server Timings
    updateMetrics(data.timings_ms, null);

    // Playback Response Audio
    if (data.audio_base64) {
      replyAudio.src = `data:${data.audio_mime || "audio/mpeg"};base64,${data.audio_base64}`;
      try {
        await replyAudio.play();
      } catch (playErr) {
        console.warn("Auto-play was blocked by browser. User can click replay:", playErr);
        // Fallback: if browser blocks autoplay, still mark client latency
        if (stopRecordingTimestamp > 0) {
          const fallbackDuration = (performance.now() - stopRecordingTimestamp) / 1000.0;
          updateMetrics(null, fallbackDuration);
          stopRecordingTimestamp = 0;
        }
        setUIState("idle", "اضغط على زر إعادة الاستماع لتشغيل الرد.");
      }
    } else {
      setUIState("idle", "جاهز للاستماع");
    }
  } catch (error) {
    console.error("Turn error:", error);
    setUIState("error", "حدث خطأ أثناء التواصل مع الخادم.");
    appendMessage("agent", "عذراً، حدث خطأ أثناء الاتصال بالخادم. يرجى المحاولة مرة أخرى.");
  }
}

// Text Fallback Form Submission
textForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;

  textInput.value = "";
  appendMessage("user", text);
  setUIState("processing", "جاري إرسال النص...");

  const t0 = performance.now();

  try {
    // 1) Ask the agent (same session as voice turns, so context is shared)
    const chatRes = await fetch(TEXT_CHAT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: sessionId }),
    });
    if (!chatRes.ok) throw new Error(`Agent failed: ${chatRes.status}`);
    const chat = await chatRes.json();
    const replyText = chat.response || "";
    appendMessage("agent", replyText);
    const tLlm = performance.now();

    // 2) Speak the agent's reply
    const res = await fetch(TTS_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: replyText, voice: voiceSelect.value }),
    });
    if (!res.ok) throw new Error("TTS failed");

    const audioBlob = await res.blob();
    const audioUrl = URL.createObjectURL(audioBlob);

    const llmMs = Math.round(tLlm - t0);
    const ttsMs = Math.round(performance.now() - tLlm);
    const clientSec = (performance.now() - t0) / 1000.0;
    updateMetrics({ stt: 0, llm: llmMs, tts: ttsMs, server_total: llmMs + ttsMs }, clientSec);

    replyAudio.src = audioUrl;
    await replyAudio.play();
  } catch (err) {
    console.error("Text turn failed:", err);
    setUIState("error", "فشل معالجة النص.");
  }
});
