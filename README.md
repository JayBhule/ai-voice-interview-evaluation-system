# 🎤 Voice Interview Simulator (STT, TTS & Ollama)

An AI-powered interview practice application for freshers that generates personalised interview questions from your resume and job description, lets you answer via voice or text, and evaluates your responses using a local LLM — all without sending your data to any cloud service.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 Resume parsing | PDF (PyMuPDF) and DOCX (python-docx) |
| 🤖 AI question generation | Personalised from your resume + JD via Ollama |
| 🎙️ Voice answers | Browser-based mic recording (streamlit-mic-recorder) |
| 🔤 Local STT | faster-whisper (runs on CPU or NVIDIA GPU) |
| 🔊 Local TTS | pyttsx3 + espeak-ng; degrades gracefully if unavailable |
| 📊 AI evaluation | 4-dimension scoring (0–20) with detailed feedback |
| 📝 Final report | Personalised Markdown report with download option |
| 💾 Interview history | SQLite storage; browse and download past sessions |
| 🔒 100% local | No cloud APIs, no API keys, no data leaves your machine |

---

## 🏗️ Architecture

```
app.py              → Streamlit UI, session state, page routing
interview_engine.py → Ollama LLM calls, document extraction, prompts
speech_utils.py     → faster-whisper STT, pyttsx3 TTS
database.py         → SQLite persistence
setup_check.py      → Pre-flight verification
```

See [STEP.md](STEP.md) for the complete execution flow diagram.

---

## ⚙️ Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| UI | Streamlit | ≥1.35 |
| LLM | Ollama (local) | ≥0.13 |
| PDF extraction | PyMuPDF | ≥1.24 |
| DOCX extraction | python-docx | ≥1.1 |
| STT | faster-whisper | ≥1.0 |
| TTS | pyttsx3 + espeak-ng | ≥2.90 |
| Mic recording | streamlit-mic-recorder | ≥0.0.8 |
| Database | SQLite3 | built-in |
| Config | python-dotenv | ≥1.0 |

---

## 📋 Prerequisites

- **Ubuntu 20.04+** (or WSL2 on Windows)
- **Python 3.11 or 3.12**
- **Ollama** installed and running
- **NVIDIA GPU** recommended but not required (CPU works)
- **~2 GB disk** for the LLM model + ~150 MB for Whisper base model
- **Internet access** for first-time model downloads only

---

## 🚀 Quick Start

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip \
    build-essential portaudio19-dev espeak-ng ffmpeg
```

> **espeak-ng** is required for TTS on Linux. Without it, the interview runs in text-only mode.

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```bash
ollama serve
```

### 3. Download a model

```bash
# Recommended (fast, works on 4 GB GPU or CPU):
ollama pull gemma2:2b

# Alternatives (larger = smarter but slower):
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

### 4. Clone / navigate to the project

```bash
cd /path/to/voice_interview_simulator
```

### 5. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 6. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on pyttsx3 in WSL:** TTS requires an audio device. In WSL2, audio may not work. The application will display a warning and continue in text-only mode.

### 7. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` if you want to change the model or Whisper size:

```env
OLLAMA_MODEL=gemma2:2b        # change to any downloaded model
WHISPER_MODEL_SIZE=base       # tiny/base/small/medium
```

### 8. Verify the setup

```bash
python setup_check.py
```

All items should show **[PASS]** or **[WARN]**. Fix any **[FAIL]** items before continuing.

### 9. Run the application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📖 How to Use

1. **Setup page:** Upload your resume (PDF or DOCX) and paste the job description.
2. **Configure:** Choose interview type (Technical / Project-based / HR / Mixed) and number of questions.
3. **Generate:** Click **Generate Interview Questions** — questions are created in ~10–30s.
4. **Preview:** Review the generated questions, then click **Start Interview**.
5. **Interview:** For each question:
   - The question is displayed (and spoken aloud if TTS is available).
   - Click **Start Recording**, speak your answer, click **Stop Recording**.
   - Review and edit the auto-transcribed text if needed.
   - Click **Submit & Next**.
6. **Feedback:** Instant AI feedback with a score out of 20 is shown after each answer.
7. **Report:** A full personalised report is generated at the end.
8. **Download:** Save the report as a Markdown file.
9. **History:** View and download past sessions from the History page.

---

## 🎙️ Microphone Notes

- The microphone is accessed **through the browser** (not the OS directly).
- In WSL2, your browser (running on Windows) accesses the **Windows microphone** — this works correctly.
- Grant microphone permission when the browser asks.
- If recording fails, use the **text fallback** mode — type your answer instead.

---

## 🔊 TTS Notes

- TTS uses `pyttsx3` with the `espeak-ng` backend.
- In **WSL2 without a virtual audio device**, TTS may not produce output.
- The application prints a warning and continues in **text-only mode** — all features remain functional.
- To enable audio in WSL2: install a PulseAudio bridge or use Windows audio passthrough.

---

## 🧠 Ollama Notes

- Ollama must be running (`ollama serve`) before starting the app.
- The model is configurable in `.env` — use any model you have pulled.
- On first use, faster-whisper downloads the Whisper model to `~/.cache/huggingface/`.
- If Ollama is slow, try `gemma2:2b` or `qwen2.5:3b` (both are fast on CPU/small GPU).

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|---------|
| `Cannot connect to Ollama` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull gemma2:2b` |
| `No module named 'faster_whisper'` | Run `pip install faster-whisper` |
| `pyttsx3 init failed` | Install espeak-ng: `sudo apt install espeak-ng` |
| Microphone not working | Use the ⌨️ Text fallback mode |
| Questions are irrelevant | Ensure resume text was extracted correctly (check preview) |
| `ModuleNotFoundError: fitz` | Run `pip install PyMuPDF` |
| Slow transcription | Use `WHISPER_MODEL_SIZE=tiny` in `.env` |
| Streamlit port in use | Run `streamlit run app.py --server.port 8502` |

---

## 🔒 Privacy Design

- **No data leaves your machine.** All LLM inference, STT, and TTS run locally.
- Resumes are held in memory only and not saved to disk.
- Audio recordings are written to a temporary file, transcribed, and immediately deleted.
- Interview data is stored in a local SQLite database (`data/interviews.db`).
- No API keys are required or used.

---

## 📁 Project Structure

```
voice_interview_simulator/
├── app.py                  # Streamlit UI and application flow
├── interview_engine.py     # LLM calls, question generation, evaluation
├── speech_utils.py         # STT (faster-whisper) and TTS (pyttsx3)
├── database.py             # SQLite operations
├── setup_check.py          # Environment verification
├── requirements.txt        # Python dependencies
├── .env.example            # Configuration template
├── .env                    # Your local config (not committed)
├── .gitignore
├── README.md               # This file
├── STEP.md                 # Execution flow documentation
└── data/
    ├── .gitkeep
    └── interviews.db       # Created automatically (not committed)
```

---

## 🔮 Potential Future Improvements

> These are NOT implemented in the current version.

- Multi-user accounts with authentication
- Cloud deployment (Streamlit Cloud / Docker)
- Vector-based RAG for large document retrieval
- Real-time audio streaming transcription
- Video recording and body language analysis
- Company-specific question bank
- Calendar integration for scheduling practice sessions

---

## 📌 Limitations

- LLM evaluation scores are **AI-generated practice feedback**, not real interview assessments.
- Question quality depends on the clarity of the resume and job description.
- TTS may not work in WSL2 or headless environments without an audio device.
- The Whisper `base` model has higher error rates with heavy accents or noisy audio.
- Ollama inference can be slow on CPU-only machines (30–120s per request).

---

## 🎓 About This Project

Built as a portfolio project to demonstrate:
- Local AI integration (Ollama + faster-whisper)
- Full-stack Python application development (Streamlit)
- Speech processing pipeline (STT + TTS)
- SQLite data persistence
- Clean, readable, well-documented Python code

**Technology:** Python 3.12, Streamlit, Ollama, faster-whisper, pyttsx3, SQLite
