# STEP.md — Voice Interview Simulator: Execution Flow

This document traces the actual execution flow of the application, explaining
which file runs when, what each function does, what inputs it receives, and
what it returns.

---

## Initial Setup (Run Once)

```
User
 └─► python setup_check.py          # verify environment (optional but recommended)
 └─► streamlit run app.py           # start the application
```

---

## Step 1 — `app.py` — Application Startup

**File:** `app.py`  
**Triggered by:** `streamlit run app.py`

What happens:
1. Streamlit loads `app.py` and executes it top to bottom.
2. `st.set_page_config()` configures the browser tab title and layout.
3. Custom CSS is injected via `st.markdown()`.
4. **`database.init_db()`** is called (see Step 2).
5. **`init_session_state()`** sets all default values in `st.session_state`
   — only on first load; existing values are preserved on reruns.
6. **`render_sidebar()`** draws the navigation menu and configuration controls.
7. **`main()`** reads `st.session_state.page` and routes to the correct page function.

---

## Step 2 — `database.py` — Database Initialisation

**File:** `database.py`  
**Called by:** `app.py` at line `database.init_db()`

```
app.py
 └─► database.init_db()
       └─► _get_connection()   # opens/creates data/interviews.db
       └─► CREATE TABLE IF NOT EXISTS sessions (...)
       └─► CREATE TABLE IF NOT EXISTS questions (...)
```

Input: none  
Output: SQLite tables are created (idempotent — safe to run every startup)

---

## Step 3 — `app.py` → `page_setup()` — Interview Setup Page

**File:** `app.py`  
**Triggered by:** user navigates to the Setup page (default on first load)

What happens:
1. User uploads a **resume** (PDF or DOCX).
2. `st.file_uploader()` captures the file as a Streamlit `UploadedFile`.
3. **`interview_engine.extract_resume_text(uploaded_file)`** is called.

```
app.py → page_setup()
  └─► interview_engine.extract_resume_text(uploaded_file)
        └─► _extract_pdf(file_bytes)   # if .pdf, uses PyMuPDF (fitz)
             OR
             _extract_docx(file_bytes) # if .docx, uses python-docx
        └─► returns str (≤4000 chars)
  └─► st.session_state.resume_text = <extracted text>
```

4. User pastes or uploads a **job description**.
5. User selects **interview type** and **question count**.
6. User clicks **Generate Interview Questions**.

---

## Step 4 — `app.py` → `_generate_questions()` — LLM Question Generation

**File:** `app.py` → calls `interview_engine.py`  
**Triggered by:** user clicks "Generate Interview Questions"

```
app.py → _generate_questions()
  │
  ├─► interview_engine.check_ollama_health()
  │     └─► GET http://localhost:11434/api/tags   (requests library)
  │     └─► returns {"ollama_running": True, "model_available": True, ...}
  │
  ├─► interview_engine.generate_questions(
  │       resume_text,  job_description,
  │       interview_type, num_questions)
  │     │
  │     ├─► Builds a structured prompt string
  │     ├─► _call_ollama(prompt, expect_json=True)
  │     │     └─► POST http://localhost:11434/api/generate
  │     │         payload: {model, prompt, format:"json", stream:false}
  │     │         returns: raw JSON string
  │     ├─► _extract_json(raw)   # parses + validates JSON
  │     └─► returns list[str]    # e.g. ["Q1 text", "Q2 text", ...]
  │
  ├─► database.create_session(session_key, interview_type, role, num_q)
  │     └─► INSERT INTO sessions (...)
  │
  └─► st.session_state.questions = [...]
      st.session_state.page = "preview"
      st.rerun()
```

---

## Step 5 — `app.py` → `page_preview()` — Question Preview

**File:** `app.py`  
**Triggered by:** automatic rerun after question generation

What happens:
- Questions are displayed in expandable cards.
- User clicks **Start Interview** → sets `interview_started = True`, routes to interview page.

---

## Step 6 — `app.py` → `page_interview()` — Live Interview (Per Question)

**File:** `app.py`  
**Triggered by:** user enters the Interview page; repeats for each question

```
app.py → page_interview()
  │
  ├─► Reads st.session_state.current_q_index (idx)
  ├─► Displays question card (HTML/CSS)
  │
  ├─► TTS (if enabled and not yet spoken):
  │     speech_utils.is_tts_available()
  │       └─► _init_tts()  # lazy init of pyttsx3 engine
  │     speech_utils.speak_text(question_text)
  │       └─► pyttsx3 engine speaks via espeak-ng / local audio
  │     st.session_state["q_spoken_<idx>"] = True   # prevent replay on rerun
  │
  ├─► Microphone recording (if user selects mic mode):
  │     streamlit_mic_recorder.mic_recorder(...)
  │       └─► browser records audio → returns WAV bytes
  │
  ├─► STT transcription (if audio received):
  │     speech_utils.transcribe_audio(audio_bytes, model_size)
  │       └─► speech_utils.get_whisper_model(model_size)
  │             └─► faster_whisper.WhisperModel(size, device, compute_type)
  │                 (loaded once, cached in module-level _whisper_model)
  │       └─► writes bytes to temp WAV file
  │       └─► model.transcribe(tmp_path, language="en", vad_filter=True)
  │       └─► returns transcript str
  │       └─► deletes temp WAV file
  │
  ├─► User edits transcript in st.text_area (optional correction)
  │
  └─► User clicks "Submit & Next" → _submit_answer(idx, final_answer, q_text)
```

---

## Step 7 — `app.py` → `_submit_answer()` — Answer Evaluation & Persistence

**File:** `app.py` → calls `interview_engine.py` → calls `database.py`

```
app.py → _submit_answer(idx, answer_text, question_text)
  │
  ├─► interview_engine.evaluate_answer(
  │       question, answer, resume_text, job_description)
  │     │
  │     ├─► Builds evaluation prompt (with 4 scoring dimensions)
  │     ├─► _call_ollama(prompt, expect_json=True)
  │     │     └─► POST /api/generate → returns JSON string
  │     ├─► _extract_json(raw)     # parse JSON
  │     ├─► Validates + clamps scores (0–5 per dimension, 0–20 total)
  │     └─► returns dict: {
  │             score_relevance, score_technical, score_depth, score_clarity,
  │             total_score, did_well, missing_concepts,
  │             suggestions, example_answer, followup_q
  │           }
  │
  ├─► st.session_state.answers[idx]     = answer_text
  │   st.session_state.evaluations[idx] = evaluation
  │
  ├─► database.save_question(session_key, idx, question, answer, evaluation)
  │     └─► INSERT OR REPLACE INTO questions (...)
  │
  ├─► _show_evaluation_card(evaluation)   # display score + feedback
  │
  ├─► st.session_state.current_q_index += 1
  │
  └─► if all questions answered → _finish_interview()
      else → st.rerun() (loops back to Step 6 for next question)
```

---

## Step 8 — `app.py` → `page_report()` — Final Report

**File:** `app.py` → calls `interview_engine.py` → calls `database.py`

```
app.py → page_report()
  │
  ├─► interview_engine.generate_final_report(
  │       interview_type, questions, answers, evaluations,
  │       resume_text, job_description)
  │     │
  │     ├─► Builds summary of all Q+A+scores
  │     ├─► _call_ollama(prompt, expect_json=False)  # plain Markdown output
  │     ├─► _build_report_header(...)  # stats table from stored data
  │     └─► returns full Markdown string
  │
  ├─► database.save_summary(session_key, avg_score, summary_md)
  │     └─► UPDATE sessions SET avg_score=?, summary_md=? WHERE session_key=?
  │
  └─► st.markdown(report_md)          # render in browser
      st.download_button(...)          # offer .md download
```

---

## Step 9 — `app.py` → `page_history()` — Interview History

**File:** `app.py` → calls `database.py`

```
app.py → page_history()
  └─► database.get_all_sessions()
        └─► SELECT * FROM sessions ORDER BY created_at DESC
        └─► returns list[dict]

  For each session:
  └─► database.get_session_detail(session_key)
        └─► SELECT * FROM sessions WHERE session_key=?
        └─► SELECT * FROM questions WHERE session_key=? ORDER BY question_index
        └─► returns {"session": {...}, "questions": [{...},...]}
```

---

## Text-Based Workflow Diagram

```
streamlit run app.py
        │
        ▼
┌──────────────────┐
│   app.py startup  │  → database.init_db()
│   session state   │  → render_sidebar()
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   page_setup()   │  Resume upload → interview_engine.extract_resume_text()
│                  │  JD paste → session state
│                  │  Click Generate
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│   interview_engine.generate_questions()│ → Ollama POST /api/generate
│   + database.create_session()         │   (returns JSON questions list)
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  page_preview()  │  Show questions → user clicks Start
└────────┬─────────┘
         │
         ▼  (loops per question)
┌──────────────────────────────────────────────────────────────────┐
│  page_interview()                                                  │
│    │                                                               │
│    ├─ speech_utils.speak_text()    → pyttsx3 → espeak-ng (if TTS)│
│    ├─ mic_recorder()               → browser WAV bytes            │
│    ├─ speech_utils.transcribe_audio() → faster-whisper → str     │
│    ├─ st.text_area (edit transcript)                              │
│    └─ _submit_answer()                                            │
│         ├─ interview_engine.evaluate_answer() → Ollama            │
│         └─ database.save_question()           → SQLite            │
└────────┬─────────────────────────────────────────────────────────┘
         │ (after all questions)
         ▼
┌──────────────────────────────────────┐
│   page_report()                       │
│    ├─ interview_engine.generate_final_report() → Ollama           │
│    └─ database.save_summary()                  → SQLite           │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  page_history()  │  → database.get_all_sessions() → SQLite SELECT
└──────────────────┘
```

---

## File Responsibilities Summary

| File | Responsibility |
|------|----------------|
| `app.py` | UI, navigation, session state, page routing, orchestration |
| `interview_engine.py` | LLM prompting, question generation, answer evaluation, report generation, document extraction |
| `speech_utils.py` | Local STT (faster-whisper), local TTS (pyttsx3), audio file handling |
| `database.py` | SQLite CRUD: sessions, questions, evaluations, history |
| `setup_check.py` | Pre-flight dependency and connectivity verification |
| `.env` / `.env.example` | Configuration (model, URLs, sizes) |
| `requirements.txt` | Python dependency list |

---

## Setup vs Runtime Flow

### Initial Setup (once)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # edit if needed
python setup_check.py       # verify
```

### Runtime (every time)
```bash
source venv/bin/activate
ollama serve &              # if not already running as a service
streamlit run app.py
# Opens http://localhost:8501 in your browser
```
