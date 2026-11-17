"""
app.py — Main Streamlit application for the Voice Interview Simulator.

This is the entry point. Run with:
    streamlit run app.py

Application flow:
  1. Startup → database.init_db()  → creates SQLite tables if needed
  2. Sidebar  → configuration (model, speech, Whisper size)
  3. Page: Setup     → upload resume, paste JD, configure interview
  4. Page: Questions Preview → show generated questions before starting
  5. Page: Interview → question-by-question, audio recording, STT, evaluation
  6. Page: Report    → final summary, download as Markdown
  7. Page: History   → list past sessions from SQLite

Session state is used throughout to persist interview data across reruns.
Key session-state variables are documented at the top of init_session_state().

IMPORTANT: Streamlit re-executes this entire script on every UI interaction.
Session state (st.session_state) is the ONLY way to preserve data across reruns.
We use boolean flags like 'interview_started', 'q_spoken_<i>' to prevent
duplicate LLM calls and repeated TTS playback on reruns.
"""

import io
import logging
import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Project modules
import database
import interview_engine as engine
import speech_utils as speech

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Interview Simulator",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.question-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 1.5rem;
    color: #e0e0ff;
    margin-bottom: 1.5rem;
}

.score-card {
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

.status-ok    { color: #3fb950; font-weight: 600; }
.status-warn  { color: #d29922; font-weight: 600; }
.status-error { color: #f85149; font-weight: 600; }

.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.progress-text {
    font-size: 0.85rem;
    color: #8b949e;
}
</style>
""", unsafe_allow_html=True)


# ── Database init (once per app start) ───────────────────────────────────────
database.init_db()


# ── Session state initialisation ──────────────────────────────────────────────
def init_session_state():
    """Initialise all session-state keys with defaults.

    This is called at the top of every rerun. Using `setdefault` means
    existing values are never overwritten.

    Key session-state variables:
      page              — current navigation page
      session_key       — UUID for the current interview session
      resume_text       — extracted text from uploaded resume
      job_description   — pasted or uploaded JD text
      questions         — list[str] of generated questions
      current_q_index   — which question we're on (0-based)
      answers           — list[str] answers (by index)
      evaluations       — list[dict] evaluations (by index)
      interview_started — bool; True once the interview has begun
      interview_done    — bool; True once all Qs are answered
      tts_enabled       — bool; from sidebar toggle
      whisper_model_size— str; from sidebar select
    """
    defaults = {
        "page": "setup",
        "session_key": str(uuid.uuid4()),
        "resume_text": "",
        "job_description": "",
        "interview_type": "Mixed",
        "num_questions": int(os.getenv("DEFAULT_QUESTIONS", "5")),
        "questions": [],
        "current_q_index": 0,
        "answers": [],
        "evaluations": [],
        "interview_started": False,
        "interview_done": False,
        "tts_enabled": os.getenv("TTS_ENABLED", "true").lower() == "true",
        "whisper_model_size": os.getenv("WHISPER_MODEL_SIZE", "base"),
        "final_report_md": "",
        "history_selected_key": None,
        # Per-question flags to prevent repeated TTS on rerun
        # Stored as "q_spoken_0", "q_spoken_1", etc.
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)


init_session_state()

MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "10"))


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with navigation and configuration controls."""
    with st.sidebar:
        st.markdown("## 🎤 Voice Interview\nSimulator")
        st.divider()

        # Navigation
        st.markdown("### Navigation")
        pages = {
            "🏠 Setup": "setup",
            "📋 Preview": "preview",
            "🎯 Interview": "interview",
            "📊 Report": "report",
            "📚 History": "history",
        }
        for label, page_id in pages.items():
            disabled = False
            if page_id == "preview" and not st.session_state.questions:
                disabled = True
            if page_id == "interview" and not st.session_state.interview_started:
                disabled = True
            if page_id == "report" and not st.session_state.interview_done:
                disabled = True
            if st.button(label, use_container_width=True, disabled=disabled, key=f"nav_{page_id}"):
                st.session_state.page = page_id
                st.rerun()

        st.divider()

        # Configuration
        st.markdown("### ⚙️ Configuration")
        st.caption(f"**Model:** `{engine.OLLAMA_MODEL}`")
        st.caption(f"**Ollama URL:** `{engine.OLLAMA_BASE_URL}`")

        # TTS toggle
        tts_toggle = st.toggle(
            "🔊 Text-to-Speech",
            value=st.session_state.tts_enabled,
            help="Read questions aloud. Requires espeak-ng on Linux.",
        )
        if tts_toggle != st.session_state.tts_enabled:
            st.session_state.tts_enabled = tts_toggle

        # Whisper model size
        whisper_options = ["tiny", "base", "small", "medium"]
        whisper_idx = whisper_options.index(st.session_state.whisper_model_size) \
            if st.session_state.whisper_model_size in whisper_options else 1
        new_whisper = st.selectbox(
            "🎙️ Whisper Model",
            whisper_options,
            index=whisper_idx,
            help="Larger = more accurate but slower. 'base' is recommended for most laptops.",
        )
        if new_whisper != st.session_state.whisper_model_size:
            st.session_state.whisper_model_size = new_whisper

        st.divider()

        # Ollama health quick-check
        if st.button("🔍 Check Ollama", use_container_width=True):
            health = engine.check_ollama_health()
            if health["ollama_running"]:
                st.success("✅ Ollama is running")
                if health["model_available"]:
                    st.success(f"✅ Model `{health['model_name']}` found")
                else:
                    st.error(
                        f"❌ Model `{health['model_name']}` not found.\n\n"
                        f"Pull it: `ollama pull {health['model_name']}`\n\n"
                        f"Available: {', '.join(health['available_models'])}"
                    )
            else:
                st.error(f"❌ {health['error']}")


render_sidebar()


# ── Page: Setup ───────────────────────────────────────────────────────────────
def page_setup():
    """Render the interview setup page.

    Handles:
      - Resume upload (PDF/DOCX) with text extraction
      - Job description input (text area or TXT file upload)
      - Interview type and question count selection
      - Validation and question generation
    """
    st.markdown("""
    <div class="main-header">
        <h1>🎤 Voice Interview Simulator</h1>
        <p>Prepare for your next interview with AI-powered personalised practice</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    # ── Left column: Resume ──────────────────────────────────────────────────
    with col1:
        st.markdown("### 📄 Your Resume")
        uploaded_resume = st.file_uploader(
            "Upload resume (PDF or DOCX)",
            type=["pdf", "docx"],
            help="Your resume text is used only during this session to generate questions.",
            key="resume_uploader",
        )

        if uploaded_resume is not None:
            with st.spinner("Extracting resume text…"):
                try:
                    text = engine.extract_resume_text(uploaded_resume)
                    st.session_state.resume_text = text
                    st.success(f"✅ Extracted {len(text)} characters from resume.")
                    with st.expander("Preview extracted text"):
                        st.text(text[:800] + ("…" if len(text) > 800 else ""))
                except (ValueError, RuntimeError) as e:
                    st.error(f"❌ {e}")
                    st.session_state.resume_text = ""

    # ── Right column: Job Description ────────────────────────────────────────
    with col2:
        st.markdown("### 💼 Job Description")
        jd_file = st.file_uploader(
            "Upload JD as .txt (optional)",
            type=["txt"],
            help="If you upload a file, it will be used instead of the text area.",
            key="jd_uploader",
        )

        if jd_file is not None:
            jd_from_file = jd_file.read().decode("utf-8", errors="replace")
            st.session_state.job_description = jd_from_file[:3000]
            st.info("✅ Job description loaded from file.")

        jd_text = st.text_area(
            "Paste job description here",
            value=st.session_state.job_description,
            height=220,
            placeholder="Paste the full job description including required skills, responsibilities, and qualifications…",
            key="jd_textarea",
        )
        st.session_state.job_description = jd_text

    st.divider()

    # ── Interview configuration ───────────────────────────────────────────────
    st.markdown("### 🎯 Interview Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

    with cfg_col1:
        interview_type = st.selectbox(
            "Interview Type",
            ["Mixed", "Technical", "Project-based", "HR/Behavioral"],
            index=["Mixed", "Technical", "Project-based", "HR/Behavioral"].index(
                st.session_state.interview_type
            ),
            help="This determines the style and focus of the questions.",
        )
        st.session_state.interview_type = interview_type

    with cfg_col2:
        num_q = st.slider(
            "Number of Questions",
            min_value=3,
            max_value=MAX_QUESTIONS,
            value=st.session_state.num_questions,
            help="More questions = longer session. 5 is recommended for a demo.",
        )
        st.session_state.num_questions = num_q

    with cfg_col3:
        st.markdown("**Tips:**")
        st.caption("• Use a clear, up-to-date resume for best questions.")
        st.caption("• Include the full JD for targeted questions.")
        st.caption("• Start with 5 questions for a quick practice run.")

    st.divider()

    # ── Validation and generation ─────────────────────────────────────────────
    can_generate = bool(
        st.session_state.resume_text and st.session_state.job_description.strip()
    )

    if not can_generate:
        st.info("👆 Upload your resume and paste a job description to get started.")

    generate_btn = st.button(
        "🚀 Generate Interview Questions",
        disabled=not can_generate,
        type="primary",
        use_container_width=True,
    )

    if generate_btn:
        _generate_questions()


def _generate_questions():
    """Call the LLM to generate questions and transition to the preview page."""
    # Start a fresh session
    st.session_state.session_key = str(uuid.uuid4())
    st.session_state.answers = []
    st.session_state.evaluations = []
    st.session_state.current_q_index = 0
    st.session_state.interview_started = False
    st.session_state.interview_done = False
    st.session_state.final_report_md = ""

    # Clear any per-question TTS flags
    for k in list(st.session_state.keys()):
        if k.startswith("q_spoken_"):
            del st.session_state[k]

    with st.spinner(
        f"🤖 Generating {st.session_state.num_questions} personalised questions… "
        f"(using {engine.OLLAMA_MODEL})"
    ):
        # Check Ollama first
        health = engine.check_ollama_health()
        if not health["ollama_running"]:
            st.error(f"❌ {health['error']}")
            return
        if not health["model_available"]:
            st.error(
                f"❌ Model `{health['model_name']}` is not available in Ollama.\n\n"
                f"Run: `ollama pull {health['model_name']}`\n\n"
                f"Available models: {', '.join(health['available_models'])}"
            )
            return

        try:
            questions = engine.generate_questions(
                resume_text=st.session_state.resume_text,
                job_description=st.session_state.job_description,
                interview_type=st.session_state.interview_type,
                num_questions=st.session_state.num_questions,
            )
            st.session_state.questions = questions

            # Create the database session record
            database.create_session(
                session_key=st.session_state.session_key,
                interview_type=st.session_state.interview_type,
                target_role=_extract_role_from_jd(st.session_state.job_description),
                num_questions=len(questions),
            )

            st.session_state.page = "preview"
            st.rerun()
        except (RuntimeError, ValueError) as e:
            st.error(f"❌ Question generation failed: {e}")


def _extract_role_from_jd(jd: str) -> str:
    """Heuristically extract a job title from the first 200 chars of the JD."""
    first_line = jd.strip().split("\n")[0][:100].strip()
    return first_line if first_line else "Unknown Role"


# ── Page: Preview ─────────────────────────────────────────────────────────────
def page_preview():
    """Show the generated questions so the candidate can review before starting."""
    st.markdown("## 📋 Interview Preview")

    if not st.session_state.questions:
        st.warning("No questions generated yet. Go to Setup first.")
        return

    st.markdown(
        f"**{len(st.session_state.questions)} questions** generated for a "
        f"**{st.session_state.interview_type}** interview. Review them below, "
        f"then click **Start Interview** when ready."
    )
    st.info(
        "💡 Questions are based on your resume and job description. "
        "You won't see these during the interview — treat this as a brief mental preparation."
    )

    for i, q in enumerate(st.session_state.questions, 1):
        with st.expander(f"Q{i}: {q[:60]}…" if len(q) > 60 else f"Q{i}: {q}"):
            st.write(q)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back to Setup", use_container_width=True):
            st.session_state.page = "setup"
            st.rerun()
    with col2:
        if st.button("🎯 Start Interview!", type="primary", use_container_width=True):
            st.session_state.interview_started = True
            st.session_state.current_q_index = 0
            st.session_state.page = "interview"
            st.rerun()


# ── Page: Interview ───────────────────────────────────────────────────────────
def page_interview():
    """Main interview page — one question at a time.

    For each question:
      1. Display the question card.
      2. (Optionally) speak the question via TTS — gated by a session-state flag
         so it doesn't replay on every rerun.
      3. Show the microphone recorder (streamlit-mic-recorder).
      4. If audio is captured, transcribe it with Whisper.
      5. Let the user edit the transcript.
      6. On submission, evaluate with Ollama and save to DB.
      7. Advance to the next question (or finish).
    """
    if not st.session_state.questions:
        st.warning("No interview in progress. Please go to Setup first.")
        return

    questions = st.session_state.questions
    idx = st.session_state.current_q_index
    total = len(questions)

    if idx >= total:
        # All questions answered — generate report
        _finish_interview()
        return

    # ── Progress bar ─────────────────────────────────────────────────────────
    st.markdown(f"### 🎯 Question {idx + 1} of {total}")
    st.progress((idx) / total)

    # ── Question card ─────────────────────────────────────────────────────────
    q_text = questions[idx]
    st.markdown(f"""
    <div class="question-card">
        <h4 style="color:#a78bfa; margin-bottom:0.5rem;">Question {idx+1}</h4>
        <p style="font-size:1.1rem; line-height:1.6;">{q_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── TTS: speak question once per question index ────────────────────────────
    # We use a flag "q_spoken_<idx>" so the question is spoken only once,
    # not on every Streamlit rerun triggered by widget interaction.
    tts_flag_key = f"q_spoken_{idx}"
    if st.session_state.tts_enabled and not st.session_state.get(tts_flag_key, False):
        if speech.is_tts_available():
            spoken = speech.speak_text(q_text)
            if spoken:
                st.session_state[tts_flag_key] = True
        else:
            st.session_state[tts_flag_key] = True  # don't keep trying

    # ── Answer input section ──────────────────────────────────────────────────
    st.markdown("#### 🎙️ Your Answer")

    # Mode: microphone OR text fallback
    answer_mode = st.radio(
        "Answer mode:",
        ["🎙️ Microphone (Voice)", "⌨️ Text (Fallback)"],
        horizontal=True,
        key=f"mode_{idx}",
    )

    transcript_from_audio = ""
    audio_error = None

    if "Microphone" in answer_mode:
        # Try to load streamlit-mic-recorder
        try:
            from streamlit_mic_recorder import mic_recorder

            st.caption(
                "🔴 Click **Start** to record, **Stop** when done. "
                "Allow microphone access in your browser when prompted."
            )
            audio_result = mic_recorder(
                start_prompt="▶ Start Recording",
                stop_prompt="⏹ Stop Recording",
                just_once=True,
                use_container_width=True,
                key=f"mic_{idx}",
            )

            if audio_result and audio_result.get("bytes"):
                audio_bytes = audio_result["bytes"]
                with st.spinner("🔄 Transcribing audio…"):
                    try:
                        transcript_from_audio = speech.transcribe_audio(
                            audio_bytes=audio_bytes,
                            model_size=st.session_state.whisper_model_size,
                        )
                    except (RuntimeError, ValueError) as e:
                        audio_error = str(e)
                        transcript_from_audio = ""

        except ImportError:
            st.warning(
                "⚠️ **streamlit-mic-recorder** is not installed. "
                "Using text fallback.\n\n"
                "Install with: `pip install streamlit-mic-recorder`"
            )
            answer_mode = "⌨️ Text (Fallback)"

    # Show transcription error if it occurred
    if audio_error:
        st.error(f"⚠️ Transcription error: {audio_error}")
        st.caption("Please use the text fallback below or try recording again.")

    # ── Editable transcript / text area ───────────────────────────────────────
    if transcript_from_audio:
        st.success("✅ Audio transcribed. You can edit the text below before submitting.")

    default_text = transcript_from_audio if transcript_from_audio else (
        st.session_state.answers[idx]
        if idx < len(st.session_state.answers) and st.session_state.answers[idx]
        else ""
    )

    final_answer = st.text_area(
        "Your answer (edit if needed):",
        value=default_text,
        height=180,
        placeholder="Speak your answer above, or type it here…",
        key=f"answer_text_{idx}",
    )

    st.divider()

    # ── Navigation buttons ────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

    with btn_col1:
        if idx > 0 and st.button("◀ Previous", key=f"prev_{idx}"):
            st.session_state.current_q_index = idx - 1
            st.rerun()

    with btn_col2:
        submit_label = "Submit & Next ▶" if idx < total - 1 else "Submit & Finish 🏁"
        if st.button(submit_label, type="primary", use_container_width=True, key=f"submit_{idx}"):
            _submit_answer(idx, final_answer, q_text)

    with btn_col3:
        if st.button("Skip ⏭", key=f"skip_{idx}"):
            _submit_answer(idx, "", q_text)

    # ── Show previous answers summary ─────────────────────────────────────────
    if st.session_state.evaluations:
        with st.expander(f"📊 Previous answers ({len(st.session_state.evaluations)} evaluated)"):
            for j, ev in enumerate(st.session_state.evaluations):
                if j < len(questions):
                    score = ev.get("total_score", 0)
                    bar = "🟢" if score >= 15 else ("🟡" if score >= 10 else "🔴")
                    st.markdown(
                        f"{bar} **Q{j+1}** — Score: **{score}/20** | "
                        f"{questions[j][:50]}…"
                    )


def _submit_answer(idx: int, answer_text: str, question_text: str):
    """Evaluate the answer and persist it, then advance to the next question.

    Args:
        idx:           Current question index.
        answer_text:   The final text answer (may be empty if skipped).
        question_text: The question that was asked.
    """
    with st.spinner("🤖 Evaluating your answer…"):
        try:
            evaluation = engine.evaluate_answer(
                question=question_text,
                answer=answer_text,
                resume_text=st.session_state.resume_text,
                job_description=st.session_state.job_description,
            )
        except (RuntimeError, ValueError) as e:
            st.error(f"❌ Evaluation failed: {e}")
            evaluation = _empty_evaluation()

    # Store answer and evaluation (extend lists if needed)
    while len(st.session_state.answers) <= idx:
        st.session_state.answers.append("")
    while len(st.session_state.evaluations) <= idx:
        st.session_state.evaluations.append({})

    st.session_state.answers[idx] = answer_text
    st.session_state.evaluations[idx] = evaluation

    # Persist to SQLite
    database.save_question(
        session_key=st.session_state.session_key,
        question_index=idx,
        question_text=question_text,
        answer_text=answer_text,
        evaluation=evaluation,
    )

    # Show quick feedback before advancing
    _show_evaluation_card(evaluation, question_text, idx)

    # Advance question index
    next_idx = idx + 1
    st.session_state.current_q_index = next_idx

    if next_idx >= len(st.session_state.questions):
        _finish_interview()
    else:
        st.rerun()


def _show_evaluation_card(evaluation: dict, question: str, idx: int):
    """Display a compact evaluation summary after answer submission."""
    score = evaluation.get("total_score", 0)
    color = "#3fb950" if score >= 15 else ("#d29922" if score >= 10 else "#f85149")

    st.markdown(f"""
    <div class="score-card">
        <h4 style="color:{color};">Score: {score}/20</h4>
        <table style="width:100%; font-size:0.85rem;">
            <tr>
                <td>Relevance: {evaluation.get('score_relevance', 0)}/5</td>
                <td>Technical: {evaluation.get('score_technical', 0)}/5</td>
                <td>Depth: {evaluation.get('score_depth', 0)}/5</td>
                <td>Clarity: {evaluation.get('score_clarity', 0)}/5</td>
            </tr>
        </table>
        <p><strong>✅ Did well:</strong> {evaluation.get('did_well', '—')}</p>
        <p><strong>⚠️ Missing:</strong> {evaluation.get('missing_concepts', '—')}</p>
        <p><strong>💡 Suggestion:</strong> {evaluation.get('suggestions', '—')}</p>
    </div>
    """, unsafe_allow_html=True)

    followup = evaluation.get("followup_q", "").strip()
    if followup:
        st.info(f"🔄 **Follow-up question:** {followup}")


def _empty_evaluation() -> dict:
    """Return a zero-score evaluation when Ollama fails."""
    return {
        "score_relevance": 0, "score_technical": 0,
        "score_depth": 0, "score_clarity": 0, "total_score": 0,
        "did_well": "Evaluation failed — Ollama may be unavailable.",
        "missing_concepts": "", "suggestions": "Check Ollama connectivity.",
        "example_answer": "", "followup_q": "",
    }


def _finish_interview():
    """Mark interview as done and transition to the report page."""
    st.session_state.interview_done = True
    st.session_state.page = "report"
    st.rerun()


# ── Page: Report ──────────────────────────────────────────────────────────────
def page_report():
    """Display the final interview report and offer a download."""
    st.markdown("## 📊 Interview Report")

    if not st.session_state.interview_done:
        st.info("Complete the interview first to see your report.")
        return

    evaluations = st.session_state.evaluations
    questions = st.session_state.questions
    answers = st.session_state.answers

    if not evaluations:
        st.warning("No evaluations found. The interview may not have been completed.")
        return

    # Generate report if not already done (e.g., on page rerun)
    if not st.session_state.final_report_md:
        with st.spinner("📝 Generating your personalised report…"):
            try:
                report_md = engine.generate_final_report(
                    interview_type=st.session_state.interview_type,
                    questions=questions,
                    answers=answers,
                    evaluations=evaluations,
                    resume_text=st.session_state.resume_text,
                    job_description=st.session_state.job_description,
                )
            except Exception as e:
                logger.error("Report generation error: %s", e)
                report_md = engine._build_report_header(
                    st.session_state.interview_type, questions, evaluations,
                    sum(e.get("total_score", 0) for e in evaluations) / max(len(evaluations), 1)
                )

            st.session_state.final_report_md = report_md

            # Save to DB
            avg = sum(ev.get("total_score", 0) for ev in evaluations) / max(len(evaluations), 1)
            database.save_summary(
                session_key=st.session_state.session_key,
                avg_score=avg,
                summary_md=report_md,
            )

    # Render the report
    st.markdown(st.session_state.final_report_md)

    st.divider()

    # Download button
    report_bytes = st.session_state.final_report_md.encode("utf-8")
    filename = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    st.download_button(
        label="⬇️ Download Report as Markdown",
        data=report_bytes,
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()

    # Start a new interview
    if st.button("🔄 Start New Interview", use_container_width=True):
        _reset_interview()
        st.session_state.page = "setup"
        st.rerun()


def _reset_interview():
    """Clear interview data from session state for a fresh start."""
    keys_to_reset = [
        "questions", "answers", "evaluations",
        "interview_started", "interview_done",
        "current_q_index", "final_report_md",
        "session_key",
    ]
    for k in keys_to_reset:
        if k in st.session_state:
            del st.session_state[k]

    # Clear TTS flags
    for k in list(st.session_state.keys()):
        if k.startswith("q_spoken_"):
            del st.session_state[k]

    # Re-init defaults
    init_session_state()


# ── Page: History ─────────────────────────────────────────────────────────────
def page_history():
    """Display past interview sessions from SQLite."""
    st.markdown("## 📚 Interview History")

    sessions = database.get_all_sessions()

    if not sessions:
        st.info("No interview history yet. Complete your first interview to see it here.")
        return

    st.markdown(f"Found **{len(sessions)}** past session(s).")

    for sess in sessions:
        created = sess.get("created_at", "")[:16].replace("T", " ")
        avg = sess.get("avg_score")
        avg_str = f"{avg:.1f}/20" if avg is not None else "N/A"
        role = sess.get("target_role", "Unknown") or "Unknown"

        with st.expander(
            f"🗓 {created} — {sess['interview_type']} — {role} — Avg: {avg_str}"
        ):
            detail = database.get_session_detail(sess["session_key"])
            if not detail:
                st.error("Could not load session details.")
                continue

            # Show report if saved
            report = detail["session"].get("summary_md")
            if report:
                st.markdown(report)
                report_bytes = report.encode("utf-8")
                st.download_button(
                    "⬇️ Download Report",
                    data=report_bytes,
                    file_name=f"report_{sess['session_key'][:8]}.md",
                    mime="text/markdown",
                    key=f"dl_{sess['session_key']}",
                )
            else:
                # Show Q&A table
                qs = detail["questions"]
                if qs:
                    for q_row in qs:
                        st.markdown(
                            f"**Q{q_row['question_index']+1}:** {q_row['question_text']}\n\n"
                            f"*Answer:* {(q_row['answer_text'] or '—')[:200]}\n\n"
                            f"*Score:* {q_row['total_score'] or 0}/20"
                        )
                        st.divider()

            # Delete button
            if st.button(
                "🗑 Delete Session",
                key=f"del_{sess['session_key']}",
                help="Permanently delete this session and all its data.",
            ):
                database.delete_session(sess["session_key"])
                st.success("Session deleted.")
                st.rerun()


# ── Page routing ──────────────────────────────────────────────────────────────
def main():
    """Route to the correct page based on session state."""
    page = st.session_state.get("page", "setup")

    if page == "setup":
        page_setup()
    elif page == "preview":
        page_preview()
    elif page == "interview":
        page_interview()
    elif page == "report":
        page_report()
    elif page == "history":
        page_history()
    else:
        page_setup()


main()
