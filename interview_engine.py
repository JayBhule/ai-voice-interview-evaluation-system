"""
interview_engine.py — LLM orchestration via local Ollama.

This module is the "brain" of the interview simulator. It:
  1. Checks Ollama connectivity and model availability.
  2. Generates interview questions from a resume + job description.
  3. Evaluates candidate answers on four dimensions (0–5 each, 20 total).
  4. Generates a personalised final report from all answers and scores.
  5. Extracts text from PDF and DOCX resume files.

All LLM calls go to the locally-running Ollama service — no cloud API is used.

Data flow:
  app.py → extract_resume_text()         → plain text
  app.py → check_ollama_health()         → status dict
  app.py → generate_questions()          → list[str]
  app.py → evaluate_answer()             → evaluation dict
  app.py → generate_final_report()       → Markdown string
"""

import json
import logging
import os
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()  # read .env if present

logger = logging.getLogger(__name__)

# ── Configuration (overridable via .env) ─────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "gemma2:2b")
REQUEST_TIMEOUT: int = 120   # seconds; generation can be slow on CPU


# ── Ollama connectivity helpers ───────────────────────────────────────────────

def check_ollama_health() -> dict:
    """Verify that Ollama is reachable and the configured model is available.

    Returns:
        {
          "ollama_running": bool,
          "model_available": bool,
          "model_name": str,
          "available_models": list[str],
          "error": str | None,
        }
    """
    result = {
        "ollama_running": False,
        "model_available": False,
        "model_name": OLLAMA_MODEL,
        "available_models": [],
        "error": None,
    }
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models_data = resp.json().get("models", [])
        result["ollama_running"] = True
        result["available_models"] = [m["name"] for m in models_data]

        # Check if our model (or a variant) is present.
        # Ollama names are like "gemma2:2b"; match by exact name or prefix.
        result["model_available"] = any(
            m["name"] == OLLAMA_MODEL or m["name"].startswith(OLLAMA_MODEL.split(":")[0])
            for m in models_data
        )
    except requests.ConnectionError:
        result["error"] = (
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running? Start it with: ollama serve"
        )
    except requests.HTTPError as e:
        result["error"] = f"Ollama returned HTTP {e.response.status_code}"
    except Exception as e:
        result["error"] = f"Unexpected error checking Ollama: {e}"

    return result


def _call_ollama(prompt: str, expect_json: bool = True) -> str:
    """Send a prompt to Ollama and return the response text.

    Args:
        prompt:      The full prompt string to send.
        expect_json: If True, the prompt instructs the model to return JSON.

    Returns:
        Raw response text (may be a JSON string or plain text).

    Raises:
        RuntimeError: On connection failure, HTTP error, or timeout.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,      # lower = more consistent/structured output
            "num_predict": 2048,     # max tokens in response
        },
    }
    if expect_json:
        payload["format"] = "json"   # Ollama-level JSON mode (model must support it)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        logger.debug("Ollama raw response (%d chars): %s …", len(raw), raw[:200])
        return raw
    except requests.ConnectionError:
        raise RuntimeError(
            "Lost connection to Ollama. Make sure it is running: ollama serve"
        )
    except requests.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after {REQUEST_TIMEOUT}s. "
            "Try a smaller model or reduce the number of questions."
        )
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error calling Ollama: {e}")


def _extract_json(text: str) -> dict | list:
    """Robustly extract a JSON object or array from a model response.

    Models sometimes wrap JSON in markdown code fences or add extra prose.
    This function tries several strategies before raising ValueError.

    Args:
        text: Raw string returned by the LLM.

    Returns:
        Parsed Python dict or list.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences (```json … ```)
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { … } or [ … ] block
    brace = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
    if brace:
        try:
            return json.loads(brace.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The model did not return valid JSON. Raw response:\n" + text[:500]
    )


# ── Resume text extraction ────────────────────────────────────────────────────

def extract_resume_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF or DOCX resume.

    Args:
        uploaded_file: A Streamlit UploadedFile object with a .name attribute
                       and .read() method.

    Returns:
        Extracted text as a UTF-8 string, truncated to 4000 characters to
        stay within reasonable LLM context limits.

    Raises:
        ValueError: For unsupported file types.
        RuntimeError: If extraction fails.
    """
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    try:
        if name.endswith(".pdf"):
            return _extract_pdf(file_bytes)
        elif name.endswith(".docx"):
            return _extract_docx(file_bytes)
        else:
            raise ValueError(
                f"Unsupported file type '{uploaded_file.name}'. "
                "Please upload a PDF or DOCX file."
            )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {uploaded_file.name}: {e}") from e


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import pymupdf as fitz   # PyMuPDF ≥1.24 preferred import
    except ImportError:
        import fitz               # fallback for older versions

    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise RuntimeError("The PDF appears to contain no extractable text (may be image-based).")
    # Limit to 4000 chars to keep prompts manageable.
    return full_text[:4000]


def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    import io
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs).strip()
    if not full_text:
        raise RuntimeError("The DOCX file contains no readable text.")
    return full_text[:4000]


# ── Question generation ───────────────────────────────────────────────────────

INTERVIEW_TYPE_INSTRUCTIONS = {
    "Technical": (
        "Focus on technical concepts, algorithms, data structures, system design, "
        "and hands-on coding skills mentioned in the resume."
    ),
    "Project-based": (
        "Focus on the candidate's specific projects, their design decisions, "
        "challenges they faced, and what they would improve. Reference projects "
        "mentioned in the resume directly."
    ),
    "HR/Behavioral": (
        "Focus on behavioural and situational questions (STAR format): teamwork, "
        "conflict resolution, leadership, time management, and career goals."
    ),
    "Mixed": (
        "Include a balanced mix of technical, project-based, and behavioural "
        "questions relevant to the candidate's background and the target role."
    ),
}


def generate_questions(
    resume_text: str,
    job_description: str,
    interview_type: str,
    num_questions: int,
) -> list[str]:
    """Generate personalised interview questions using the local LLM.

    The prompt explicitly instructs the model to:
      - Base questions on the supplied resume (not invented skills).
      - Target the job description requirements.
      - Return structured JSON so we can parse it reliably.

    Args:
        resume_text:      Extracted text from the candidate's resume (≤4000 chars).
        job_description:  Pasted or uploaded JD text (≤2000 chars).
        interview_type:   One of the keys in INTERVIEW_TYPE_INSTRUCTIONS.
        num_questions:    How many questions to generate (typically 3–10).

    Returns:
        A list of question strings.

    Raises:
        RuntimeError: On Ollama failure.
        ValueError:   If the model returns malformed JSON or too few questions.
    """
    type_hint = INTERVIEW_TYPE_INSTRUCTIONS.get(interview_type, INTERVIEW_TYPE_INSTRUCTIONS["Mixed"])
    jd_snippet = job_description[:2000]  # cap JD size

    prompt = f"""You are an expert technical interviewer. Generate exactly {num_questions} interview questions.

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_snippet}

INTERVIEW TYPE: {interview_type}
INSTRUCTION: {type_hint}

RULES:
- Base every question on skills, projects, or experience that ACTUALLY APPEAR in the resume above.
- Do NOT invent skills or experiences not mentioned.
- Questions must be relevant to the job description requirements.
- Vary difficulty: include conceptual, practical, and applied questions.
- Each question must be clear, specific, and answerable in 2-5 minutes.

Return ONLY valid JSON in this exact format:
{{
  "questions": [
    "Question 1 text here",
    "Question 2 text here"
  ]
}}
Do not include any explanation outside the JSON."""

    raw = _call_ollama(prompt, expect_json=True)
    data = _extract_json(raw)

    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(
            f"Model returned unexpected JSON structure. Expected {{\"questions\": [...]}}.\n"
            f"Got: {str(data)[:300]}"
        )

    questions = [q.strip() for q in data["questions"] if isinstance(q, str) and q.strip()]

    if len(questions) == 0:
        raise ValueError("The model returned an empty question list. Try again.")

    # Pad or trim to the requested count.
    if len(questions) < num_questions:
        logger.warning(
            "Model returned %d questions instead of %d.", len(questions), num_questions
        )
    questions = questions[:num_questions]

    logger.info("Generated %d questions for '%s' interview.", len(questions), interview_type)
    return questions


# ── Answer evaluation ─────────────────────────────────────────────────────────

SCORE_SCHEMA = {
    "score_relevance": (0, 5),
    "score_technical": (0, 5),
    "score_depth":     (0, 5),
    "score_clarity":   (0, 5),
}


def evaluate_answer(
    question: str,
    answer: str,
    resume_text: str,
    job_description: str,
) -> dict:
    """Evaluate a candidate's answer on four dimensions using the local LLM.

    Dimensions (0–5 each, 20 total):
      - Relevance:   Does the answer address the question?
      - Technical:   Is the technical content correct?
      - Depth:       Is it thorough and detailed?
      - Clarity:     Is it well-structured and clearly communicated?

    Args:
        question:         The interview question that was asked.
        answer:           The candidate's answer (possibly STT transcript).
        resume_text:      Used as context so the LLM can assess accuracy.
        job_description:  Used to benchmark what level of answer is expected.

    Returns:
        A validated dict with these keys:
          score_relevance, score_technical, score_depth, score_clarity,
          total_score, did_well, missing_concepts, suggestions,
          example_answer, followup_q

    Raises:
        RuntimeError: On Ollama failure.
        ValueError:   On malformed JSON from the model.
    """
    if not answer or not answer.strip():
        # Return a neutral non-penalising result for empty/failed transcripts.
        return {
            "score_relevance": 0,
            "score_technical": 0,
            "score_depth": 0,
            "score_clarity": 0,
            "total_score": 0,
            "did_well": "No answer was submitted.",
            "missing_concepts": "Unable to evaluate — no answer text.",
            "suggestions": "Please try answering again using the text fallback.",
            "example_answer": "",
            "followup_q": "",
        }

    prompt = f"""You are an expert interviewer providing structured, honest feedback on a candidate's answer.

QUESTION ASKED:
{question}

CANDIDATE'S ANSWER:
{answer}

CANDIDATE'S RESUME CONTEXT:
{resume_text[:1500]}

JOB DESCRIPTION CONTEXT:
{job_description[:800]}

TASK: Evaluate the answer on four dimensions, each scored 0–5:
  - score_relevance (0-5): How directly does it address the question?
  - score_technical (0-5): Is the technical content accurate and appropriate?
  - score_depth     (0-5): How thorough and detailed is the explanation?
  - score_clarity   (0-5): Is it well-structured and clearly communicated?

Scoring guide: 0=no answer/irrelevant, 1=very weak, 2=basic, 3=adequate, 4=good, 5=excellent.

Return ONLY valid JSON:
{{
  "score_relevance": <int 0-5>,
  "score_technical": <int 0-5>,
  "score_depth": <int 0-5>,
  "score_clarity": <int 0-5>,
  "total_score": <sum of above, 0-20>,
  "did_well": "<one or two sentences on what the candidate did well>",
  "missing_concepts": "<key concepts or points that were missing or incorrect>",
  "suggestions": "<specific, actionable suggestions to improve this answer>",
  "example_answer": "<a concise example of a stronger answer in 3-5 sentences>",
  "followup_q": "<one follow-up question to probe deeper, or empty string>"
}}"""

    raw = _call_ollama(prompt, expect_json=True)
    data = _extract_json(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Evaluation response is not a JSON object: {str(data)[:200]}")

    # Validate and clamp numeric scores.
    evaluation = {}
    total = 0
    for field, (lo, hi) in SCORE_SCHEMA.items():
        raw_val = data.get(field, 0)
        try:
            val = max(lo, min(hi, int(raw_val)))
        except (TypeError, ValueError):
            val = lo
        evaluation[field] = val
        total += val

    evaluation["total_score"] = data.get("total_score", total)
    # Ensure total_score is consistent with the sum of dimensions.
    try:
        evaluation["total_score"] = max(0, min(20, int(evaluation["total_score"])))
    except (TypeError, ValueError):
        evaluation["total_score"] = total

    # Overwrite with computed total if model's value differs significantly.
    if abs(evaluation["total_score"] - total) > 2:
        evaluation["total_score"] = total

    evaluation["did_well"]         = str(data.get("did_well", ""))
    evaluation["missing_concepts"] = str(data.get("missing_concepts", ""))
    evaluation["suggestions"]      = str(data.get("suggestions", ""))
    evaluation["example_answer"]   = str(data.get("example_answer", ""))
    evaluation["followup_q"]       = str(data.get("followup_q", ""))

    logger.info(
        "Evaluation complete. Total: %d/20 (R:%d T:%d D:%d C:%d)",
        evaluation["total_score"],
        evaluation["score_relevance"],
        evaluation["score_technical"],
        evaluation["score_depth"],
        evaluation["score_clarity"],
    )
    return evaluation


# ── Final report generation ───────────────────────────────────────────────────

def generate_final_report(
    interview_type: str,
    questions: list[str],
    answers: list[str],
    evaluations: list[dict],
    resume_text: str,
    job_description: str,
) -> str:
    """Generate a Markdown interview report from all questions and evaluations.

    This is called only once at the end of the interview. The report is built
    from the actual stored evaluations — no data is fabricated.

    Args:
        interview_type: The interview category.
        questions:      Ordered list of question texts.
        answers:        Ordered list of answer texts.
        evaluations:    Ordered list of evaluation dicts from evaluate_answer().
        resume_text:    For context in the final LLM prompt.
        job_description: For context.

    Returns:
        A Markdown string suitable for displaying in Streamlit or downloading.
    """
    if not evaluations:
        return "# Interview Report\n\nNo evaluations were completed."

    # Build a structured summary of all Q&A pairs.
    qa_summary_parts = []
    for i, (q, a, ev) in enumerate(zip(questions, answers, evaluations), 1):
        qa_summary_parts.append(
            f"Q{i}: {q}\n"
            f"Answer: {a[:300] if a else '(no answer)'}\n"
            f"Score: {ev.get('total_score', 0)}/20 "
            f"(R:{ev.get('score_relevance',0)} T:{ev.get('score_technical',0)} "
            f"D:{ev.get('score_depth',0)} C:{ev.get('score_clarity',0)})\n"
            f"Did well: {ev.get('did_well','')}\n"
            f"Missing: {ev.get('missing_concepts','')}"
        )
    qa_summary = "\n\n".join(qa_summary_parts)

    avg_score = sum(ev.get("total_score", 0) for ev in evaluations) / max(len(evaluations), 1)

    prompt = f"""You are an expert career coach summarising an interview practice session.

INTERVIEW TYPE: {interview_type}
TOTAL QUESTIONS: {len(questions)}
AVERAGE SCORE: {avg_score:.1f}/20

QUESTION-BY-QUESTION SUMMARY:
{qa_summary}

JOB DESCRIPTION TARGET ROLE:
{job_description[:500]}

Write a structured, honest, and encouraging interview performance report in Markdown.
Include these sections:
1. **Overall Performance** — summary and average score
2. **Strongest Areas** — what the candidate consistently did well
3. **Topics Needing Improvement** — specific weak areas from the answers
4. **Key Missing Concepts** — important concepts that were not covered
5. **Personalised Recommendations** — 3-5 specific, actionable steps to improve
6. **Suggested Practice Questions** — 3 questions targeting weak topics

Be specific and honest. Reference actual scores and content from the answers above.
Do NOT fabricate information. Do NOT be generic. This is AI-generated practice feedback."""

    try:
        raw = _call_ollama(prompt, expect_json=False)
        report_body = raw.strip()
    except RuntimeError as e:
        # Fallback: build a basic report from the stored data.
        report_body = _fallback_report(interview_type, questions, evaluations, avg_score)

    # Prepend a header with factual stats.
    header = _build_report_header(interview_type, questions, evaluations, avg_score)
    return header + "\n\n" + report_body


def _build_report_header(
    interview_type: str,
    questions: list[str],
    evaluations: list[dict],
    avg_score: float,
) -> str:
    """Build the top section of the report from stored data (no LLM)."""
    from datetime import datetime

    lines = [
        f"# 🎤 Interview Practice Report",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Date | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
        f"| Interview Type | {interview_type} |",
        f"| Questions Completed | {len(evaluations)}/{len(questions)} |",
        f"| Average Score | **{avg_score:.1f} / 20** |",
        f"",
        f"## 📊 Question Scores",
        f"",
        f"| # | Score | Relevance | Technical | Depth | Clarity |",
        f"|---|---|---|---|---|---|",
    ]
    for i, ev in enumerate(evaluations, 1):
        lines.append(
            f"| Q{i} | {ev.get('total_score',0)}/20 | "
            f"{ev.get('score_relevance',0)}/5 | "
            f"{ev.get('score_technical',0)}/5 | "
            f"{ev.get('score_depth',0)}/5 | "
            f"{ev.get('score_clarity',0)}/5 |"
        )
    return "\n".join(lines)


def _fallback_report(
    interview_type: str,
    questions: list[str],
    evaluations: list[dict],
    avg_score: float,
) -> str:
    """Simple text report built purely from stored evaluations (no LLM call)."""
    parts = [
        f"\n## Overall Performance\n",
        f"You completed {len(evaluations)} {interview_type} questions "
        f"with an average score of {avg_score:.1f}/20.\n",
        f"\n## Individual Feedback\n",
    ]
    for i, (q, ev) in enumerate(zip(questions, evaluations), 1):
        parts.append(
            f"**Q{i}:** {q}\n"
            f"- Score: {ev.get('total_score',0)}/20\n"
            f"- Did well: {ev.get('did_well','—')}\n"
            f"- Suggestions: {ev.get('suggestions','—')}\n"
        )
    return "\n".join(parts)
