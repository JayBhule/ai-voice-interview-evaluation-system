"""
database.py — SQLite persistence layer for the Voice Interview Simulator.

This module is responsible for:
  - Creating the database and required tables on first run (called by app.py at startup).
  - Saving new interview sessions when an interview begins.
  - Storing each question, the candidate's answer, and the LLM evaluation.
  - Saving the final summary report generated at the end of an interview.
  - Retrieving historical sessions for the Interview History view.

All SQL uses parameterised queries (? placeholders) to prevent injection.
The LLM output is parsed and validated BEFORE being stored here — never
execute SQL constructed from untrusted LLM text.

Data flow:
  app.py → database.create_session()     → returns session_id
  app.py → database.save_question()      → stores Q+A+evaluation
  app.py → database.save_summary()       → stores final report text
  app.py → database.get_all_sessions()   → returns history list
  app.py → database.get_session_detail() → returns full session data
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# The database file lives in the data/ subdirectory so it can be easily
# excluded from git via .gitignore without affecting source files.
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "interviews.db")


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection.

    Using check_same_thread=False is safe here because Streamlit reruns
    happen on the same OS thread within a session. Each call gets its own
    connection; we close it after every transaction.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # rows accessible as dicts
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    return conn


def init_db() -> None:
    """Create all required tables if they do not already exist.

    Called once at Streamlit startup (inside app.py). Safe to call multiple
    times — CREATE TABLE IF NOT EXISTS is idempotent.

    Schema overview:
      sessions  — one row per interview session
      questions — one row per question (linked to a session)
    """
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT    NOT NULL UNIQUE,   -- Streamlit session UUID
                created_at  TEXT    NOT NULL,
                interview_type TEXT NOT NULL,
                target_role TEXT,
                num_questions INTEGER NOT NULL,
                avg_score   REAL,
                summary_md  TEXT    -- final report Markdown
            );

            CREATE TABLE IF NOT EXISTS questions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key     TEXT    NOT NULL,
                question_index  INTEGER NOT NULL,
                question_text   TEXT    NOT NULL,
                answer_text     TEXT,
                score_relevance INTEGER,
                score_technical INTEGER,
                score_depth     INTEGER,
                score_clarity   INTEGER,
                total_score     INTEGER,
                did_well        TEXT,
                missing_concepts TEXT,
                suggestions     TEXT,
                example_answer  TEXT,
                followup_q      TEXT,
                evaluated_at    TEXT,
                FOREIGN KEY (session_key) REFERENCES sessions(session_key)
            );
        """)
        conn.commit()
        logger.info("Database initialised at %s", DB_PATH)
    except sqlite3.Error as e:
        logger.error("DB init failed: %s", e)
        raise
    finally:
        conn.close()


def create_session(
    session_key: str,
    interview_type: str,
    target_role: Optional[str],
    num_questions: int,
) -> bool:
    """Insert a new interview session row.

    Args:
        session_key:    A UUID string identifying the Streamlit session.
        interview_type: "Technical", "HR/Behavioral", "Project-based", or "Mixed".
        target_role:    Optional job title extracted from the job description.
        num_questions:  How many questions were generated.

    Returns:
        True on success, False on database error.
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_key, created_at, interview_type, target_role, num_questions)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_key, datetime.now().isoformat(), interview_type, target_role, num_questions),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error("create_session error: %s", e)
        return False
    finally:
        conn.close()


def save_question(
    session_key: str,
    question_index: int,
    question_text: str,
    answer_text: str,
    evaluation: dict,
) -> bool:
    """Persist one question, its answer, and the LLM evaluation.

    Args:
        session_key:    Links this row to the sessions table.
        question_index: 0-based index within the session.
        question_text:  The interview question shown to the candidate.
        answer_text:    The final (possibly edited) transcript submitted.
        evaluation:     Validated dict from interview_engine.evaluate_answer().
                        Expected keys: score_relevance, score_technical,
                        score_depth, score_clarity, total_score, did_well,
                        missing_concepts, suggestions, example_answer, followup_q.

    Returns:
        True on success, False on database error.
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO questions (
                session_key, question_index, question_text, answer_text,
                score_relevance, score_technical, score_depth, score_clarity,
                total_score, did_well, missing_concepts, suggestions,
                example_answer, followup_q, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_key,
                question_index,
                question_text,
                answer_text,
                evaluation.get("score_relevance"),
                evaluation.get("score_technical"),
                evaluation.get("score_depth"),
                evaluation.get("score_clarity"),
                evaluation.get("total_score"),
                evaluation.get("did_well"),
                evaluation.get("missing_concepts"),
                evaluation.get("suggestions"),
                evaluation.get("example_answer"),
                evaluation.get("followup_q"),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error("save_question error: %s", e)
        return False
    finally:
        conn.close()


def save_summary(session_key: str, avg_score: float, summary_md: str) -> bool:
    """Update the session row with the final report.

    Args:
        session_key: Identifies the session to update.
        avg_score:   Mean score across all evaluated questions.
        summary_md:  Full Markdown text of the interview report.

    Returns:
        True on success, False on database error.
    """
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE sessions SET avg_score=?, summary_md=? WHERE session_key=?",
            (avg_score, summary_md, session_key),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error("save_summary error: %s", e)
        return False
    finally:
        conn.close()


def get_all_sessions() -> list[dict]:
    """Retrieve all sessions ordered by most recent first.

    Returns a list of dicts with keys: id, session_key, created_at,
    interview_type, target_role, num_questions, avg_score.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, session_key, created_at, interview_type,
                   target_role, num_questions, avg_score
            FROM sessions
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.error("get_all_sessions error: %s", e)
        return []
    finally:
        conn.close()


def get_session_detail(session_key: str) -> dict:
    """Return one session's metadata and all its questions.

    Returns:
        {
          "session": {...},           # sessions row
          "questions": [{...}, ...]   # questions rows ordered by index
        }
        or {} on error / not found.
    """
    conn = _get_connection()
    try:
        session_row = conn.execute(
            "SELECT * FROM sessions WHERE session_key=?", (session_key,)
        ).fetchone()
        if not session_row:
            return {}
        question_rows = conn.execute(
            "SELECT * FROM questions WHERE session_key=? ORDER BY question_index",
            (session_key,),
        ).fetchall()
        return {
            "session": dict(session_row),
            "questions": [dict(q) for q in question_rows],
        }
    except sqlite3.Error as e:
        logger.error("get_session_detail error: %s", e)
        return {}
    finally:
        conn.close()


def delete_session(session_key: str) -> bool:
    """Delete a session and all its questions (cascade).

    Used by the History view's delete button.
    """
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM questions WHERE session_key=?", (session_key,))
        conn.execute("DELETE FROM sessions WHERE session_key=?", (session_key,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error("delete_session error: %s", e)
        return False
    finally:
        conn.close()
