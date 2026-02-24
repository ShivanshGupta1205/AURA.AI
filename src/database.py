"""
SQLite metadata store + helpers for persisting artifacts and derived outputs.

Tables
──────
  users          – login credentials (bcrypt-hashed passwords)
  subjects       – one per course / class the student is taking
  lectures       – one per recorded session, linked to a subject
  usage_logs     – tracks every OpenAI call (tokens, cost, action)
"""

from __future__ import annotations

import sqlite3
import hashlib
import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "aura.db"


# ────────────────────────────────────────────────────────────────────────────
# Connection helper
# ────────────────────────────────────────────────────────────────────────────

def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db(db_path: Path = DB_PATH):
    conn = _get_conn(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ────────────────────────────────────────────────────────────────────────────
# Schema creation
# ────────────────────────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH):
    """Create all tables if they do not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_db(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id),
                name          TEXT    NOT NULL,
                syllabus_path TEXT,
                created_at    TEXT    DEFAULT (datetime('now')),
                UNIQUE(user_id, name)
            );

            CREATE TABLE IF NOT EXISTS lectures (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id         INTEGER NOT NULL REFERENCES subjects(id),
                title              TEXT    NOT NULL,
                date               TEXT    DEFAULT (date('now')),
                duration_seconds   REAL,
                transcript_path    TEXT,
                short_summary_path TEXT,
                long_summary_path  TEXT,
                created_at         TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS exam_papers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id  INTEGER NOT NULL REFERENCES subjects(id),
                name        TEXT    NOT NULL,
                file_path   TEXT    NOT NULL,
                uploaded_at TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS usage_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER REFERENCES users(id),
                action          TEXT    NOT NULL,
                model           TEXT,
                prompt_tokens   INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                cost_usd        REAL    DEFAULT 0.0,
                created_at      TEXT    DEFAULT (datetime('now'))
            );
        """)


# ────────────────────────────────────────────────────────────────────────────
# Password helpers  (simple SHA-256 hash — adequate for a local-only POC)
# ────────────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# User CRUD
# ────────────────────────────────────────────────────────────────────────────

def create_user(username: str, password: str) -> int | None:
    """Create a new user. Returns user id or None if username exists."""
    try:
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, _hash_password(password)),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def authenticate_user(username: str, password: str) -> dict | None:
    """Return user row dict if credentials match, else None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username, _hash_password(password)),
        ).fetchone()
        return dict(row) if row else None


# ────────────────────────────────────────────────────────────────────────────
# Subject CRUD
# ────────────────────────────────────────────────────────────────────────────

def create_subject(user_id: int, name: str, syllabus_path: str | None = None) -> int | None:
    try:
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO subjects (user_id, name, syllabus_path) VALUES (?, ?, ?)",
                (user_id, name, syllabus_path),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_subjects(user_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM subjects WHERE user_id = ? ORDER BY name", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_subject(subject_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM lectures WHERE subject_id = ?", (subject_id,))
        conn.execute("DELETE FROM exam_papers WHERE subject_id = ?", (subject_id,))
        conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))


# ────────────────────────────────────────────────────────────────────────────
# Lecture CRUD
# ────────────────────────────────────────────────────────────────────────────

def create_lecture(
    subject_id: int,
    title: str,
    duration_seconds: float = 0.0,
    transcript_path: str | None = None,
    short_summary_path: str | None = None,
    long_summary_path: str | None = None,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO lectures
               (subject_id, title, duration_seconds,
                transcript_path, short_summary_path, long_summary_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (subject_id, title, duration_seconds,
             transcript_path, short_summary_path, long_summary_path),
        )
        return cur.lastrowid


def update_lecture(lecture_id: int, **kwargs):
    allowed = {
        "title", "duration_seconds", "transcript_path",
        "short_summary_path", "long_summary_path",
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [lecture_id]
    with get_db() as conn:
        conn.execute(
            f"UPDATE lectures SET {set_clause} WHERE id = ?", values
        )


def get_lectures(subject_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM lectures WHERE subject_id = ? ORDER BY created_at DESC",
            (subject_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_lecture(lecture_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM lectures WHERE id = ?", (lecture_id,)
        ).fetchone()
        return dict(row) if row else None


# ────────────────────────────────────────────────────────────────────────────
# Exam paper CRUD (frontend-only for now)
# ────────────────────────────────────────────────────────────────────────────

def add_exam_paper(subject_id: int, name: str, file_path: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO exam_papers (subject_id, name, file_path) VALUES (?, ?, ?)",
            (subject_id, name, file_path),
        )
        return cur.lastrowid


def get_exam_papers(subject_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM exam_papers WHERE subject_id = ? ORDER BY uploaded_at DESC",
            (subject_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ────────────────────────────────────────────────────────────────────────────
# Usage / logs
# ────────────────────────────────────────────────────────────────────────────

# Pricing per 1M tokens (gpt-4o-mini as of early 2026)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def log_usage(
    user_id: int,
    action: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
):
    """Log an API call and compute estimated cost."""
    total = prompt_tokens + completion_tokens
    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000

    with get_db() as conn:
        conn.execute(
            """INSERT INTO usage_logs
               (user_id, action, model, prompt_tokens, completion_tokens,
                total_tokens, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, action, model, prompt_tokens, completion_tokens, total, cost),
        )


def get_usage_stats(user_id: int) -> dict:
    """Aggregate usage stats for the dashboard header."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT
                COALESCE(SUM(prompt_tokens), 0)      AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0)   AS total_completion_tokens,
                COALESCE(SUM(total_tokens), 0)        AS total_tokens,
                COALESCE(SUM(cost_usd), 0.0)          AS total_cost,
                COUNT(*)                               AS total_api_calls
               FROM usage_logs WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        stats = dict(row)

        # Count lectures and summaries
        lec_row = conn.execute(
            """SELECT
                COUNT(*)                                         AS total_lectures,
                COUNT(transcript_path)                           AS total_transcriptions,
                COUNT(short_summary_path)                        AS total_summaries,
                COALESCE(SUM(l.duration_seconds), 0.0)           AS total_duration
               FROM lectures l
               JOIN subjects s ON l.subject_id = s.id
               WHERE s.user_id = ?""",
            (user_id,),
        ).fetchone()
        stats.update(dict(lec_row))

        return stats
