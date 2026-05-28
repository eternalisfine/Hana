# memory.py — Persistent storage: conversation history, mistakes, style profile

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "japanese_tutor.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                role            TEXT    NOT NULL,  -- 'user' | 'assistant'
                content         TEXT    NOT NULL,
                flagged         INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS mistakes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                original        TEXT    NOT NULL,
                correction      TEXT    NOT NULL,
                mistake_type    TEXT    DEFAULT 'general',
                occurrences     INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS style_profile (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                updated_at      TEXT,
                level_estimate  TEXT    DEFAULT 'beginner',
                grammar_notes   TEXT    DEFAULT '',
                vocabulary_notes TEXT   DEFAULT '',
                general_notes   TEXT    DEFAULT ''
            );
        """)
        c.execute(
            "INSERT OR IGNORE INTO style_profile (id, updated_at) VALUES (1, ?)",
            (datetime.now().isoformat(),)
        )


# ── Messages ──────────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str, flagged: bool = False):
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, timestamp, role, content, flagged) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, datetime.now().isoformat(), role, content, int(flagged))
        )


def get_recent_messages(limit: int = 24) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_user_message_count() -> int:
    with _conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM messages WHERE role='user'"
        ).fetchone()[0]


# ── Mistakes ──────────────────────────────────────────────────────────────────

def log_mistake(original: str, correction: str, mistake_type: str = "general"):
    with _conn() as c:
        existing = c.execute(
            "SELECT id, occurrences FROM mistakes WHERE original=? AND correction=?",
            (original, correction)
        ).fetchone()
        if existing:
            c.execute(
                "UPDATE mistakes SET occurrences=? WHERE id=?",
                (existing["occurrences"] + 1, existing["id"])
            )
        else:
            c.execute(
                "INSERT INTO mistakes (timestamp, original, correction, mistake_type) "
                "VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), original, correction, mistake_type)
            )


def get_mistake_summary(limit: int = 12) -> str:
    with _conn() as c:
        rows = c.execute(
            "SELECT original, correction, mistake_type, occurrences "
            "FROM mistakes ORDER BY occurrences DESC LIMIT ?", (limit,)
        ).fetchall()
    if not rows:
        return "No recorded mistakes yet."
    lines = []
    for r in rows:
        lines.append(
            f"  • Said: 「{r['original']}」 → Correct: 「{r['correction']}」"
            f" [{r['mistake_type']}] (×{r['occurrences']})"
        )
    return "\n".join(lines)


# ── Style Profile ─────────────────────────────────────────────────────────────

def update_style_profile(**kwargs):
    allowed = {"level_estimate", "grammar_notes", "vocabulary_notes", "general_notes"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v}
    if not updates:
        return
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [datetime.now().isoformat()]
    with _conn() as c:
        c.execute(
            f"UPDATE style_profile SET {set_clause}, updated_at=? WHERE id=1", values
        )


def get_style_profile() -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM style_profile WHERE id=1").fetchone()
    return dict(row) if row else {}


def build_context_block() -> str:
    """Formatted block injected into every system prompt."""
    profile = get_style_profile()
    mistakes = get_mistake_summary()
    total = get_user_message_count()

    level = profile.get("level_estimate", "beginner")
    grammar = profile.get("grammar_notes", "")
    vocab = profile.get("vocabulary_notes", "")
    general = profile.get("general_notes", "")

    return f"""
=== Student Profile ===
Total exchanges: {total}
Estimated level: {level}
Grammar notes: {grammar or 'None yet'}
Vocabulary notes: {vocab or 'None yet'}
General notes: {general or 'None yet'}

=== Recurring Mistakes ===
{mistakes}
""".strip()