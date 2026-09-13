# memory.py — Persistent storage: conversation history, mistakes, style profile

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "japanese_tutor.db"


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH, timeout=20.0)
    c.row_factory = sqlite3.Row
    try:
        with c:
            yield c
    finally:
        c.close()


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

            CREATE TABLE IF NOT EXISTS vocabulary_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                word            TEXT    NOT NULL,
                reading         TEXT,
                pos             TEXT,
                first_seen      TEXT    NOT NULL,
                last_seen       TEXT    NOT NULL,
                times_seen      INTEGER DEFAULT 1,
                times_produced  INTEGER DEFAULT 0,
                times_corrected INTEGER DEFAULT 0,
                ease_factor     REAL    DEFAULT 2.5,
                interval_days   REAL    DEFAULT 0,
                next_review     TEXT,
                UNIQUE(word, reading)
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                date            TEXT    PRIMARY KEY,
                messages_sent   INTEGER DEFAULT 0,
                words_learned   INTEGER DEFAULT 0,
                minutes_active  REAL    DEFAULT 0,
                xp_earned       INTEGER DEFAULT 0
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


# ── Vocabulary Log ────────────────────────────────────────────────────────────

def upsert_vocabulary(word: str, reading: str = "", pos: str = "",
                      produced: bool = False, corrected: bool = False):
    """Insert or update a vocabulary entry. Called after each conversation turn."""
    now = datetime.now().isoformat()
    with _conn() as c:
        existing = c.execute(
            "SELECT id, times_seen, times_produced, times_corrected "
            "FROM vocabulary_log WHERE word=? AND reading=?",
            (word, reading or "")
        ).fetchone()

        if existing:
            updates = {
                "times_seen": existing["times_seen"] + 1,
                "last_seen": now,
            }
            if produced:
                updates["times_produced"] = existing["times_produced"] + 1
            if corrected:
                updates["times_corrected"] = existing["times_corrected"] + 1
            set_clause = ", ".join(f"{k}=?" for k in updates)
            values = list(updates.values()) + [existing["id"]]
            c.execute(f"UPDATE vocabulary_log SET {set_clause} WHERE id=?", values)
        else:
            c.execute(
                "INSERT INTO vocabulary_log "
                "(word, reading, pos, first_seen, last_seen, times_produced, times_corrected) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (word, reading or "", pos or "", now, now,
                 1 if produced else 0, 1 if corrected else 0)
            )


def get_known_vocabulary(min_exposures: int = 3) -> list[dict]:
    """Words the student likely 'knows' (seen enough times)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT word, reading, pos, times_seen, times_produced "
            "FROM vocabulary_log WHERE times_seen >= ? "
            "ORDER BY times_seen DESC",
            (min_exposures,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_weak_vocabulary(limit: int = 15) -> list[dict]:
    """Words seen but rarely produced, or frequently corrected."""
    with _conn() as c:
        rows = c.execute(
            "SELECT word, reading, pos, times_seen, times_produced, times_corrected "
            "FROM vocabulary_log "
            "WHERE (times_produced = 0 AND times_seen >= 2) "
            "   OR times_corrected > 0 "
            "ORDER BY times_corrected DESC, times_seen DESC "
            "LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_total_vocabulary_count() -> int:
    """Total unique words encountered."""
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM vocabulary_log").fetchone()[0]


def get_vocabulary_known_count(min_exposures: int = 3) -> int:
    """Count of words considered 'known'."""
    with _conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM vocabulary_log WHERE times_seen >= ?",
            (min_exposures,)
        ).fetchone()[0]


def get_recent_vocabulary(limit: int = 20) -> list[dict]:
    """Most recently encountered words."""
    with _conn() as c:
        rows = c.execute(
            "SELECT word, reading, pos, times_seen, times_produced, first_seen, last_seen "
            "FROM vocabulary_log ORDER BY last_seen DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Daily Stats & Streaks ─────────────────────────────────────────────────────

def log_daily_activity(messages: int = 0, words: int = 0,
                       minutes: float = 0, xp: int = 0):
    """Increment today's activity counters."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_stats (date, messages_sent, words_learned, minutes_active, xp_earned) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET "
            "  messages_sent = messages_sent + excluded.messages_sent, "
            "  words_learned = words_learned + excluded.words_learned, "
            "  minutes_active = minutes_active + excluded.minutes_active, "
            "  xp_earned = xp_earned + excluded.xp_earned",
            (today, messages, words, minutes, xp)
        )


def get_streak() -> int:
    """Calculate current consecutive-day streak."""
    with _conn() as c:
        rows = c.execute(
            "SELECT date FROM daily_stats WHERE messages_sent > 0 "
            "ORDER BY date DESC"
        ).fetchall()

    if not rows:
        return 0

    from datetime import timedelta
    streak = 0
    expected = datetime.now().date()

    for row in rows:
        day = datetime.strptime(row["date"], "%Y-%m-%d").date()
        if day == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif day == expected - timedelta(days=1):
            # Allow checking yesterday if today has no activity yet
            if streak == 0:
                expected = day
                streak = 1
                expected -= timedelta(days=1)
            else:
                break
        else:
            break

    return streak


def get_daily_stats_history(days: int = 30) -> list[dict]:
    """Get daily stats for the last N days."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?",
            (days,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_today_xp() -> int:
    """Get today's XP earned."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as c:
        row = c.execute(
            "SELECT xp_earned FROM daily_stats WHERE date=?", (today,)
        ).fetchone()
    return row["xp_earned"] if row else 0