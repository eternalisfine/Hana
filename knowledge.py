# knowledge.py — Vocabulary & grammar extraction from conversation using GiNZA/spaCy

"""
Extracts vocabulary and grammar patterns from conversation text and tracks
them in the database. This is the intelligence layer that makes Hana aware
of what the student has learned through conversation.
"""

import re
from typing import Optional

import memory

import nlp_core


# ── Japanese text detection ───────────────────────────────────────────────────

_JP_CHARS = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]')

def _contains_japanese(text: str) -> bool:
    return bool(_JP_CHARS.search(text))


# ── Vocabulary extraction ─────────────────────────────────────────────────────

# Parts of speech we care about for vocabulary tracking
_TRACKED_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

# Common particles/function words to exclude (too basic to track)
_STOP_WORDS = {
    "する", "いる", "ある", "なる", "できる", "れる", "られる",
    "の", "こと", "もの", "ため", "よう", "ところ",
    "これ", "それ", "あれ", "ここ", "そこ", "あそこ",
    "私", "僕", "あなた", "彼", "彼女",
    "いい", "ない", "ほしい",
}


def extract_words_ginza(text: str) -> list[dict]:
    """
    Extract vocabulary from Japanese text using GiNZA NLP.
    Returns list of {word, reading, pos} dicts.
    """
    doc = nlp_core.process_text_safely(text)
    if not doc:
        return extract_words_fallback(text)

    results = []
    seen = set()

    try:
        for token in doc:
            # Skip punctuation, particles, auxiliary words
            if token.pos_ not in _TRACKED_POS:
                continue

            # Use lemma (dictionary form) for consistency
            word = token.lemma_ if token.lemma_ else token.text

            # Skip very short words and stop words
            if len(word) < 2 or word in _STOP_WORDS:
                continue

            # Skip if not Japanese
            if not _contains_japanese(word):
                continue

            # Deduplicate within this extraction
            if word in seen:
                continue
            seen.add(word)

            # Get reading from morphological analysis
            reading = ""
            if hasattr(token, "morph"):
                reading_info = token.morph.get("Reading")
                if reading_info:
                    reading = reading_info[0] if isinstance(reading_info, list) else str(reading_info)

            results.append({
                "word": word,
                "reading": reading,
                "pos": token.pos_,
            })
    except Exception as e:
        print(f"[Knowledge] GiNZA extraction error: {e}")
        return extract_words_fallback(text)

    return results


def extract_words_fallback(text: str) -> list[dict]:
    """
    Fallback vocabulary extraction without GiNZA — uses regex to find
    Japanese word-like sequences. Splits on particles and punctuation
    for better granularity.
    """
    # First, split on common sentence-ending punctuation
    text = re.sub(r'[。、！？…・「」『』（）\s]+', ' ', text)

    # Split on common particles to isolate words better
    # This regex splits before/after particles like は が を に で と も の
    text = re.sub(r'([はがをにでともへのかよねえなぁ])(?=[^\u3040-\u309f]|$)', r'\1 ', text)

    # Match sequences of kanji+hiragana (like 食べる) or katakana words
    pattern = re.compile(
        r'[\u4e00-\u9fff][\u4e00-\u9fff\u3040-\u309f]*|'  # Kanji + optional hiragana
        r'[\u30a0-\u30ff]{2,}|'                              # Katakana words (2+ chars)
        r'[\u3040-\u309f]{3,}'                               # Hiragana words (3+ chars)
    )
    matches = pattern.findall(text)

    results = []
    seen = set()
    for word in matches:
        word = word.strip()
        if len(word) < 2 or word in seen or word in _STOP_WORDS:
            continue
        seen.add(word)
        results.append({
            "word": word,
            "reading": "",
            "pos": "UNKNOWN",
        })

    return results


# ── Grammar pattern extraction ────────────────────────────────────────────────

# Common grammar patterns to detect (pattern_regex, label)
_GRAMMAR_PATTERNS = [
    (re.compile(r'てください'), "て-form request"),
    (re.compile(r'ています'), "progressive/state"),
    (re.compile(r'てしまう|ちゃう'), "regrettable completion"),
    (re.compile(r'たい(です)?'), "want to (tai-form)"),
    (re.compile(r'ことができ'), "ability (koto ga dekiru)"),
    (re.compile(r'なければなら'), "must/have to"),
    (re.compile(r'てもいい'), "permission (temo ii)"),
    (re.compile(r'たことがある'), "experience (koto ga aru)"),
    (re.compile(r'そうです'), "hearsay/appearance"),
    (re.compile(r'ようにする'), "make effort to"),
    (re.compile(r'ために'), "in order to"),
    (re.compile(r'かもしれ'), "might/maybe"),
    (re.compile(r'でしょう'), "probably/right?"),
    (re.compile(r'と思います'), "I think that"),
    (re.compile(r'ましょう'), "let's (volitional)"),
    (re.compile(r'なくてもいい'), "don't have to"),
    (re.compile(r'ば[いよ]い'), "conditional (ba-form)"),
    (re.compile(r'たら'), "conditional (tara-form)"),
]


def extract_grammar_patterns(text: str) -> list[str]:
    """Detect grammar patterns present in text."""
    found = []
    for pattern, label in _GRAMMAR_PATTERNS:
        if pattern.search(text):
            found.append(label)
    return found


# ── Main tracking interface ───────────────────────────────────────────────────

def process_user_message(text: str):
    """
    Process a user's spoken message — extract vocabulary they PRODUCED
    and log it. Production is a stronger learning signal than just seeing a word.
    """
    if not _contains_japanese(text):
        return

    words = extract_words_ginza(text)
    new_words = 0
    for w in words:
        memory.upsert_vocabulary(
            word=w["word"],
            reading=w.get("reading", ""),
            pos=w.get("pos", ""),
            produced=True,
        )
        new_words += 1

    # Log activity
    if new_words > 0:
        memory.log_daily_activity(messages=1, words=new_words, xp=5)
    else:
        memory.log_daily_activity(messages=1, xp=2)


def process_tutor_message(text: str):
    """
    Process Hana's response — extract vocabulary the student was EXPOSED to.
    Only looks at the Japanese portion (before ---).
    """
    japanese_part = text.split("---")[0] if "---" in text else text

    if not _contains_japanese(japanese_part):
        return

    words = extract_words_ginza(japanese_part)
    for w in words:
        memory.upsert_vocabulary(
            word=w["word"],
            reading=w.get("reading", ""),
            pos=w.get("pos", ""),
            produced=False,  # Student saw it, didn't produce it
        )


# ── Knowledge summary for system prompt ───────────────────────────────────────

def build_knowledge_summary() -> str:
    """
    Build a formatted knowledge block for injection into Hana's system prompt.
    This tells Hana exactly what the student knows, what's weak, and what to
    introduce next.
    """
    from config import VOCAB_KNOWN_THRESHOLD

    known = memory.get_known_vocabulary(VOCAB_KNOWN_THRESHOLD)
    weak = memory.get_weak_vocabulary(limit=10)
    total = memory.get_total_vocabulary_count()
    known_count = memory.get_vocabulary_known_count(VOCAB_KNOWN_THRESHOLD)

    # Format known words (limit to 40 for prompt size)
    known_words = ", ".join(w["word"] for w in known[:40])
    if len(known) > 40:
        known_words += f" ... (+{len(known) - 40} more)"

    # Format weak words
    weak_lines = []
    for w in weak:
        reason = []
        if w.get("times_produced", 0) == 0:
            reason.append("never produced")
        if w.get("times_corrected", 0) > 0:
            reason.append(f"corrected {w['times_corrected']}x")
        weak_lines.append(f"  • {w['word']} ({', '.join(reason)})")

    weak_str = "\n".join(weak_lines) if weak_lines else "  None yet"

    # Get mistake patterns from the existing mistake system
    mistake_summary = memory.get_mistake_summary(limit=5)

    return f"""=== Student Knowledge ===
Total vocabulary encountered: {total}
Words considered known ({VOCAB_KNOWN_THRESHOLD}+ exposures): {known_count}
Known words: {known_words or 'None yet — student is a complete beginner'}

Weak/needs review:
{weak_str}

=== Recurring Mistakes ===
{mistake_summary}"""
