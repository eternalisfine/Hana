# tutor.py — Ollama conversation engine with adaptive Japanese tutoring

import json
import requests
import threading
from config import OLLAMA_URL, OLLAMA_MODEL, CONTEXT_MESSAGES, PROFILE_UPDATE_EVERY
import memory

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_BASE = """You are Hana (はな), a warm, patient, and deeply knowledgeable Japanese language tutor. You are having a real spoken conversation with a student practicing Japanese.

## CRITICAL RULES — Never violate these:
1. Only use Japanese grammar, vocabulary, and expressions you are 100% certain are correct and natural
2. Prefer simpler, unambiguous phrasing over complex forms you are less sure about
3. Use standard Tokyo/NHK-style Japanese (no regional dialect unless student asks)
4. Never fabricate words, grammar rules, or claim something is natural if you are uncertain
5. Never ignore a student mistake — always correct it warmly, clearly, and immediately
6. If you are genuinely unsure whether a phrase is natural, say so explicitly

## Response Format (always follow this structure):
Japanese response here — spoken, natural, not too long
---
🇬🇧 English: translation of what you just said
💡 Note: (optional) brief grammar/vocabulary explanation if relevant
✗ Mistake → ✓ Correction (if the student made an error, format exactly like this)

## Conversation Style:
- Scale Japanese complexity to the student's level (from the profile below)
- Beginners: add furigana in brackets after kanji, e.g. 食べ物(たべもの)
- Respond at spoken length — not essays
- Be genuinely warm and human, like a real language partner
- Naturally reference things the student has said or struggled with before
- Ask follow-up questions to keep the conversation flowing

## {context}
"""

def _build_system_prompt() -> str:
    context = memory.build_context_block()
    return _SYSTEM_BASE.format(context=context)


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat(user_message: str, session_id: str) -> dict:
    """
    Send user message → get tutor response.
    Returns: { response, error, flagged_by_tutor }
    """
    memory.add_message(session_id, "user", user_message)

    history = memory.get_recent_messages(CONTEXT_MESSAGES)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    payload = {
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "system":  _build_system_prompt(),
        "stream":  False,
        "options": {
            "temperature": 0.65,
            "num_ctx":     4096,
            "repeat_penalty": 1.1,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        response_text = resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return _error("Cannot connect to Ollama. Is it running? (ollama serve)")
    except requests.exceptions.Timeout:
        return _error("Ollama timed out. The model may be loading — try again.")
    except Exception as e:
        return _error(f"Ollama error: {e}")

    memory.add_message(session_id, "assistant", response_text)

    # Async profile update every N user messages
    count = memory.get_user_message_count()
    if count > 0 and count % PROFILE_UPDATE_EVERY == 0:
        threading.Thread(
            target=_update_profile, daemon=True
        ).start()

    return {
        "response": response_text,
        "error":    False,
        "flagged_by_tutor": False,
    }


def _error(msg: str) -> dict:
    return {"response": msg, "error": True, "flagged_by_tutor": False}


# ── Profile Updater (background) ──────────────────────────────────────────────

def _update_profile():
    """Ask the model to analyze recent conversation and update the student profile."""
    recent = memory.get_recent_messages(20)
    if len(recent) < 4:
        return

    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    prompt = (
        "Analyze this Japanese tutoring conversation and output ONLY valid JSON "
        "(no markdown, no explanation):\n\n"
        f"{convo}\n\n"
        "JSON format:\n"
        '{"level_estimate":"beginner|elementary|intermediate|upper-intermediate|advanced",'
        '"grammar_notes":"brief note on grammar strengths/weaknesses",'
        '"vocabulary_notes":"brief note on vocabulary level and gaps",'
        '"general_notes":"speaking style, topics of interest, pace"}'
    )

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":   OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":  False,
            "options": {"temperature": 0.1, "num_ctx": 2048}
        }, timeout=60)
        text = resp.json()["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        profile = json.loads(text)
        memory.update_style_profile(**profile)
    except Exception:
        pass  # Best-effort — never crash the app


# ── Utilities ─────────────────────────────────────────────────────────────────

def extract_japanese_for_tts(response: str) -> str:
    """
    Pull out only the Japanese portion (before the --- separator)
    so TTS doesn't read out English explanations.
    """
    if "---" in response:
        japanese_part = response.split("---")[0].strip()
    else:
        japanese_part = response.strip()

    # Remove furigana brackets for cleaner TTS: 食べ物(たべもの) → 食べ物
    import re
    japanese_part = re.sub(r'\([ぁ-ん]+\)', '', japanese_part)
    japanese_part = re.sub(r'[✗✓•→].*', '', japanese_part, flags=re.MULTILINE)

    return japanese_part.strip()


def check_ollama() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False