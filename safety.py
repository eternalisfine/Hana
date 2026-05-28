# safety.py — Japanese accuracy checking via GiNZA (no extra LLM calls)

import re

# GiNZA is optional — app works without it but checking is skipped
try:
    import spacy
    _nlp = spacy.load("ja_ginza")
    GINZA_OK = True
except Exception:
    _nlp = None
    GINZA_OK = False


# ── Japanese extraction ────────────────────────────────────────────────────────

_JP_PATTERN = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff00-\uffef]+')

def extract_japanese_segments(text: str) -> list[str]:
    """Return list of Japanese-character runs from mixed text."""
    # Only look at the part before --- (the actual Japanese response)
    japanese_section = text.split("---")[0] if "---" in text else text
    return [m.group() for m in _JP_PATTERN.finditer(japanese_section)
            if len(m.group()) > 2]


# ── GiNZA structural check ────────────────────────────────────────────────────

def _ginza_check(text: str) -> list[str]:
    if not GINZA_OK or not _nlp:
        return []

    warnings = []
    try:
        doc = _nlp(text)
        for sent in doc.sents:
            tokens = list(sent)
            if len(tokens) < 2:
                continue

            has_verb = any(t.pos_ in ("VERB", "AUX") for t in tokens)
            has_noun = any(t.pos_ in ("NOUN", "PROPN", "PRON") for t in tokens)

            # Flag sentences with no verb and multiple nouns (likely malformed)
            if len(tokens) > 4 and not has_verb:
                warnings.append(f"Possible missing verb in: 「{sent.text.strip()}」")

            # Flag very long noun chains (often model hallucination)
            noun_run = 0
            for t in tokens:
                if t.pos_ in ("NOUN", "PROPN"):
                    noun_run += 1
                    if noun_run >= 5:
                        warnings.append(f"Suspicious noun chain in: 「{sent.text.strip()}」")
                        break
                else:
                    noun_run = 0

    except Exception:
        pass

    return warnings


# ── Confidence heuristics ─────────────────────────────────────────────────────

# Phrases the model sometimes produces when unsure
_UNCERTAINTY_MARKERS = [
    "かもしれません", "かもしれない", "と思います", "でしょうか",
    "ちょっとわかりません", "確かではありません"
]

def _has_uncertainty_markers(text: str) -> bool:
    return any(m in text for m in _UNCERTAINTY_MARKERS)


# Patterns that often indicate model confusion
_SUSPICIOUS = [
    re.compile(r'[^\u3000-\u9fff][はがをにでもと]{3,}'),  # particle cluster
    re.compile(r'[\u4e00-\u9fff]{8,}'),                    # very long kanji run
]

def _has_suspicious_patterns(text: str) -> bool:
    return any(p.search(text) for p in _SUSPICIOUS)


# ── Public interface ──────────────────────────────────────────────────────────

def check(response_text: str) -> dict:
    """
    Run all safety checks on tutor response.
    Returns: { flagged: bool, warnings: list[str], confidence: str }
    """
    segments = extract_japanese_segments(response_text)
    if not segments:
        return {"flagged": False, "warnings": [], "confidence": "high"}

    combined = "。".join(segments)
    warnings = []

    # Check 1: GiNZA structural analysis
    warnings.extend(_ginza_check(combined))

    # Check 2: Uncertainty markers in response
    if _has_uncertainty_markers(combined):
        warnings.append("Tutor expressed uncertainty in this response")

    # Check 3: Suspicious patterns
    if _has_suspicious_patterns(combined):
        warnings.append("Unusual character sequence detected — verify this phrasing")

    confidence = "high"
    if len(warnings) >= 2:
        confidence = "low"
    elif len(warnings) == 1:
        confidence = "medium"

    return {
        "flagged":    len(warnings) > 0,
        "warnings":   warnings,
        "confidence": confidence,
    }


def ginza_status() -> str:
    if GINZA_OK:
        return "GiNZA ✓"
    return "GiNZA ✗ (install: pip install ginza ja-ginza)"