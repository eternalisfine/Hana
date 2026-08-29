# Deep Research Findings — はな (Hana) Japanese Conversation Tutor

**Date:** 2026-08-28  
**Project:** はな / Hana — Local AI Japanese Conversation Tutor  
**Platform:** Desktop (Linux, macOS, Windows)

---

## 1. Executive Summary & Problem Statement

Language learners struggle most with **spoken fluency** due to:
1. High anxiety / fear of making mistakes in front of human native speakers.
2. High recurring costs and scheduling constraints of 1-on-1 human tutors (e.g., iTalki, Preply).
3. Cloud AI voice tools suffering from high latency, expensive per-minute API fees (OpenAI Realtime API, ElevenLabs), and cloud privacy concerns.
4. Existing language apps (Duolingo, Rosetta Stone) relying on rigid multiple-choice or button-press repeating rather than dynamic, conversational practice.

**Hana** solves this by delivering a **100% local, zero-cost, private, always-listening conversational partner** with natural voice synthesis, intelligent grammar checking, and persistent memory of student strengths, weaknesses, and recurring mistakes.

---

## 2. Competitive Landscape

| Feature | Hana (はな) | Cloud Voice Tutors (ChatGPT Voice, Speak) | Traditional Apps (Duolingo) | Human Tutors (iTalki) |
| :--- | :--- | :--- | :--- | :--- |
| **Hosting & Privacy** | 100% Local / Offline | Cloud / Third-party servers | Cloud | Live human video |
| **Cost** | $0 / Free / Open Source | $15–$30/mo subscription | Freemium / $10–$20/mo | $15–$40/hour |
| **Latency & Interruption** | Instant Barge-In via Silero VAD | Variable network latency; interruption supported | N/A (Turn-based buttons) | Natural human conversation |
| **Voice Naturalness (Japanese)** | High (VOICEVOX Neural TTS) | High (OpenAI/ElevenLabs) | Low to Medium | Native speaker |
| **Grammar Verification** | Dual layer (LLM + GiNZA NLP) | LLM only (can hallucinate) | Scripted answers | Human expertise |
| **Long-term Student Memory** | Local SQLite profile & mistake tracker | Session-based or account history | Fixed curriculum progress | Human tutor memory |

---

## 3. Technical Feasibility & Component Breakdown

1. **Voice Activity Detection (VAD):**
   - **Silero VAD v4**: Runs on CPU (<1ms per 30ms chunk), accurately isolates Japanese speech from background ambient noise, detects speech boundaries deterministically.
2. **Speech-to-Text (STT):**
   - **faster-whisper (medium/int8)**: CTranslate2-accelerated Whisper model. Transcribes Japanese audio on modern CPUs in <1.2 seconds with high character accuracy and furigana/kanji handling.
3. **Conversational Intelligence (LLM):**
   - **Ollama (`qwen2.5:7b` or `qwen2.5:3b`)**: Qwen 2.5 series exhibits state-of-the-art Japanese syntactic and pragmatic capability among open-weight models under 8B parameters.
4. **Japanese Text-to-Speech (TTS):**
   - **VOICEVOX (HTTP Engine on `:50021`)**: Specifically trained on Japanese phonetics, mora timing, and pitch accents with customizable voice personas (ずんだもん, 四国めたん, 青山龍星, etc.).
5. **Accuracy & Safety Verification:**
   - **GiNZA (spaCy + SudachiPy)**: Deep dependency parsing catches missing predicates, particle mismatches, and unnatural noun chains without requiring secondary LLM roundtrips.

---

## 4. Hardware Requirements & Performance Targets

- **CPU:** 4+ cores (Intel i5/i7/i9 8th gen+, AMD Ryzen 3000+, Apple Silicon M1/M2/M3/M4).
- **RAM:** Minimum 8GB (16GB recommended for `qwen2.5:7b` + Whisper `medium`).
- **Disk:** ~10GB storage (Ollama models + Whisper weights + VOICEVOX engine).
- **Target Turnaround Latency:** < 2.5 seconds from user stopping speech to VOICEVOX voice playback start on modern hardware.
