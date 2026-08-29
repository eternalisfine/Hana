# AGENTS.md — Master Plan for はな (Hana) Japanese Conversation Tutor

## Project Overview & Stack
**App:** はな (Hana) — Local Japanese Conversation Tutor  
**Overview:** A 100% offline, zero-API-cost desktop Japanese speaking practice application. Uses Silero VAD for hands-free audio capture, faster-whisper for speech-to-text, local Ollama (Qwen 2.5) for conversational intelligence, VOICEVOX for neural Japanese voice synthesis, GiNZA NLP for grammar safety verification, and SQLite for adaptive student memory.  
**Stack:** Python 3.10+, PySide6 (Qt6), faster-whisper, Silero VAD (PyTorch), Ollama, VOICEVOX, GiNZA / spaCy, SQLite  
**Critical Constraints:**
- 100% local execution — no external cloud API dependencies.
- Non-blocking GUI: all heavy compute (VAD, STT, LLM, TTS, NLP) must run on worker threads communicating via Qt Signals.
- Spoken Japanese must follow standard Tokyo/NHK conventions with level scaling and furigana where appropriate.
- Barge-in voice interruption must immediately stop TTS audio output in <100ms when the user speaks.

---

## Setup & Commands
Execute these commands for standard development workflows.
- **Virtual Environment Activation:**
  - Linux / macOS: `source venv/bin/activate`
  - Windows: `venv\Scripts\activate`
- **Dependency Installation:** `pip install -r requirements.txt`
- **Launch Application:** `python main.py` or `./run.sh`
- **Verify Ollama Server:** `curl http://localhost:11434/api/tags`
- **Verify VOICEVOX Server:** `curl http://localhost:50021/version`
- **Testing:** `python -m unittest discover -s tests` (or `pytest`)
- **Linting & Code Style:** `flake8 .` or `ruff check .`

---

## Protected Areas
Do NOT modify these areas without explicit human approval:
- **Core Database Schema:** Modifications to existing columns/tables in `memory.py` without backward-compatible migrations.
- **Audio Capture Parameters:** Hardcoded sample rates (16000Hz mono) in `listener.py` and `stt.py` required by Silero VAD and Whisper.
- **Safety Base Rules:** The critical 6 safety rules in the system prompt inside `tutor.py`.

---

## Coding Conventions
- **Formatting & Style:** PEP 8 compliance, clean docstrings, descriptive function and variable names (`snake_case` for functions/variables, `PascalCase` for classes).
- **Architecture Rules:** Modular separation of concerns (`listener.py` for audio capture, `stt.py` for transcription, `tutor.py` for LLM inference, `tts.py` for voice synthesis, `safety.py` for NLP verification, `memory.py` for DB, `main.py` for UI).
- **Type Hints:** Use standard Python type annotations (`Callable`, `Optional`, `tuple`, `dict`, `list`).
- **Error Handling:** Never crash the UI thread. Use defensive `try/except` with informative console logging and user-friendly error banners.

---

## Agent Behaviors
These rules apply across all AI coding assistants:
1. **Plan Before Execution:** ALWAYS propose a brief step-by-step plan and obtain user confirmation before making non-trivial modifications.
2. **Refactor Over Rewrite:** Keep working code intact; perform incremental refactors rather than total rewrites.
3. **Context Compaction:** Write architectural decisions and bug resolutions into `MEMORY.md` instead of cluttering chat history.
4. **Iterative Verification:** Test new logic and verify GUI stability after each change (refer to `REVIEW-CHECKLIST.md`).
5. **No Silent Failures:** Always log or surface exceptions with actionable recovery instructions.
