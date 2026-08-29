# GEMINI.md — Gemini CLI / Agent-First IDE Configuration for はな (Hana)

## Project Context
**App:** はな (Hana) — Local Japanese Conversation Tutor  
**Stack:** Python 3.10+, PySide6, faster-whisper, Silero VAD, Ollama, VOICEVOX, GiNZA, SQLite  
**Stage:** MVP In Progress / Polish & Expansion  

## Directives
1. **Master Plan:** Always read `AGENTS.md` first. It contains the project blueprint, architectural boundaries, and guidelines.
2. **Documentation:** Refer to `agent_docs/` for deep dive details on tech stack, code patterns, product requirements, and testing.
3. **Plan-First:** Propose a concise plan and await approval before modifying code across multiple files.
4. **Incremental Build & Non-Blocking GUI:** Keep UI responsive at all times; all I/O and heavy compute must remain on background worker threads.
5. **Memory Continuity:** Keep `MEMORY.md` updated with key architectural decisions and bug fixes.
6. **Communication:** Be clear, concise, and focused on practical solutions.

## Commands
- `python main.py` — Run desktop application
- `source venv/bin/activate` — Activate virtual environment
- `pip install -r requirements.txt` — Install dependencies
- `python -m unittest discover -s tests` — Run test suite
- `curl http://localhost:11434/api/tags` — Check Ollama status
- `curl http://localhost:50021/version` — Check VOICEVOX status
