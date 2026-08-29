# Artifact Review Checklist 🔍

> **AGENTS:** Do not mark a feature or task as "Complete" until you verify these checks manually or via automated test runs. Provide terminal logs or UI testing results as proof.  
> **HUMANS:** Use this checklist before merging Agent-generated code.

## Code Quality & Safety
- [ ] No type regressions or unhandled edge cases in audio buffers.
- [ ] Protected files/directories (like DB schemas or core VAD/Whisper parameters) were NOT modified without permission.
- [ ] No existing, unrelated tests were deleted or skipped.
- [ ] Modules remain cleanly decoupled (`main.py` does not run blocking I/O on UI thread).

## Execution & Testing
- [ ] Application compiles and launches without syntax or runtime exceptions (`python main.py`).
- [ ] External service connectivity checks (Ollama `:11434`, VOICEVOX `:50021`) degrade gracefully if servers are offline.
- [ ] Unit & integration tests pass (`python -m unittest` or `pytest`).
- [ ] Database read/write operations execute within transactions and do not lock SQLite tables.
- [ ] Desktop UI remains fluid, responsive, and properly rendered across screen resolutions.

## Artifact Handoff
- [ ] `MEMORY.md` was updated with any new architectural decisions or resolved issues.
- [ ] Any temporary test scripts in `scratch/` or debug logs were cleaned up.
