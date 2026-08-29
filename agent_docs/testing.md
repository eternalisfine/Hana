# Testing Strategy

## Frameworks & Tools
- **Unit & Integration Testing:** Python `unittest` / `pytest`
- **Mocking:** `unittest.mock` (for mocking audio devices, Ollama API, VOICEVOX API)

---

## Testing Matrix

| Module | Test Focus | Verification Method |
| :--- | :--- | :--- |
| `memory.py` | Message logging, mistake counting, style profile updates | In-memory SQLite (`:memory:`) unit tests |
| `safety.py` | GiNZA dependency parsing, uncertainty heuristics, noun-chain detection | Text test vectors with known valid/invalid sentences |
| `tutor.py` | Japanese extraction regex, furigana stripping, system prompt synthesis | Unit tests with mock responses |
| `tts.py` | Interruption stop flag, audio query & synthesis error handling | Mocked HTTP requests and audio streams |
| `stt.py` | Audio buffer WAV serialization and transcription handling | Synthetic audio array tests |

---

## Execution Commands
- **Run all unit tests:**
  ```bash
  python -m unittest discover -s tests
  ```
- **Run specific test file:**
  ```bash
  python -m unittest tests/test_memory.py
  ```

---

## Quality Gates
- All tests must pass before marking tasks as complete.
- Verify that GUI remains responsive and does not freeze during background model operations.
