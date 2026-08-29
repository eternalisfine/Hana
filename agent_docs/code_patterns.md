# Code Patterns & Architecture Guidelines

## 1. Architecture Pattern
- **Pattern:** Layered Service Architecture with Dedicated Background Worker Threads and Qt Signal Dispatching.
- **Rule 1:** UI Thread Isolation — `main.py` handles visual rendering and user interaction only. It must NEVER call blocking I/O (Whisper inference, Ollama requests, VOICEVOX synthesis, file writes).
- **Rule 2:** Event-Driven Concurrency — Use `Signals(QObject)` with PyQt/PySide signals (`status_changed`, `user_message`, `tutor_message`, etc.) to bridge background threads and the UI.
- **Rule 3:** Barge-In Safety — Always monitor `_stop_flag` during audio playback to support instant cancellation when user speaks.

---

## 2. Concurrency & Pipeline Pattern

```python
# Canonical Pipeline Worker Pattern
from PySide6.QtCore import QThread, Signal, QObject

class PipelineWorker(QThread):
    def __init__(self, audio_queue, signals):
        super().__init__()
        self.audio_queue = audio_queue
        self.signals = signals
        self._running = True

    def run(self):
        while self._running:
            try:
                audio = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            
            # Step 1: STT
            self.signals.status_changed.emit("transcribing")
            text = stt.transcribe(audio)
            if not text:
                self.signals.status_changed.emit("listening")
                continue

            self.signals.user_message.emit(text)

            # Step 2: LLM
            self.signals.status_changed.emit("thinking")
            result = tutor.chat(text, session_id)

            # Step 3: Safety & TTS
            safety_res = safety.check(result["response"])
            self.signals.tutor_message.emit(result["response"], safety_res["flagged"])
            
            self.signals.status_changed.emit("speaking")
            tts_player.speak(tutor.extract_japanese_for_tts(result["response"]))
```

---

## 3. Database Access Pattern (`memory.py`)
- SQLite queries should use context managers (`with sqlite3.connect(...) as conn:`) with `sqlite3.Row` factory.
- Write queries should be transactional and quick.
- Provide safe fallback values if rows do not exist yet.

```python
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def get_style_profile() -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM style_profile WHERE id=1").fetchone()
    return dict(row) if row else {}
```

---

## 4. Error Handling
- Never allow an unhandled exception in a background thread or callback to terminate the Qt event loop.
- Surface meaningful warnings in the UI status area.
- Log full traceback in developer terminal output.

---

## 5. Naming & Style Conventions
- **Files:** `snake_case.py` (e.g. `listener.py`, `safety.py`)
- **Classes:** `PascalCase` (e.g. `VoiceListener`, `TTSPlayer`, `GlassPanel`)
- **Functions & Methods:** `snake_case()` (e.g. `transcribe()`, `extract_japanese_for_tts()`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g. `SAMPLE_RATE`, `VOICEVOX_URL`)
