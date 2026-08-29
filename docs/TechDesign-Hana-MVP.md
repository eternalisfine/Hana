# Technical Design Document — はな (Hana) Japanese Conversation Tutor

**Project:** はな / Hana — Local AI Japanese Conversation Tutor  
**Version:** MVP (v1.0.0)  
**Author:** AI Tech Lead  
**Date:** 2026-08-28  

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph AudioIO["Audio Capture & Voice Detection"]
        MIC[🎤 Microphone (16kHz Mono)] --> SD[sounddevice InputStream]
        SD --> VAD[Silero VAD (Torch Hub)]
        VAD -->|Voice detected & silenced| QUEUE[Audio Chunk Queue]
    end

    subgraph PipelineThread["Background Processing Pipeline (QThread)"]
        QUEUE --> STT[faster-whisper (medium int8)]
        STT -->|Transcribed Text| LLM[Ollama qwen2.5:7b / 3b]
        LLM -->|Tutor Text Response| SAFETY[GiNZA NLP + Heuristic Safety Check]
        SAFETY -->|Parsed Japanese| TTS[VOICEVOX Engine :50021]
        TTS --> SPEAKER[🔊 Audio Output (sounddevice)]
    end

    subgraph Storage["Persistent Storage"]
        DB[(SQLite japanese_tutor.db)]
        LLM <-->|Context & History| DB
        LLM -.->|Async Profile Update| DB
    end

    subgraph UI["Desktop GUI (PySide6 / Qt6)"]
        UI_MAIN[Glassmorphism UI Window]
        PULSE[Pulsing Status Indicator]
        SIGNALS[Qt Signal Dispatcher]
        PipelineThread <--> SIGNALS
        SIGNALS --> UI_MAIN
        SIGNALS --> PULSE
    end

    MIC -.->|Barge-In Interrupt| TTS
```

---

## 2. Component Specifications

### 2.1. Voice Activity Detection (`listener.py`)
- **Library:** `silero-vad` (Torch Hub v4), `sounddevice` (InputStream callback).
- **Audio Stream:** 16,000 Hz, Mono, `float32`, chunk size of 512 samples (~32ms).
- **State Machine:** `listening` ➔ `recording` ➔ `flushed/speech_ready`.
- **Silence Timeout:** Triggers audio dispatch once silence duration exceeds `SILENCE_SECONDS` (default: 1.2s). Minimum threshold: `MIN_SPEECH_SECONDS` (default: 0.4s).

### 2.2. Speech-to-Text Engine (`stt.py`)
- **Engine:** `faster-whisper` (`WhisperModel`).
- **Device & Quantization:** `device="cpu"`, `compute_type="int8"`.
- **Transcribe Options:** `language="ja"`, `beam_size=5`, `temperature=0.0`, `vad_filter=True`.
- **Eager Loading:** Model initialized during startup inside worker thread to eliminate first-speech UI stutter.

### 2.3. Tutor Conversation Engine (`tutor.py`)
- **Model:** Ollama HTTP REST API (`/api/chat`), default model `qwen2.5:7b` (or `qwen2.5:3b`).
- **Prompt Structure:** Fixed role contract + dynamic context block dynamically loaded from `memory.build_context_block()`.
- **Response Format:**
  ```text
  [Japanese spoken response]
  ---
  🇬🇧 English: [Translation]
  💡 Note: [Grammar/Nuance Note]
  ✗ Mistake → ✓ Correction
  ```
- **Async Student Profiler:** Every `PROFILE_UPDATE_EVERY` messages (default 8), a daemon thread sends conversation history to Ollama with a strict JSON extraction schema to update learner proficiency, grammar weaknesses, and style notes.

### 2.4. Japanese Text-to-Speech Engine (`tts.py`)
- **Engine:** VOICEVOX HTTP server running locally on `http://localhost:50021`.
- **Synthesis API:** Two-step pipeline (`POST /audio_query` followed by `POST /synthesis`).
- **Text Preprocessing:** Strips furigana annotations (e.g. `食べ物(たべもの)` ➔ `食べ物`) and ignores the English explanation block after the `---` delimiter.
- **Barge-In Playback:** Audio is streamed in 2048-sample chunks with continuous checking of `threading.Event` stop flags. If user starts speaking, audio playback ceases immediately (<100ms).

### 2.5. Safety & Grammar Verification (`safety.py`)
- **NLP Parser:** `ja_ginza` / `spacy` 3.7+.
- **Checks:**
  1. Structural verb/predicate presence in multi-noun sentences.
  2. Detection of suspicious repetitive noun chains (hallucination patterns).
  3. Detection of model uncertainty markers (`かもしれません`, `ちょっとわかりません`, etc.).
  4. Particle cluster and excessive kanji run detection.
- **Output:** Returns confidence level (`high`, `medium`, `low`) and triggers UI badge alerts when flagged.

### 2.6. Persistent Storage (`memory.py`)
- **Database Engine:** SQLite 3 (`japanese_tutor.db`).
- **Schema:**
  ```sql
  CREATE TABLE IF NOT EXISTS messages (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id  TEXT    NOT NULL,
      timestamp   TEXT    NOT NULL,
      role        TEXT    NOT NULL,  -- 'user' | 'assistant'
      content     TEXT    NOT NULL,
      flagged     INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS mistakes (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp    TEXT    NOT NULL,
      original     TEXT    NOT NULL,
      correction   TEXT    NOT NULL,
      mistake_type TEXT    DEFAULT 'general',
      occurrences  INTEGER DEFAULT 1
  );

  CREATE TABLE IF NOT EXISTS style_profile (
      id               INTEGER PRIMARY KEY CHECK (id = 1),
      updated_at       TEXT,
      level_estimate   TEXT    DEFAULT 'beginner',
      grammar_notes    TEXT    DEFAULT '',
      vocabulary_notes TEXT    DEFAULT '',
      general_notes    TEXT    DEFAULT ''
  );
  ```

### 2.7. Desktop GUI (`main.py`)
- **Framework:** PySide6 (Qt6).
- **Visual Style:** 2026 Dark Glassmorphic design (`#0a0a12` deep background, rounded translucent cards, subtle glow accents, custom scrollbars, animated pulsing status dot).
- **Concurrency Model:** All audio recording, Whisper inference, Ollama calls, and VOICEVOX queries execute on dedicated background worker threads and communicate strictly via Qt signals (`Signals`) to keep the GUI thread at 60 FPS.

---

## 3. Configuration Parameters (`config.py`)

| Key | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_MODEL` | `qwen2.5:7b` / `qwen2.5:3b` | Target local LLM |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama endpoint |
| `WHISPER_MODEL` | `medium` | Whisper model size (`small`, `medium`, `large-v3`) |
| `WHISPER_LANGUAGE` | `ja` | Forced STT language |
| `VOICEVOX_URL` | `http://localhost:50021` | VOICEVOX server endpoint |
| `VOICEVOX_SPEAKER_ID` | `13` (青山龍星) / `1` (ずんだもん) | Active VOICEVOX speaker ID |
| `VAD_THRESHOLD` | `0.5` | Silero sensitivity threshold |
| `SILENCE_SECONDS` | `1.2` | Pause length to trigger end of user speech |
| `MIN_SPEECH_SECONDS` | `0.4` | Minimum speech duration to discard accidental clicks/breaths |
| `CONTEXT_MESSAGES` | `24` | Historical chat turns passed to LLM |
| `PROFILE_UPDATE_EVERY`| `8` | Frequency of background student profile updates |

---

## 4. Testing & Verification Strategy

- **Component Isolation:**
  - Mock audio buffer inputs for `stt.py` and `safety.py`.
  - Mock REST response payloads for `tutor.py` and `tts.py`.
- **Database Integrity:**
  - In-memory SQLite tests (`sqlite3.connect(":memory:")`) for CRUD operations in `memory.py`.
- **End-to-End Conversation Loop:**
  - Simulated multi-turn dialogue checking prompt synthesis, GiNZA rule checks, and profile updates.
