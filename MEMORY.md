# System Memory & Context 🧠

## 🏗️ Active Phase & Goal
**Current Task:** Project setup and stabilization under the Vibe-Coding workflow framework.  
**Active Phase:** Phase 1 — MVP Core Validation & Robustness Enhancement.  
**Next Steps:**
1. Verify end-to-end speech pipeline (Mic ➔ Silero VAD ➔ faster-whisper ➔ Ollama ➔ VOICEVOX ➔ GiNZA).
2. Validate UI responsiveness and barge-in interruption timing.
3. Prepare unit & integration test suites for `memory.py`, `safety.py`, and `tutor.py`.

---

## 📂 Architectural Decisions
- **2026-08-28** — Adopted PySide6 (Qt6) with custom dark glassmorphism for native cross-platform performance.
- **2026-08-28** — Implemented two-pass speech detection: Silero VAD for real-time boundary detection + faster-whisper internal VAD for transcription filtering.
- **2026-08-28** — Isolated all heavy workloads in background QThreads and daemon threads to guarantee 60 FPS GUI fluidity.
- **2026-08-28** — Added offline GiNZA NLP safety verification to catch structural grammar hallucinations before audio playback.

---

## 🐛 Known Issues & Quirks
- Ollama first request may take a few seconds if model needs to be loaded into VRAM/RAM.
- VOICEVOX engine must be launched separately as a background service on port `50021`.
- Arch Linux requires virtual environment (`venv`) for pip installations due to PEP 668.

---

## 📜 Completed Phases
- [x] Initial scaffold & module architecture
- [x] SQLite database schema creation (`messages`, `mistakes`, `style_profile`)
- [x] Silero VAD real-time audio listener & endpointing
- [x] faster-whisper CPU integration with int8 quantization
- [x] Ollama chat engine with adaptive prompt context injection
- [x] VOICEVOX HTTP API integration & streaming audio playback with barge-in interruption
- [x] GiNZA Japanese NLP safety & grammar verification layer
- [x] PySide6 glassmorphic desktop GUI with animated status indicators
