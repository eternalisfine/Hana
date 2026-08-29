# Product Requirements Document (PRD) — はな (Hana)

**Product Name:** はな / Hana — Japanese Conversation Tutor  
**Version:** MVP (v1.0.0)  
**Status:** In Progress / Active Development  
**Author:** AI Tech Lead & Development Team  

---

## 1. Product Vision & Overview

**Hana** is a fully local, open-source Japanese conversation practice application designed for desktop. It acts as an empathetic, patient, and knowledgeable language tutor that listens continuously, provides immediate spoken replies, automatically adapts to the learner's proficiency level, and remembers past interactions and common mistakes without requiring any internet connection or third-party cloud APIs.

---

## 2. Target Audience & Personas

- **Primary Persona: The Self-Studying Japanese Learner (JLPT N5 – N2)**
  - Knows hiragana/katakana and basic grammar, but lacks opportunities to speak out loud.
  - Wants a zero-pressure environment to practice real conversations without fear of judgment.
  - Prioritizes privacy, offline capability, and zero recurring subscription fees.
- **Secondary Persona: Intermediate Japanese Practice Enthusiast**
  - Needs continuous conversational immersion, immediate correction of unnatural phrasing, and nuance explanations.

---

## 3. Primary User Stories

1. **Hands-Free Conversational Practice:**
   - *As a student*, I want to talk into my microphone and receive immediate spoken Japanese replies without having to click "record" or "stop" buttons, so that our dialogue feels like an authentic conversation.
2. **Instant Barge-In / Interruption:**
   - *As a student*, I want to be able to interrupt Hana while she is speaking simply by talking, so that I can correct myself or change topics naturally.
3. **Structured Pedagogical Feedback:**
   - *As a student*, I want Hana's responses to include a Japanese utterance, an English translation, cultural/grammar notes, and explicit corrections of any mistakes I made, so that I learn from every exchange.
4. **Adaptive Long-Term Memory:**
   - *As a student*, I want the tutor to remember my estimated proficiency level, past topics, and recurring mistakes across sessions, so that conversations stay challenging and personalized.
5. **Accuracy & Hallucination Guardrails:**
   - *As a student*, I want to be warned if the tutor's Japanese grammar or phrasing might be structurally suspect or uncertain, so that I don't internalize bad habits.

---

## 4. MVP Feature Breakdown

### 4.1. Must-Have Features (In MVP)
- [x] **Continuous Voice Activity Detection (VAD):** Always-on microphone listening powered by Silero VAD, detecting speech start and automatic silence endpointing (configurable threshold and silence duration).
- [x] **Offline Speech-to-Text (STT):** Fast CPU transcription via `faster-whisper` (medium model, int8 quantized) with automatic Japanese language forcing.
- [x] **Conversational Engine (Ollama):** Prompt-engineered Japanese persona (`Hana`) running locally via Ollama (`qwen2.5:7b` or `qwen2.5:3b`) with structured multi-part output (Japanese, English translation, grammar note, mistake correction).
- [x] **Neural Speech Synthesis (TTS):** High quality Japanese vocal synthesis via local VOICEVOX engine with selectable speaker character personas and speed modulation.
- [x] **Barge-In Interruption:** Instant cancellation of active VOICEVOX audio playback upon detecting user speech.
- [x] **Safety & Grammar Checking Layer:** GiNZA NLP dependency parser + heuristic uncertainty detector flagging suspicious sentences with UI alert badges (⚠).
- [x] **Persistent SQLite Memory:** Tracking chat history, mistake frequency log (with occurrences count), and background asynchronous student profile refinement.
- [x] **Modern Desktop UI:** PySide6 (Qt6) dark glassmorphic interface with pulsing animated status indicators (listening, recording, transcribing, thinking, speaking), settings dialog, and history views.

### 4.2. Nice-to-Have Features (Post-MVP Roadmap)
- [ ] Direct audio speed & pitch tuning per speaker persona.
- [ ] Exportable mistake flashcards (Anki `.apkg` or CSV export).
- [ ] JLPT Kanji/Grammar level tagger for generated sentences.
- [ ] Audio pitch accent visualizer / pitch contour graphs.
- [ ] Multi-voice roleplay scenarios (e.g., ordering at a restaurant, job interview, doctor visit).

### 4.3. Explicitly Out of Scope (NOT in MVP)
- Cloud server synchronization / web authentication.
- Paid cloud API integrations (OpenAI / Azure / ElevenLabs).
- Mobile apps (iOS / Android).

---

## 5. Success Metrics & Quality Gates

- **Zero API Bills:** 100% functionality operational with network disconnected.
- **Turnaround Latency:** End of user speech to start of voice synthesis under 2.5s on recommended hardware.
- **Interruption Responsiveness:** Barge-in halts audio playback in < 200ms.
- **Stability:** Zero UI thread locking during audio processing or model inference.
- **Accuracy:** Zero unflagged structural hallucinations passing through the GiNZA check.
