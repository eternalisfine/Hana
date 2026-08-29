# Product Requirements Summary

## Must-Have Features (MVP)
1. **Always-On Voice Listening:** Continuous microphone listening using Silero VAD at 16kHz mono.
2. **Local Speech-to-Text:** Offline Japanese transcription using `faster-whisper` (medium, int8).
3. **Conversational Engine:** Adaptive persona `Hana` powered by local Ollama (`qwen2.5:7b` / `3b`) with structured multi-part response (Japanese, English translation, grammar notes, mistake corrections).
4. **Natural Speech Synthesis:** High quality Japanese TTS via local VOICEVOX engine with multiple character voices and playback speed controls.
5. **Instant Barge-In Interruption:** Immediate cutoff of TTS playback when student starts speaking.
6. **Japanese NLP Safety Layer:** GiNZA/spaCy dependency parser + heuristics flagging uncertain or ungrammatical output with UI warning indicators (⚠).
7. **Persistent Memory & Profiling:** SQLite database (`japanese_tutor.db`) tracking conversation history, recurring mistakes count, and automatic student level updates.
8. **Modern PySide6 GUI:** 2026 Dark Glassmorphic desktop interface with animated pulsing indicators and status displays.

---

## User Stories & Flow
- **Dialogue Loop:** User speaks ➔ Silero VAD detects voice & silence endpoint ➔ faster-whisper transcribes audio ➔ Ollama generates structured response ➔ GiNZA runs safety check ➔ VOICEVOX plays speech ➔ UI displays text with furigana/notes.
- **Interruption Flow:** User speaks during playback ➔ Silero VAD signals speech ➔ TTS player stops stream immediately ➔ Pipeline transitions to recording.
