# Project Brief — はな (Hana)

- **Product Vision:** A completely offline, private, zero-API-cost Japanese speaking tutor that listens in real time, converses naturally, adapts to the student's level, and remembers their progress across sessions.
- **Target Audience:** Independent Japanese learners (JLPT N5–N2) who want speaking immersion without subscription fees or social performance anxiety.

---

## Core Conventions
- **Zero-Cloud Guarantee:** All speech recognition, voice synthesis, inference, and memory run on the local machine.
- **Hands-Free Priority:** No buttons required during a practice session. Audio capture starts and stops automatically via voice detection.
- **Instant Interruption (Barge-In):** The student can interrupt the tutor mid-speech at any moment by speaking.
- **Pedagogical Structure:** Every response includes spoken Japanese, English translation, contextual notes, and explicit corrections of any mistakes.

---

## Key Development Commands
- Start Application: `python main.py` (or `./run.sh`)
- Run Tests: `python -m unittest discover -s tests`
- Check Ollama: `curl http://localhost:11434/api/tags`
- Check VOICEVOX: `curl http://localhost:50021/version`
