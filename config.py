# config.py — Edit these to match your setup

OLLAMA_MODEL        = "qwen2.5:3b"
OLLAMA_URL          = "http://localhost:11434/api/chat"

WHISPER_MODEL       = "medium"
WHISPER_LANGUAGE    = "ja"                  # Force Japanese transcription

VOICEVOX_URL        = "http://localhost:50021"
VOICEVOX_SPEAKER_ID = 1
                                            # Run: GET /speakers for full list

VAD_THRESHOLD       = 0.5                   # 0.0–1.0, higher = less sensitive
SILENCE_SECONDS     = 1.2                   # Seconds of silence before sending audio
MIN_SPEECH_SECONDS  = 0.4                   # Minimum speech length to process

CONTEXT_MESSAGES    = 24                    # How many past messages to feed the tutor
PROFILE_UPDATE_EVERY = 8                    # Update style profile every N user messages