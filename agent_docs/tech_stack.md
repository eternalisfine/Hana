# Tech Stack & Tools

## Core Stack
- **Language & Runtime:** Python 3.10+ (tested on 3.10, 3.11, 3.12)
- **GUI Framework:** PySide6 (Qt 6.6+)
- **Speech-to-Text (STT):** `faster-whisper` (CTranslate2, `medium` model, CPU `int8` quantization)
- **Voice Activity Detection (VAD):** `silero-vad` (PyTorch Hub v4)
- **Audio Capture & Streaming:** `sounddevice` + `soundfile` + `numpy`
- **Conversational AI Engine (LLM):** Ollama API (`qwen2.5:7b` / `qwen2.5:3b` on `http://localhost:11434`)
- **Text-to-Speech (TTS):** VOICEVOX Engine (`http://localhost:50021`)
- **Japanese NLP & Grammar Safety:** `ginza` + `ja-ginza` + `spacy`
- **Persistent Storage:** SQLite 3 (`japanese_tutor.db`)

---

## Service Integrations & Endpoints

### 1. Ollama (LLM)
- Endpoint: `POST http://localhost:11434/api/chat`
- Default model: `qwen2.5:7b` (fallback: `qwen2.5:3b`)
- Health check: `GET http://localhost:11434/api/tags`

### 2. VOICEVOX (TTS)
- Endpoint: `POST http://localhost:50021/audio_query` & `POST http://localhost:50021/synthesis`
- Speaker list: `GET http://localhost:50021/speakers`
- Health check: `GET http://localhost:50021/version`

---

## Audio Pipeline Parameters
- **Sampling Rate:** 16,000 Hz
- **Channels:** 1 (Mono)
- **Data Type:** `float32` (normalized between -1.0 and 1.0)
- **VAD Frame Size:** 512 samples (~32 ms)
- **Playback Chunk Size:** 2,048 samples

---

## Error Handling Pattern
```python
import requests

def check_service_connection(url: str, timeout: float = 3.0) -> bool:
    """Safe ping to local service endpoints without raising exceptions."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False
    except Exception as e:
        print(f"[Network] Unexpected error checking {url}: {e}")
        return False
```
