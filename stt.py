# stt.py — faster-whisper speech-to-text (offline, CPU)

import io
import numpy as np
import soundfile as sf
from config import WHISPER_MODEL, WHISPER_LANGUAGE

_model = None


def load_model():
    """Eagerly load Whisper (call at startup to avoid first-use delay)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"[STT] Loading Whisper '{WHISPER_MODEL}'...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"          # Fastest on CPU, minimal quality loss
        )
        print("[STT] Whisper ready ✓")
    return _model


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribe a float32 mono numpy array.
    Returns the transcribed text, or empty string if nothing detected.
    """
    model = load_model()

    # Write to in-memory WAV so Whisper can read it
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)

    segments, info = model.transcribe(
        buf,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        best_of=5,
        temperature=0.0,               # Deterministic
        vad_filter=True,               # Extra VAD pass inside Whisper
        vad_parameters=dict(
            min_silence_duration_ms=400,
            speech_pad_ms=200,
        ),
        word_timestamps=False,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text