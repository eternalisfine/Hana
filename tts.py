# tts.py — VOICEVOX text-to-speech (local, offline, natural Japanese)

import io
import threading
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from config import VOICEVOX_URL, VOICEVOX_SPEAKER_ID


class TTSPlayer:
    """
    Plays Japanese speech via VOICEVOX.
    Supports instant interruption (e.g. when user starts speaking).
    """

    def __init__(self):
        self._stop_flag  = threading.Event()
        self._lock       = threading.Lock()
        self._thread     = None
        self.speaker_id  = VOICEVOX_SPEAKER_ID
        self.speed       = 0.9    # Playback speed (0.5–2.0)
        self.on_start    = None   # Callback when playback starts
        self.on_end      = None   # Callback when playback ends

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Speak text asynchronously. Interrupts any current playback."""
        if not text.strip():
            return
        self.interrupt()
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._worker, args=(text,), daemon=True
        )
        self._thread.start()

    def interrupt(self):
        """Stop playback immediately."""
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def is_speaking(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _worker(self, text: str):
        if self.on_start:
            self.on_start()
        try:
            audio_data, sample_rate = self._synthesize(text)
            if audio_data is not None and not self._stop_flag.is_set():
                self._play(audio_data, sample_rate)
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            if self.on_end:
                self.on_end()

    def _synthesize(self, text: str):
        """Call VOICEVOX API → return (audio_array, sample_rate)."""
        # Step 1: audio_query
        qr = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": self.speaker_id},
            timeout=30,
        )
        if qr.status_code != 200:
            print(f"[TTS] audio_query failed {qr.status_code}: {qr.text[:200]}")
            return None, None

        query = qr.json()

        # Speed controlled by slider; intonation fixed for clarity
        query["speedScale"]      = self.speed
        query["intonationScale"] = 1.1

        # Step 2: synthesis
        sr = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": self.speaker_id},
            json=query,
            timeout=60,
        )
        if sr.status_code != 200:
            print(f"[TTS] synthesis failed {sr.status_code}")
            return None, None

        audio_data, sample_rate = sf.read(io.BytesIO(sr.content), dtype="float32")
        return audio_data, sample_rate

    def _play(self, audio: np.ndarray, sample_rate: int):
        """Stream audio in chunks, checking stop_flag between each."""
        CHUNK = 2048
        channels = 1 if audio.ndim == 1 else audio.shape[1]

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        ) as stream:
            for start in range(0, len(audio), CHUNK):
                if self._stop_flag.is_set():
                    break
                chunk = audio[start : start + CHUNK]
                if chunk.ndim == 1:
                    chunk = chunk.reshape(-1, 1)
                stream.write(chunk)

    # ── Status ────────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{VOICEVOX_URL}/version", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_speakers(self) -> list:
        try:
            r = requests.get(f"{VOICEVOX_URL}/speakers", timeout=10)
            return r.json()
        except Exception:
            return []