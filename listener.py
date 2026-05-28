# listener.py — Always-on mic with Silero VAD (no button needed)

import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch
from typing import Callable
from config import VAD_THRESHOLD, SILENCE_SECONDS, MIN_SPEECH_SECONDS

SAMPLE_RATE  = 16000
CHUNK_FRAMES = 512          # ~32ms at 16kHz — Silero's required chunk size


class VoiceListener:
    """
    Continuously listens on the microphone.
    When speech is detected and then stops, calls `on_speech(audio_array)`.
    Automatically interrupts when the user speaks during TTS playback.
    """

    def __init__(self, on_speech: Callable[[np.ndarray], None],
                 on_state_change: Callable[[str], None] | None = None):
        self.on_speech       = on_speech
        self.on_state_change = on_state_change or (lambda _: None)
        self._stop_event     = threading.Event()
        self._thread         = None
        self._audio_queue    = queue.Queue()
        self._vad_model      = None
        self.muted           = False    # Soft mute (still detects, just discards)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self):
        """Load Silero VAD — call once at startup."""
        print("[Listener] Loading Silero VAD...")
        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        self._vad_model.eval()
        print("[Listener] VAD ready ✓")

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def mute(self):
        self.muted = True

    def unmute(self):
        self.muted = False

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _speech_prob(self, chunk: np.ndarray) -> float:
        tensor = torch.from_numpy(chunk)
        with torch.no_grad():
            prob = self._vad_model(tensor, SAMPLE_RATE).item()
        return prob

    def _run(self):
        audio_buffer       = []
        recording          = False
        last_speech_time   = None

        def _mic_callback(indata, frames, time_info, status):
            self._audio_queue.put(indata[:, 0].copy())  # mono

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_FRAMES,
            callback=_mic_callback,
        ):
            self.on_state_change("listening")

            while not self._stop_event.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check for end of speech during timeout
                    if recording and last_speech_time:
                        elapsed = time.monotonic() - last_speech_time
                        if elapsed >= SILENCE_SECONDS:
                            self._flush(audio_buffer)
                            audio_buffer     = []
                            recording        = False
                            last_speech_time = None
                    continue

                prob = self._speech_prob(chunk)
                is_speech = prob > VAD_THRESHOLD

                if is_speech:
                    if not recording:
                        recording = True
                        self.on_state_change("recording")

                    last_speech_time = time.monotonic()
                    audio_buffer.append(chunk)

                elif recording:
                    audio_buffer.append(chunk)   # keep buffering brief silence
                    elapsed = time.monotonic() - last_speech_time
                    if elapsed >= SILENCE_SECONDS:
                        self._flush(audio_buffer)
                        audio_buffer     = []
                        recording        = False
                        last_speech_time = None
                        self.on_state_change("listening")

    def _flush(self, buffer: list):
        if not buffer:
            return
        audio = np.concatenate(buffer)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_SECONDS:
            return
        if not self.muted:
            self.on_speech(audio)