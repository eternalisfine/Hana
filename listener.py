# listener.py — Always-on mic with energy-based VAD (no button needed)

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from config import MIN_SPEECH_SECONDS, SILENCE_SECONDS, VAD_THRESHOLD

SAMPLE_RATE  = 16000
CHUNK_FRAMES = 480          # 30ms at 16kHz


class VoiceListener:
    """
    Continuously listens on the microphone.
    When speech is detected and then stops, calls `on_speech(audio_array)`.
    Automatically interrupts when the user speaks during TTS playback.

    Uses RMS energy-based voice activity detection — lightweight, no external
    dependencies beyond numpy, works on any Python version.
    """

    def __init__(self, on_speech: Callable[[np.ndarray], None],
                 on_state_change: Optional[Callable[[str], None]] = None):
        self.on_speech       = on_speech
        self.on_state_change = on_state_change or (lambda _: None)
        self._stop_event     = threading.Event()
        self._thread         = None
        self._audio_queue    = queue.Queue()
        self.muted           = False    # Soft mute (still detects, just discards)

        # Adaptive noise floor — auto-calibrated from initial ambient audio
        self._noise_floor    = 0.0
        self._calibrated     = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self):
        """Initialize VAD — call once at startup."""
        print("[Listener] Loading energy-based VAD...")
        print(f"[Listener] VAD ready ✓ (threshold={VAD_THRESHOLD})")

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def mute(self):
        self.muted = True

    def unmute(self):
        self.muted = False

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _rms(self, chunk: np.ndarray) -> float:
        """Root-mean-square energy of an audio chunk."""
        return float(np.sqrt(np.mean(chunk ** 2)))

    def _calibrate_noise_floor(self, chunk: np.ndarray):
        """Exponential moving average of ambient noise during first ~0.5s."""
        rms = self._rms(chunk)
        if self._noise_floor == 0.0:
            self._noise_floor = rms
        else:
            # Smooth: 90% old, 10% new
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * rms

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """
        Detect speech by comparing RMS energy against the noise floor.
        VAD_THRESHOLD (0.0–1.0) controls sensitivity:
          - Lower = more sensitive (triggers on quieter speech)
          - Higher = less sensitive (requires louder speech)
        """
        rms = self._rms(chunk)

        # Dynamic threshold: noise_floor + scaled gap above it
        # At threshold=0.5, speech must be ~3x the noise floor
        # At threshold=0.1, speech must be ~1.5x the noise floor
        # At threshold=0.9, speech must be ~10x the noise floor
        multiplier = 1.0 + VAD_THRESHOLD * 18.0
        threshold = max(self._noise_floor * multiplier, 0.005)

        return rms > threshold

    def _run(self):
        audio_buffer       = []
        recording          = False
        last_speech_time   = None
        calibration_chunks = 0
        calibration_target = int(0.5 * SAMPLE_RATE / CHUNK_FRAMES)  # ~0.5s

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

                # Auto-calibrate noise floor from initial silent frames
                if calibration_chunks < calibration_target:
                    self._calibrate_noise_floor(chunk)
                    calibration_chunks += 1
                    if calibration_chunks == calibration_target:
                        self._calibrated = True
                        print(f"[Listener] Noise floor calibrated: {self._noise_floor:.6f}")
                    continue

                # Continuously adapt noise floor during silence (slow drift)
                if not recording:
                    self._noise_floor = 0.995 * self._noise_floor + 0.005 * self._rms(chunk)

                is_speech = self._is_speech(chunk)

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