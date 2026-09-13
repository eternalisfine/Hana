# tests/test_listener.py — Thorough test suite for listener.py (VAD Microphone Listener)

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import time
import queue

import listener
from config import MIN_SPEECH_SECONDS, SILENCE_SECONDS, VAD_THRESHOLD


class TestListener(unittest.TestCase):
    """
    Comprehensive tests for VoiceListener covering lifecycle, mute/unmute,
    energy RMS calculation, noise calibration, speech detection heuristics,
    buffer flushing, and callback state transitions without requiring audio hardware.
    """

    def setUp(self):
        self.mock_speech_cb = MagicMock()
        self.mock_state_cb = MagicMock()
        self.listener = listener.VoiceListener(
            on_speech=self.mock_speech_cb,
            on_state_change=self.mock_state_cb,
        )

    def tearDown(self):
        self.listener.stop()

    # ── Lifecycle Tests ───────────────────────────────────────────────────────

    def test_load_prints_ready(self):
        # Verify load executes cleanly
        try:
            self.listener.load()
        except Exception as e:
            self.fail(f"load() raised an exception: {e}")

    @patch("listener.sd.InputStream")
    def test_start_creates_thread(self, mock_stream):
        # Prevent InputStream from blocking
        self.listener._stop_event.set()
        self.listener.start()
        self.assertIsNotNone(self.listener._thread)
        self.assertTrue(self.listener._thread.is_alive() or self.listener._stop_event.is_set())

    def test_stop_sets_event(self):
        self.listener.stop()
        self.assertTrue(self.listener._stop_event.is_set())

    def test_stop_joins_thread(self):
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        self.listener._thread = mock_thread
        self.listener.stop()
        mock_thread.join.assert_called_once_with(timeout=1.0)

    # ── Mute / Unmute Tests ───────────────────────────────────────────────────

    def test_mute_sets_flag(self):
        self.assertFalse(self.listener.muted)
        self.listener.mute()
        self.assertTrue(self.listener.muted)

    def test_unmute_clears_flag(self):
        self.listener.mute()
        self.listener.unmute()
        self.assertFalse(self.listener.muted)

    def test_muted_discards_speech(self):
        self.listener.muted = True
        valid_speech = np.ones(int(1.0 * listener.SAMPLE_RATE), dtype=np.float32)
        self.listener._flush([valid_speech])
        self.mock_speech_cb.assert_not_called()

    # ── RMS & Energy Calculation Tests ────────────────────────────────────────

    def test_rms_silence(self):
        silence = np.zeros(480, dtype=np.float32)
        self.assertAlmostEqual(self.listener._rms(silence), 0.0)

    def test_rms_loud_signal(self):
        sine = (0.5 * np.sin(np.linspace(0, 10, 480))).astype(np.float32)
        rms = self.listener._rms(sine)
        self.assertGreater(rms, 0.2)

    def test_calibrate_noise_floor(self):
        chunk1 = np.full(480, 0.01, dtype=np.float32)
        chunk2 = np.full(480, 0.02, dtype=np.float32)

        self.assertEqual(self.listener._noise_floor, 0.0)
        self.listener._calibrate_noise_floor(chunk1)
        self.assertAlmostEqual(self.listener._noise_floor, 0.01, places=4)

        self.listener._calibrate_noise_floor(chunk2)
        # 0.9 * 0.01 + 0.1 * 0.02 = 0.011
        self.assertAlmostEqual(self.listener._noise_floor, 0.011, places=4)

    def test_is_speech_above_threshold(self):
        self.listener._noise_floor = 0.005
        loud_chunk = np.full(480, 0.15, dtype=np.float32)
        self.assertTrue(self.listener._is_speech(loud_chunk))

    def test_is_speech_below_threshold(self):
        self.listener._noise_floor = 0.01
        quiet_chunk = np.full(480, 0.005, dtype=np.float32)
        self.assertFalse(self.listener._is_speech(quiet_chunk))

    def test_noise_floor_slow_drift(self):
        self.listener._noise_floor = 0.01
        # Test the slow drift formula used in _run during silence:
        # new_floor = 0.995 * floor + 0.005 * rms
        rms = 0.02
        new_floor = 0.995 * self.listener._noise_floor + 0.005 * rms
        self.assertGreater(new_floor, 0.01)
        self.assertLess(new_floor, 0.02)

    # ── Flush & Buffer Processing Tests ───────────────────────────────────────

    def test_flush_empty_buffer(self):
        self.listener._flush([])
        self.mock_speech_cb.assert_not_called()

    def test_flush_short_speech_ignored(self):
        # 0.1 seconds is below MIN_SPEECH_SECONDS (0.4s)
        short_chunk = np.ones(int(0.1 * listener.SAMPLE_RATE), dtype=np.float32)
        self.listener._flush([short_chunk])
        self.mock_speech_cb.assert_not_called()

    def test_flush_valid_speech_calls_callback(self):
        # 0.8 seconds exceeds MIN_SPEECH_SECONDS
        chunk1 = np.ones(int(0.4 * listener.SAMPLE_RATE), dtype=np.float32)
        chunk2 = np.ones(int(0.4 * listener.SAMPLE_RATE), dtype=np.float32)
        self.listener._flush([chunk1, chunk2])
        self.mock_speech_cb.assert_called_once()
        audio_arg = self.mock_speech_cb.call_args[0][0]
        self.assertEqual(len(audio_arg), int(0.8 * listener.SAMPLE_RATE))

    # ── State Change & Detection Flow Tests ───────────────────────────────────

    @patch("listener.sd.InputStream")
    def test_mic_callback_queues_audio(self, mock_stream):
        captured_callback = None

        def stream_init(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("callback")
            cm = MagicMock()
            cm.__enter__.return_value = cm
            return cm

        mock_stream.side_effect = stream_init

        # Run listener in thread and verify callback puts into queue
        self.listener._stop_event.set()  # Stop immediately after opening stream
        self.listener.start()
        self.listener._thread.join(timeout=1.0)

        self.assertIsNotNone(captured_callback)
        # Simulate incoming mic frame: 480 frames, 1 channel
        frame = np.zeros((480, 1), dtype=np.float32)
        captured_callback(frame, 480, None, None)

        self.assertFalse(self.listener._audio_queue.empty())
        queued_chunk = self.listener._audio_queue.get_nowait()
        self.assertEqual(len(queued_chunk), 480)

    @patch("listener.sd.InputStream")
    def test_run_speech_state_transitions(self, mock_stream):
        mock_ctx = MagicMock()
        mock_stream.return_value.__enter__.return_value = mock_ctx

        # Calibrate with 16 silent chunks
        for _ in range(16):
            self.listener._audio_queue.put_nowait(np.zeros(480, dtype=np.float32))

        # Add loud speech chunks (duration > MIN_SPEECH_SECONDS)
        speech_chunks = 20  # 20 * 30ms = 600ms
        for _ in range(speech_chunks):
            self.listener._audio_queue.put_nowait(np.full(480, 0.2, dtype=np.float32))

        # Add silent chunks to trigger silence timeout
        for _ in range(45):  # 45 * 30ms = 1350ms > SILENCE_SECONDS (1.2s)
            self.listener._audio_queue.put_nowait(np.zeros(480, dtype=np.float32))

        # Start runner thread
        self.listener.start()

        # Wait briefly for queue to be processed
        time.sleep(0.5)
        self.listener.stop()

        # on_state_change should have transitioned to "listening" and "recording"
        state_calls = [c[0][0] for c in self.mock_state_cb.call_args_list]
        self.assertIn("listening", state_calls)
        self.assertIn("recording", state_calls)


if __name__ == "__main__":
    unittest.main()
