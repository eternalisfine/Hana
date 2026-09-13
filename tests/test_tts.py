# tests/test_tts.py — Thorough test suite for tts.py (VOICEVOX TTS Player)

import unittest
from unittest.mock import patch, MagicMock
import io
import time
import threading
import numpy as np
import soundfile as sf

import tts
from tests.conftest import MockHTTPResponse, sample_audio


def _create_dummy_wav_bytes(duration: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Helper to generate valid WAV bytes in memory."""
    buf = io.BytesIO()
    audio = sample_audio(duration=duration, sample_rate=sample_rate, kind="sine")
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestTTS(unittest.TestCase):
    """
    Comprehensive tests for TTSPlayer covering playback, interruption,
    synthesis API, audio chunk streaming, error handling, and callbacks.
    """

    def setUp(self):
        self.player = tts.TTSPlayer()

    def tearDown(self):
        self.player.interrupt()

    # ── Core Playback & Threading Tests ───────────────────────────────────────

    @patch.object(tts.TTSPlayer, "_play")
    @patch.object(tts.TTSPlayer, "_synthesize")
    def test_speak_calls_synthesize_and_play(self, mock_synth, mock_play):
        dummy_audio = sample_audio(0.1)
        mock_synth.return_value = (dummy_audio, 16000)

        self.player.speak("こんにちは")
        # Wait for worker thread to complete
        if self.player._thread:
            self.player._thread.join(timeout=1.0)

        mock_synth.assert_called_once_with("こんにちは")
        mock_play.assert_called_once_with(dummy_audio, 16000)

    @patch.object(tts.TTSPlayer, "_synthesize")
    def test_speak_empty_string_noop(self, mock_synth):
        self.player.speak("")
        mock_synth.assert_not_called()
        self.assertFalse(self.player.is_speaking())

    @patch.object(tts.TTSPlayer, "_synthesize")
    def test_speak_whitespace_only_noop(self, mock_synth):
        self.player.speak("   \n\t  ")
        mock_synth.assert_not_called()
        self.assertFalse(self.player.is_speaking())

    @patch.object(tts.TTSPlayer, "interrupt")
    @patch.object(tts.TTSPlayer, "_worker")
    def test_speak_interrupts_previous(self, mock_worker, mock_interrupt):
        self.player.speak("新メッセージ")
        mock_interrupt.assert_called_once()

    # ── Interruption Tests ────────────────────────────────────────────────────

    def test_interrupt_sets_stop_flag(self):
        self.player._stop_flag.clear()
        self.player.interrupt()
        self.assertTrue(self.player._stop_flag.is_set())

    def test_interrupt_when_not_speaking(self):
        # Should not raise exception
        try:
            self.player.interrupt()
        except Exception as e:
            self.fail(f"interrupt() raised exception: {e}")

    def test_interrupt_joins_thread(self):
        started = threading.Event()

        def slow_worker(text):
            started.set()
            time.sleep(0.3)

        self.player._worker = slow_worker
        self.player.speak("ゆっくり")
        started.wait(timeout=1.0)
        self.assertTrue(self.player.is_speaking())

        self.player.interrupt()
        self.assertFalse(self.player.is_speaking())

    def test_is_speaking_state(self):
        self.assertFalse(self.player.is_speaking())
        dummy_thread = threading.Thread(target=lambda: time.sleep(0.1))
        self.player._thread = dummy_thread
        dummy_thread.start()
        self.assertTrue(self.player.is_speaking())
        dummy_thread.join()
        self.assertFalse(self.player.is_speaking())

    # ── Synthesis Tests (Mocked HTTP) ─────────────────────────────────────────

    @patch("tts.requests.post")
    def test_synthesize_sends_audio_query_and_synthesis(self, mock_post):
        wav_bytes = _create_dummy_wav_bytes(0.1)

        # First call: /audio_query, Second call: /synthesis
        mock_post.side_effect = [
            MockHTTPResponse(200, json_data={"speedScale": 1.0, "intonationScale": 1.0}),
            MockHTTPResponse(200, content=wav_bytes),
        ]

        audio_data, sr = self.player._synthesize("テスト")
        self.assertIsNotNone(audio_data)
        self.assertEqual(sr, 16000)
        self.assertEqual(mock_post.call_count, 2)

        # Check audio_query params
        first_call = mock_post.call_args_list[0]
        self.assertIn("audio_query", first_call[0][0])
        self.assertEqual(first_call[1]["params"]["text"], "テスト")

        # Check synthesis params
        second_call = mock_post.call_args_list[1]
        self.assertIn("synthesis", second_call[0][0])
        self.assertEqual(second_call[1]["json"]["speedScale"], self.player.speed)
        self.assertEqual(second_call[1]["json"]["intonationScale"], 1.1)

    @patch("tts.requests.post")
    def test_synthesize_audio_query_fails(self, mock_post):
        mock_post.return_value = MockHTTPResponse(500, text="Internal Server Error")
        audio_data, sr = self.player._synthesize("失敗テスト")
        self.assertIsNone(audio_data)
        self.assertIsNone(sr)

    @patch("tts.requests.post")
    def test_synthesize_synthesis_fails(self, mock_post):
        mock_post.side_effect = [
            MockHTTPResponse(200, json_data={"speedScale": 1.0}),
            MockHTTPResponse(500, text="Synthesis failed"),
        ]
        audio_data, sr = self.player._synthesize("合成失敗")
        self.assertIsNone(audio_data)
        self.assertIsNone(sr)

    @patch("tts.requests.post")
    def test_synthesize_custom_speed(self, mock_post):
        wav_bytes = _create_dummy_wav_bytes(0.1)
        self.player.speed = 1.25

        mock_post.side_effect = [
            MockHTTPResponse(200, json_data={}),
            MockHTTPResponse(200, content=wav_bytes),
        ]
        self.player._synthesize("スピード")
        synth_call = mock_post.call_args_list[1]
        self.assertEqual(synth_call[1]["json"]["speedScale"], 1.25)

    # ── Playback Tests (Mocked sounddevice) ────────────────────────────────────

    @patch("tts.sd.OutputStream")
    def test_play_streams_in_chunks(self, mock_output_stream):
        mock_stream_inst = MagicMock()
        mock_output_stream.return_value.__enter__.return_value = mock_stream_inst

        # 6000 samples > CHUNK (2048) -> should produce 3 chunks
        audio = np.zeros(6000, dtype=np.float32)
        self.player._play(audio, sample_rate=16000)

        self.assertEqual(mock_stream_inst.write.call_count, 3)

    @patch("tts.sd.OutputStream")
    def test_play_stops_on_flag(self, mock_output_stream):
        mock_stream_inst = MagicMock()
        mock_output_stream.return_value.__enter__.return_value = mock_stream_inst

        audio = np.zeros(10000, dtype=np.float32)

        def set_stop_on_first_write(chunk):
            self.player._stop_flag.set()

        mock_stream_inst.write.side_effect = set_stop_on_first_write

        self.player._stop_flag.clear()
        self.player._play(audio, sample_rate=16000)
        # Should stop after first chunk
        self.assertEqual(mock_stream_inst.write.call_count, 1)

    @patch("tts.sd.OutputStream")
    def test_play_handles_mono_audio(self, mock_output_stream):
        mock_stream_inst = MagicMock()
        mock_output_stream.return_value.__enter__.return_value = mock_stream_inst

        audio_1d = np.zeros(2048, dtype=np.float32)
        self.player._play(audio_1d, sample_rate=16000)

        # Must be reshaped to (N, 1)
        written = mock_stream_inst.write.call_args[0][0]
        self.assertEqual(written.shape, (2048, 1))

    @patch("tts.sd.OutputStream")
    def test_play_handles_stereo_audio(self, mock_output_stream):
        mock_stream_inst = MagicMock()
        mock_output_stream.return_value.__enter__.return_value = mock_stream_inst

        audio_2d = np.zeros((2048, 2), dtype=np.float32)
        self.player._play(audio_2d, sample_rate=16000)

        # OutputStream opened with channels=2
        mock_output_stream.assert_called_once()
        self.assertEqual(mock_output_stream.call_args[1]["channels"], 2)

    # ── Status & Callbacks Tests ──────────────────────────────────────────────

    @patch("tts.requests.get")
    def test_is_running_voicevox_up(self, mock_get):
        mock_get.return_value = MockHTTPResponse(200)
        self.assertTrue(self.player.is_running())

    @patch("tts.requests.get", side_effect=Exception("Connection refused"))
    def test_is_running_voicevox_down(self, mock_get):
        self.assertFalse(self.player.is_running())

    @patch("tts.requests.get")
    def test_get_speakers_success(self, mock_get):
        mock_get.return_value = MockHTTPResponse(200, json_data=[{"name": "四国めたん"}])
        speakers = self.player.get_speakers()
        self.assertEqual(len(speakers), 1)
        self.assertEqual(speakers[0]["name"], "四国めたん")

    @patch("tts.requests.get", side_effect=Exception("Timeout"))
    def test_get_speakers_failure(self, mock_get):
        self.assertEqual(self.player.get_speakers(), [])

    @patch.object(tts.TTSPlayer, "_play")
    @patch.object(tts.TTSPlayer, "_synthesize")
    def test_on_start_and_on_end_callbacks(self, mock_synth, mock_play):
        mock_synth.return_value = (sample_audio(0.05), 16000)

        events = []
        self.player.on_start = lambda: events.append("start")
        self.player.on_end = lambda: events.append("end")

        self.player.speak("コールバックテスト")
        if self.player._thread:
            self.player._thread.join(timeout=1.0)

        self.assertEqual(events, ["start", "end"])


if __name__ == "__main__":
    unittest.main()
