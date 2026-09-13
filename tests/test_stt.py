# tests/test_stt.py — Thorough test suite for stt.py (Whisper Speech-to-Text)

import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import numpy as np

import stt
from config import WHISPER_MODEL, WHISPER_LANGUAGE
from tests.conftest import sample_audio


class TestSTT(unittest.TestCase):
    """
    Comprehensive tests for stt.py covering lazy model loading,
    audio formatting, transcription segment concatenation, and parameter passing.
    """

    def setUp(self):
        # Reset module-level singleton before each test
        stt._model = None

    def tearDown(self):
        stt._model = None

    # ── Model Loading Tests ───────────────────────────────────────────────────

    @patch("faster_whisper.WhisperModel")
    def test_load_model_lazy_initialization(self, mock_whisper_cls):
        mock_instance = MagicMock()
        mock_whisper_cls.return_value = mock_instance

        # First call creates the model
        model1 = stt.load_model()
        self.assertEqual(mock_whisper_cls.call_count, 1)

        # Second call returns cached instance without re-instantiation
        model2 = stt.load_model()
        self.assertEqual(mock_whisper_cls.call_count, 1)
        self.assertIs(model1, model2)

    @patch("faster_whisper.WhisperModel")
    def test_load_model_uses_config_values(self, mock_whisper_cls):
        stt.load_model()
        mock_whisper_cls.assert_called_once_with(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

    @patch("faster_whisper.WhisperModel")
    def test_load_model_cpu_device(self, mock_whisper_cls):
        stt.load_model()
        call_kwargs = mock_whisper_cls.call_args[1]
        self.assertEqual(call_kwargs["device"], "cpu")
        self.assertEqual(call_kwargs["compute_type"], "int8")

    # ── Transcription Tests ───────────────────────────────────────────────────

    @patch.object(stt, "load_model")
    def test_transcribe_returns_text(self, mock_load):
        mock_model = MagicMock()
        segment = SimpleNamespace(text="こんにちは")
        mock_model.transcribe.return_value = ([segment], None)
        mock_load.return_value = mock_model

        audio = sample_audio(0.5)
        result = stt.transcribe(audio)
        self.assertEqual(result, "こんにちは")

    @patch.object(stt, "load_model")
    def test_transcribe_empty_segments(self, mock_load):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        mock_load.return_value = mock_model

        audio = sample_audio(0.5)
        result = stt.transcribe(audio)
        self.assertEqual(result, "")

    @patch.object(stt, "load_model")
    def test_transcribe_multiple_segments(self, mock_load):
        mock_model = MagicMock()
        segments = [
            SimpleNamespace(text="はい、"),
            SimpleNamespace(text="わかりました。"),
        ]
        mock_model.transcribe.return_value = (segments, None)
        mock_load.return_value = mock_model

        audio = sample_audio(0.5)
        result = stt.transcribe(audio)
        self.assertEqual(result, "はい、 わかりました。")

    @patch.object(stt, "load_model")
    def test_transcribe_strips_whitespace(self, mock_load):
        mock_model = MagicMock()
        segments = [
            SimpleNamespace(text="   おはようございます   \n"),
        ]
        mock_model.transcribe.return_value = (segments, None)
        mock_load.return_value = mock_model

        audio = sample_audio(0.5)
        result = stt.transcribe(audio)
        self.assertEqual(result, "おはようございます")

    @patch("faster_whisper.WhisperModel")
    def test_transcribe_auto_loads_model(self, mock_whisper_cls):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([SimpleNamespace(text="テスト")], None)
        mock_whisper_cls.return_value = mock_model

        self.assertIsNone(stt._model)
        result = stt.transcribe(sample_audio(0.1))
        self.assertEqual(result, "テスト")
        mock_whisper_cls.assert_called_once()

    @patch.object(stt, "load_model")
    def test_transcribe_passes_vad_params(self, mock_load):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        mock_load.return_value = mock_model

        audio = sample_audio(0.5)
        stt.transcribe(audio)

        call_kwargs = mock_model.transcribe.call_args[1]
        self.assertEqual(call_kwargs["language"], WHISPER_LANGUAGE)
        self.assertEqual(call_kwargs["beam_size"], 5)
        self.assertEqual(call_kwargs["temperature"], 0.0)
        self.assertTrue(call_kwargs["vad_filter"])
        self.assertIn("vad_parameters", call_kwargs)
        self.assertEqual(call_kwargs["vad_parameters"]["min_silence_duration_ms"], 400)
        self.assertEqual(call_kwargs["vad_parameters"]["speech_pad_ms"], 200)

    @patch("soundfile.write")
    @patch.object(stt, "load_model")
    def test_transcribe_wav_format(self, mock_load, mock_sf_write):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        mock_load.return_value = mock_model

        audio = sample_audio(0.5)
        stt.transcribe(audio, sample_rate=16000)

        mock_sf_write.assert_called_once()
        args, kwargs = mock_sf_write.call_args
        self.assertEqual(kwargs["format"], "WAV")
        self.assertEqual(kwargs["subtype"], "PCM_16")


if __name__ == "__main__":
    unittest.main()
