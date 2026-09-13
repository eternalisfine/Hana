# tests/test_nlp_core.py — Thorough test suite for nlp_core.py (Shared GiNZA/spaCy Init)

import unittest
from unittest.mock import patch, MagicMock
import threading

import nlp_core


class TestNLPCore(unittest.TestCase):
    """
    Comprehensive tests for nlp_core.py covering lazy initialization, thread-safe access,
    success/failure handling for spacy/ginza loading, and safe text processing.
    """

    def setUp(self):
        # Save original state to restore in tearDown
        self.orig_instance = nlp_core._nlp_instance
        self.orig_ok = nlp_core._ginza_ok
        self.orig_initialized = nlp_core._initialized

    def tearDown(self):
        # Restore state
        nlp_core._nlp_instance = self.orig_instance
        nlp_core._ginza_ok = self.orig_ok
        nlp_core._initialized = self.orig_initialized

    def _reset_nlp_state(self):
        nlp_core._nlp_instance = None
        nlp_core._ginza_ok = False
        nlp_core._initialized = False

    # ── Initialization Tests ──────────────────────────────────────────────────

    def test_get_nlp_lazy_init(self):
        self._reset_nlp_state()
        with patch.object(nlp_core, "_init_ginza") as mock_init:
            def fake_init():
                nlp_core._nlp_instance = "fake_nlp"
                nlp_core._ginza_ok = True
                nlp_core._initialized = True
            mock_init.side_effect = fake_init

            nlp, ok = nlp_core.get_nlp()
            self.assertEqual(mock_init.call_count, 1)
            self.assertEqual(nlp, "fake_nlp")
            self.assertTrue(ok)

            # Second call should not call _init_ginza again
            nlp2, ok2 = nlp_core.get_nlp()
            self.assertEqual(mock_init.call_count, 1)
            self.assertIs(nlp, nlp2)

    def test_get_nlp_thread_safe(self):
        self._reset_nlp_state()
        init_call_count = 0
        lock = threading.Lock()

        def slow_init():
            nonlocal init_call_count
            with lock:
                init_call_count += 1
            nlp_core._nlp_instance = "fake_nlp"
            nlp_core._ginza_ok = True
            nlp_core._initialized = True

        with patch.object(nlp_core, "_init_ginza", side_effect=slow_init):
            threads = [threading.Thread(target=nlp_core.get_nlp) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # _init_ginza should be called exactly once due to _nlp_lock
            self.assertEqual(init_call_count, 1)

    @patch("spacy.load")
    def test_init_ginza_success(self, mock_spacy_load):
        self._reset_nlp_state()
        mock_model = MagicMock()
        mock_spacy_load.return_value = mock_model

        nlp_core._init_ginza()
        self.assertTrue(nlp_core._initialized)
        self.assertTrue(nlp_core._ginza_ok)
        self.assertIs(nlp_core._nlp_instance, mock_model)

    @patch("spacy.load", side_effect=Exception("Model not found"))
    def test_init_ginza_failure_handled_gracefully(self, mock_spacy_load):
        self._reset_nlp_state()
        try:
            nlp_core._init_ginza()
        except Exception as e:
            self.fail(f"_init_ginza() raised an exception: {e}")

        self.assertTrue(nlp_core._initialized)
        self.assertFalse(nlp_core._ginza_ok)
        self.assertIsNone(nlp_core._nlp_instance)

    # ── Safe Processing Tests ─────────────────────────────────────────────────

    def test_process_text_safely_success(self):
        mock_model = MagicMock()
        mock_doc = MagicMock()
        mock_model.return_value = mock_doc

        nlp_core._nlp_instance = mock_model
        nlp_core._ginza_ok = True
        nlp_core._initialized = True

        res = nlp_core.process_text_safely("テスト文")
        self.assertIs(res, mock_doc)
        mock_model.assert_called_once_with("テスト文")

    def test_process_text_safely_nlp_unavailable(self):
        nlp_core._nlp_instance = None
        nlp_core._ginza_ok = False
        nlp_core._initialized = True

        res = nlp_core.process_text_safely("テスト文")
        self.assertIsNone(res)

    def test_process_text_safely_handles_inference_exception(self):
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("Inference memory fault")

        nlp_core._nlp_instance = mock_model
        nlp_core._ginza_ok = True
        nlp_core._initialized = True

        res = nlp_core.process_text_safely("エラーテスト")
        self.assertIsNone(res)

    def test_process_text_safely_uses_lock(self):
        mock_model = MagicMock()
        nlp_core._nlp_instance = mock_model
        nlp_core._ginza_ok = True
        nlp_core._initialized = True

        mock_lock = MagicMock()
        with patch.object(nlp_core, "_nlp_lock", mock_lock):
            nlp_core.process_text_safely("ロックテスト")
            self.assertTrue(mock_lock.__enter__.called)


if __name__ == "__main__":
    unittest.main()
