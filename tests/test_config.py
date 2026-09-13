# tests/test_config.py — Configuration sanity and constraint tests

import unittest
import config


class TestConfig(unittest.TestCase):
    """
    Sanity checks on configuration constants to prevent misconfigurations
    that could break the audio pipeline or HTTP API clients.
    """

    def test_ollama_url_format(self):
        self.assertTrue(config.OLLAMA_URL.startswith("http://") or config.OLLAMA_URL.startswith("https://"))
        self.assertIn("/api/chat", config.OLLAMA_URL)
        self.assertTrue(len(config.OLLAMA_MODEL) > 0)

    def test_voicevox_url_format(self):
        self.assertTrue(config.VOICEVOX_URL.startswith("http://") or config.VOICEVOX_URL.startswith("https://"))
        self.assertIsInstance(config.VOICEVOX_SPEAKER_ID, int)
        self.assertGreaterEqual(config.VOICEVOX_SPEAKER_ID, 0)

    def test_vad_threshold_range(self):
        self.assertIsInstance(config.VAD_THRESHOLD, (int, float))
        self.assertGreaterEqual(config.VAD_THRESHOLD, 0.0)
        self.assertLessEqual(config.VAD_THRESHOLD, 1.0)

    def test_silence_seconds_positive(self):
        self.assertGreater(config.SILENCE_SECONDS, 0.0)
        # Should be reasonable conversation pause (between 0.5s and 5.0s)
        self.assertLessEqual(config.SILENCE_SECONDS, 5.0)

    def test_min_speech_seconds_positive(self):
        self.assertGreater(config.MIN_SPEECH_SECONDS, 0.0)
        self.assertLess(config.MIN_SPEECH_SECONDS, config.SILENCE_SECONDS)

    def test_context_messages_positive(self):
        self.assertIsInstance(config.CONTEXT_MESSAGES, int)
        self.assertGreater(config.CONTEXT_MESSAGES, 0)

    def test_whisper_config_valid(self):
        self.assertEqual(config.WHISPER_LANGUAGE, "ja")
        self.assertIn(config.WHISPER_MODEL, ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])

    def test_daily_goals_and_vocab_thresholds(self):
        self.assertGreater(config.DAILY_XP_GOAL, 0)
        self.assertGreaterEqual(config.VOCAB_KNOWN_THRESHOLD, 1)
        self.assertGreater(config.PROFILE_UPDATE_EVERY, 0)


if __name__ == "__main__":
    unittest.main()
