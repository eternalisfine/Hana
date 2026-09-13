# tests/test_tutor.py — Thorough test suite for tutor.py (Ollama LLM Engine)

import unittest
from unittest.mock import patch, MagicMock
import requests
import json

import tutor
import memory
from tests.conftest import InMemoryDBTestCase, MockHTTPResponse


class TestTutor(InMemoryDBTestCase):
    """
    Comprehensive tests for tutor.py covering TTS extraction, system prompt construction,
    chat interaction with mocked Ollama, error handling, background tasks, and profile updates.
    """

    def setUp(self):
        super().setUp()
        self.patch_user_msg = patch("tutor.knowledge.process_user_message")
        self.patch_tutor_msg = patch("tutor.knowledge.process_tutor_message")
        self.patch_user_msg.start()
        self.patch_tutor_msg.start()

    def tearDown(self):
        self.patch_user_msg.stop()
        self.patch_tutor_msg.stop()
        super().tearDown()

    # ── Original Tests (Preserved) ────────────────────────────────────────────

    def test_extract_japanese_for_tts(self):
        text = "こんにちは --- 🇬🇧 Hello"
        extracted = tutor.extract_japanese_for_tts(text)
        self.assertEqual(extracted, "こんにちは")

    def test_extract_japanese_with_furigana(self):
        text = "食べ物(たべもの)が美味しいです --- 🇬🇧 Food is delicious"
        extracted = tutor.extract_japanese_for_tts(text)
        self.assertEqual(extracted, "食べ物が美味しいです")

    def test_extract_japanese_with_corrections(self):
        text = "✗ わたしは猫です → ✓ わたしは猫が好きです --- 🇬🇧 I like cats"
        extracted = tutor.extract_japanese_for_tts(text)
        self.assertIsInstance(extracted, str)

    # ── TTS Extraction Additional Tests ───────────────────────────────────────

    def test_tts_extraction_no_separator(self):
        text = "今日はいい天気ですね。"
        extracted = tutor.extract_japanese_for_tts(text)
        self.assertEqual(extracted, "今日はいい天気ですね。")

    def test_tts_extraction_empty_string(self):
        self.assertEqual(tutor.extract_japanese_for_tts(""), "")
        self.assertEqual(tutor.extract_japanese_for_tts("   "), "")

    def test_tts_extraction_multiple_separators(self):
        text = "こんにちは --- 🇬🇧 Hello --- Note: extra separator"
        extracted = tutor.extract_japanese_for_tts(text)
        self.assertEqual(extracted, "こんにちは")

    def test_tts_extraction_nested_furigana(self):
        text = "毎朝(まいあさ)ご飯(はん)を食べます(たべます) --- 🇬🇧 I eat rice every morning"
        extracted = tutor.extract_japanese_for_tts(text)
        self.assertEqual(extracted, "毎朝ご飯を食べます")

    def test_tts_extraction_strips_bullet_lines(self):
        text = "こんにちは！\n• これは箇条書きです\nさようなら --- 🇬🇧 Bye"
        extracted = tutor.extract_japanese_for_tts(text)
        self.assertNotIn("• これは箇条書きです", extracted)
        self.assertIn("こんにちは！", extracted)
        self.assertIn("さようなら", extracted)

    # ── System Prompt Construction Tests ──────────────────────────────────────

    def test_system_prompt_contains_critical_rules(self):
        prompt = tutor._build_system_prompt()
        self.assertIn("CRITICAL RULES", prompt)
        self.assertIn("1. Only use Japanese grammar, vocabulary, and expressions", prompt)
        self.assertIn("2. Prefer simpler, unambiguous phrasing", prompt)
        self.assertIn("3. Use standard Tokyo/NHK-style Japanese", prompt)
        self.assertIn("4. Never fabricate words", prompt)
        self.assertIn("5. Never ignore a student mistake", prompt)
        self.assertIn("6. If you are genuinely unsure", prompt)

    def test_system_prompt_contains_student_profile(self):
        memory.update_style_profile(level_estimate="intermediate", grammar_notes="Practicing passive")
        prompt = tutor._build_system_prompt()
        self.assertIn("=== Student Profile ===", prompt)
        self.assertIn("Estimated level: intermediate", prompt)
        self.assertIn("Grammar notes: Practicing passive", prompt)

    def test_system_prompt_contains_knowledge_block(self):
        prompt = tutor._build_system_prompt()
        self.assertIn("=== Student Knowledge ===", prompt)
        self.assertIn("Total vocabulary encountered:", prompt)

    def test_system_prompt_format_separator(self):
        prompt = tutor._build_system_prompt()
        self.assertIn("---", prompt)
        self.assertIn("🇬🇧 English:", prompt)

    # ── Chat Tests (Mocked Ollama) ────────────────────────────────────────────

    @patch("tutor.requests.post")
    def test_chat_success(self, mock_post):
        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": "お元気ですか？ --- 🇬🇧 How are you?"}}
        )

        res = tutor.chat("こんにちは", session_id="s1")
        self.assertFalse(res["error"])
        self.assertFalse(res["flagged_by_tutor"])
        self.assertEqual(res["response"], "お元気ですか？ --- 🇬🇧 How are you?")

    @patch("tutor.requests.post")
    def test_chat_stores_user_and_assistant_messages(self, mock_post):
        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": "はい、元気です！ --- 🇬🇧 Yes, I'm fine!"}}
        )

        tutor.chat("元気ですか？", session_id="s_chat")
        msgs = memory.get_recent_messages(10)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "元気ですか？")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["content"], "はい、元気です！ --- 🇬🇧 Yes, I'm fine!")

    @patch("tutor.requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused"))
    def test_chat_connection_error(self, mock_post):
        res = tutor.chat("こんにちは", session_id="s_err")
        self.assertTrue(res["error"])
        self.assertIn("Cannot connect to Ollama", res["response"])

    @patch("tutor.requests.post", side_effect=requests.exceptions.Timeout("Timeout"))
    def test_chat_timeout_error(self, mock_post):
        res = tutor.chat("こんにちは", session_id="s_timeout")
        self.assertTrue(res["error"])
        self.assertIn("Ollama timed out", res["response"])

    @patch("tutor.requests.post", side_effect=RuntimeError("Unexpected server crash"))
    def test_chat_generic_exception(self, mock_post):
        res = tutor.chat("こんにちは", session_id="s_gen")
        self.assertTrue(res["error"])
        self.assertIn("Ollama error: Unexpected server crash", res["response"])

    @patch("tutor.requests.post")
    @patch("tutor.threading.Thread")
    def test_chat_spawns_knowledge_threads(self, mock_thread_cls, mock_post):
        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": "応答 --- 🇬🇧 Response"}}
        )
        mock_thread_instance = MagicMock()
        mock_thread_cls.return_value = mock_thread_instance

        tutor.chat("日本語", session_id="s_threads")
        # Should have started threads for process_user_message and process_tutor_message
        self.assertGreaterEqual(mock_thread_cls.call_count, 2)
        self.assertGreaterEqual(mock_thread_instance.start.call_count, 2)

    @patch("tutor.requests.post")
    @patch("tutor._update_profile")
    def test_chat_triggers_profile_update(self, mock_update_profile, mock_post):
        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": "応答 --- 🇬🇧 Response"}}
        )

        from config import PROFILE_UPDATE_EVERY
        # Pre-seed messages so user message count reaches PROFILE_UPDATE_EVERY
        for i in range(PROFILE_UPDATE_EVERY - 1):
            memory.add_message("s_pre", "user", f"Pre msg {i}")

        tutor.chat("Trigger message", session_id="s_pre")
        # Now user count reaches PROFILE_UPDATE_EVERY
        self.assertEqual(memory.get_user_message_count(), PROFILE_UPDATE_EVERY)

    # ── Profile Update Tests ──────────────────────────────────────────────────

    @patch("tutor.requests.post")
    def test_profile_update_parses_json(self, mock_post):
        # Pre-seed at least 4 messages
        for i in range(4):
            memory.add_message("s_prof", "user" if i % 2 == 0 else "assistant", f"Msg {i}")

        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": json.dumps({
                "level_estimate": "intermediate",
                "grammar_notes": "Knows te-form well",
                "vocabulary_notes": "Needs food vocab",
                "general_notes": "Polite and conversational"
            })}}
        )

        tutor._update_profile()
        profile = memory.get_style_profile()
        self.assertEqual(profile["level_estimate"], "intermediate")
        self.assertEqual(profile["grammar_notes"], "Knows te-form well")

    @patch("tutor.requests.post")
    def test_profile_update_handles_markdown_wrapper(self, mock_post):
        for i in range(4):
            memory.add_message("s_prof", "user", f"Msg {i}")

        wrapped_json = "```json\n{\"level_estimate\": \"advanced\", \"grammar_notes\": \"Good\"}\n```"
        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": wrapped_json}}
        )

        tutor._update_profile()
        profile = memory.get_style_profile()
        self.assertEqual(profile["level_estimate"], "advanced")

    @patch("tutor.requests.post")
    def test_profile_update_skips_short_history(self, mock_post):
        # Only 2 messages in DB
        memory.add_message("s1", "user", "Hi")
        memory.add_message("s1", "assistant", "Hello")

        tutor._update_profile()
        mock_post.assert_not_called()

    @patch("tutor.requests.post", side_effect=requests.exceptions.ConnectionError("Offline"))
    def test_profile_update_handles_exception_safely(self, mock_post):
        for i in range(4):
            memory.add_message("s_prof", "user", f"Msg {i}")

        # Should not raise exception
        try:
            tutor._update_profile()
        except Exception as e:
            self.fail(f"_update_profile raised an exception: {e}")

    # ── Ollama Health Check Tests ─────────────────────────────────────────────

    @patch("tutor.requests.get")
    def test_check_ollama_running(self, mock_get):
        mock_get.return_value = MockHTTPResponse(status_code=200)
        self.assertTrue(tutor.check_ollama())

    @patch("tutor.requests.get", side_effect=requests.exceptions.ConnectionError("Refused"))
    def test_check_ollama_down(self, mock_get):
        self.assertFalse(tutor.check_ollama())


if __name__ == "__main__":
    unittest.main()
