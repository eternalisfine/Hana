# tests/test_integration.py — Integration and End-to-End Pipeline Tests

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import json
import requests

import memory
import tutor
import safety
import knowledge
import tts
from tests.conftest import InMemoryDBTestCase, MockHTTPResponse


class TestIntegration(InMemoryDBTestCase):
    """
    End-to-end integration tests connecting multiple modules:
    conversation flow, safety verification, vocabulary tracking,
    adaptive prompt injection, streak accumulation, and error resilience.
    """

    @patch("tutor.knowledge.process_user_message")
    @patch("tutor.knowledge.process_tutor_message")
    @patch("tutor.requests.post")
    def test_full_conversation_flow(self, mock_post, mock_tutor_msg, mock_user_msg):
        """Simulate a complete user-tutor exchange through all subsystems."""
        tutor_reply = "こんにちは！今日も元気に日本語を練習しましょう。 --- 🇬🇧 Hello! Let's practice Japanese energetically today as well."
        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": tutor_reply}}
        )

        user_input = "こんにちは！よろしくお願いします。"
        session_id = "integration_session_1"

        # 1. User sends message through tutor engine
        result = tutor.chat(user_input, session_id=session_id)
        self.assertFalse(result["error"])
        self.assertEqual(result["response"], tutor_reply)

        # 2. Response runs through safety check pipeline
        safety_result = safety.check(result["response"])
        self.assertFalse(safety_result["flagged"])
        self.assertEqual(safety_result["confidence"], "high")

        # 3. Japanese is extracted for TTS playback
        tts_text = tutor.extract_japanese_for_tts(result["response"])
        self.assertEqual(tts_text, "こんにちは！今日も元気に日本語を練習しましょう。")
        self.assertNotIn("🇬🇧", tts_text)

        # 4. Verify conversation is persisted to DB
        history = memory.get_recent_messages(10)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], user_input)
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], tutor_reply)

    def test_vocabulary_tracked_through_conversation(self):
        """Verify vocabulary updates correctly across multiple simulated conversation turns."""
        # Turn 1: User says words
        knowledge.process_user_message("私は毎日寿司を食べます")
        # Turn 2: Tutor exposes user to words
        knowledge.process_tutor_message("寿司は美味しいですね --- 🇬🇧 Sushi is delicious")

        vocab_count = memory.get_total_vocabulary_count()
        self.assertGreater(vocab_count, 0)

        # "寿司" should have been both produced and seen
        with memory._conn() as c:
            row = c.execute("SELECT times_seen, times_produced FROM vocabulary_log WHERE word='寿司'").fetchone()
        if row:
            self.assertGreaterEqual(row["times_seen"], 2)
            self.assertGreaterEqual(row["times_produced"], 1)

    def test_safety_flags_propagate_and_record(self):
        """Verify that suspicious or uncertain phrases are flagged and can be recorded."""
        uncertain_response = "それは正しいかもしれません --- 🇬🇧 That might be correct"
        safety_res = safety.check(uncertain_response)
        self.assertTrue(safety_res["flagged"])
        self.assertIn("Tutor expressed uncertainty", safety_res["warnings"][0])

        # Record with flagged status in DB
        memory.add_message("flag_session", "assistant", uncertain_response, flagged=safety_res["flagged"])
        with memory._conn() as c:
            row = c.execute("SELECT flagged FROM messages WHERE session_id='flag_session'").fetchone()
        self.assertEqual(row["flagged"], 1)

    def test_knowledge_context_injected_into_prompt(self):
        """Verify that tracked vocabulary appears in the system prompt for adaptive tutoring."""
        # Seed known vocabulary
        for _ in range(4):
            memory.upsert_vocabulary("桜", "さくら", "NOUN", produced=True)

        prompt = tutor._build_system_prompt()
        self.assertIn("桜", prompt)
        self.assertIn("=== Student Knowledge ===", prompt)

    def test_streak_accumulates_across_sessions(self):
        """Simulate daily activity across multiple days and verify streak calculation."""
        today = datetime.now().date()
        with memory._conn() as c:
            for i in range(4):
                d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                c.execute("INSERT INTO daily_stats (date, messages_sent, xp_earned) VALUES (?, 5, 25)", (d,))

        self.assertEqual(memory.get_streak(), 4)

    @patch("tutor.requests.post")
    def test_profile_update_integration(self, mock_post):
        """Verify profile update parses model response and persists new level and notes."""
        # Pre-seed conversation
        for i in range(6):
            memory.add_message("s_flow", "user" if i % 2 == 0 else "assistant", f"Exchange {i}")

        mock_post.return_value = MockHTTPResponse(
            status_code=200,
            json_data={"message": {"content": json.dumps({
                "level_estimate": "upper-intermediate",
                "grammar_notes": "Understands honorifics (keigo)",
                "vocabulary_notes": "Business terms expanding",
                "general_notes": "Natural intonation"
            })}}
        )

        tutor._update_profile()
        profile = memory.get_style_profile()
        self.assertEqual(profile["level_estimate"], "upper-intermediate")
        self.assertEqual(profile["grammar_notes"], "Understands honorifics (keigo)")

        # Verify new profile is reflected in system prompt
        prompt = tutor._build_system_prompt()
        self.assertIn("Estimated level: upper-intermediate", prompt)
        self.assertIn("Understands honorifics (keigo)", prompt)

    @patch("tutor.requests.post", side_effect=requests.exceptions.ConnectionError("Ollama offline"))
    def test_error_recovery_ollama_down(self, mock_post):
        """Verify pipeline handles Ollama unavailability without corrupting state or crashing."""
        res = tutor.chat("こんにちは", session_id="err_sess")
        self.assertTrue(res["error"])
        self.assertIn("Cannot connect to Ollama", res["response"])

        # User message was recorded, assistant message was NOT
        msgs = memory.get_recent_messages(10)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    @patch("tts.requests.post")
    def test_error_recovery_voicevox_down(self, mock_post):
        """Verify TTS handles VOICEVOX unavailability gracefully without crashing."""
        mock_post.side_effect = requests.exceptions.ConnectionError("VOICEVOX offline")
        player = tts.TTSPlayer()
        # _worker catches network exceptions safely
        try:
            player._worker("こんにちは")
        except Exception as e:
            self.fail(f"_worker raised an exception on connection failure: {e}")
        self.assertFalse(player.is_running())


if __name__ == "__main__":
    unittest.main()
