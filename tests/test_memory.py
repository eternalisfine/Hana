# tests/test_memory.py — Thorough test suite for memory.py (SQLite DB layer)

import unittest
from unittest.mock import patch
import threading
from datetime import datetime, timedelta

import memory
from tests.conftest import InMemoryDBTestCase


class TestMemory(InMemoryDBTestCase):
    """
    Unit tests for memory.py covering all tables: messages, mistakes,
    style_profile, vocabulary_log, daily_stats, along with edge cases and concurrency.
    """

    # ── Original Tests (Preserved) ────────────────────────────────────────────

    def test_add_and_get_messages(self):
        memory.add_message("session_1", "user", "こんにちは")
        memory.add_message("session_1", "assistant", "こんにちは！元気ですか？")
        
        messages = memory.get_recent_messages(10)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "こんにちは")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "こんにちは！元気ですか？")

    def test_vocabulary_tracking(self):
        memory.upsert_vocabulary("猫", "ねこ", "NOUN", produced=False)
        memory.upsert_vocabulary("猫", "ねこ", "NOUN", produced=True)
        memory.upsert_vocabulary("猫", "ねこ", "NOUN", produced=True)
        
        vocab = memory.get_recent_vocabulary(10)
        self.assertEqual(len(vocab), 1)
        self.assertEqual(vocab[0]["word"], "猫")
        self.assertEqual(vocab[0]["times_seen"], 3)
        self.assertEqual(vocab[0]["times_produced"], 2)

    def test_streak_calculation(self):
        self.assertEqual(memory.get_streak(), 0)
        memory.log_daily_activity(messages=1)
        self.assertEqual(memory.get_streak(), 1)

    # ── Messages Tests ────────────────────────────────────────────────────────

    def test_add_message_stores_all_fields(self):
        memory.add_message("sess_abc", "user", "テストメッセージ", flagged=False)
        with memory._conn() as c:
            row = c.execute("SELECT * FROM messages WHERE session_id='sess_abc'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["session_id"], "sess_abc")
        self.assertEqual(row["role"], "user")
        self.assertEqual(row["content"], "テストメッセージ")
        self.assertEqual(row["flagged"], 0)
        self.assertTrue(len(row["timestamp"]) > 0)

    def test_get_recent_messages_ordering(self):
        memory.add_message("s1", "user", "First")
        memory.add_message("s1", "assistant", "Second")
        memory.add_message("s1", "user", "Third")
        
        msgs = memory.get_recent_messages(10)
        self.assertEqual([m["content"] for m in msgs], ["First", "Second", "Third"])

    def test_get_recent_messages_limit(self):
        for i in range(25):
            memory.add_message("s1", "user", f"Msg {i}")
        
        msgs = memory.get_recent_messages(5)
        self.assertEqual(len(msgs), 5)
        self.assertEqual(msgs[0]["content"], "Msg 20")
        self.assertEqual(msgs[-1]["content"], "Msg 24")

    def test_get_recent_messages_empty_db(self):
        msgs = memory.get_recent_messages(10)
        self.assertEqual(msgs, [])

    def test_add_message_with_unicode(self):
        complex_text = "日本語とEnglish 🎉 食べ物(たべもの) 【漢字】「テスト」\n改行あり"
        memory.add_message("s_uni", "user", complex_text)
        msgs = memory.get_recent_messages(1)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], complex_text)

    def test_add_message_flagged_true(self):
        memory.add_message("s_flag", "assistant", "懸念のある回答", flagged=True)
        with memory._conn() as c:
            row = c.execute("SELECT flagged FROM messages WHERE session_id='s_flag'").fetchone()
        self.assertEqual(row["flagged"], 1)

    def test_get_user_message_count(self):
        self.assertEqual(memory.get_user_message_count(), 0)
        memory.add_message("s1", "user", "Hello")
        memory.add_message("s1", "assistant", "Hi")
        memory.add_message("s1", "user", "How are you?")
        self.assertEqual(memory.get_user_message_count(), 2)

    def test_get_user_message_count_excludes_assistant(self):
        for _ in range(5):
            memory.add_message("s1", "assistant", "Response")
        self.assertEqual(memory.get_user_message_count(), 0)

    # ── Mistakes Tests ────────────────────────────────────────────────────────

    def test_log_mistake_new(self):
        memory.log_mistake("食べますでした", "食べませんでした", "verb_tense")
        with memory._conn() as c:
            row = c.execute("SELECT * FROM mistakes").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["original"], "食べますでした")
        self.assertEqual(row["correction"], "食べませんでした")
        self.assertEqual(row["mistake_type"], "verb_tense")
        self.assertEqual(row["occurrences"], 1)

    def test_log_mistake_duplicate_increments(self):
        memory.log_mistake("猫を好き", "猫が好き", "particle")
        memory.log_mistake("猫を好き", "猫が好き", "particle")
        memory.log_mistake("猫を好き", "猫が好き", "particle")
        
        with memory._conn() as c:
            rows = c.execute("SELECT * FROM mistakes WHERE original='猫を好き'").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["occurrences"], 3)

    def test_log_mistake_different_corrections(self):
        memory.log_mistake("いい", "よい", "vocab")
        memory.log_mistake("いい", "いいです", "politeness")
        
        with memory._conn() as c:
            rows = c.execute("SELECT * FROM mistakes WHERE original='いい'").fetchall()
        self.assertEqual(len(rows), 2)

    def test_get_mistake_summary_format(self):
        memory.log_mistake("犬は好き", "犬が好き", "particle")
        summary = memory.get_mistake_summary()
        self.assertIn("Said: 「犬は好き」", summary)
        self.assertIn("Correct: 「犬が好き」", summary)
        self.assertIn("[particle]", summary)
        self.assertIn("(×1)", summary)

    def test_get_mistake_summary_empty(self):
        self.assertEqual(memory.get_mistake_summary(), "No recorded mistakes yet.")

    def test_get_mistake_summary_ordered_by_occurrences(self):
        memory.log_mistake("MistakeA", "CorrectionA")
        memory.log_mistake("MistakeB", "CorrectionB")
        memory.log_mistake("MistakeB", "CorrectionB")
        
        summary = memory.get_mistake_summary()
        self.assertTrue(summary.find("MistakeB") < summary.find("MistakeA"))

    # ── Style Profile Tests ───────────────────────────────────────────────────

    def test_update_style_profile(self):
        memory.update_style_profile(
            level_estimate="intermediate",
            grammar_notes="Good with te-form, needs work on passive"
        )
        profile = memory.get_style_profile()
        self.assertEqual(profile["level_estimate"], "intermediate")
        self.assertEqual(profile["grammar_notes"], "Good with te-form, needs work on passive")

    def test_update_style_profile_ignores_unknown_keys(self):
        memory.update_style_profile(hacker_mode=True, invalid_column="drop table")
        profile = memory.get_style_profile()
        self.assertNotIn("hacker_mode", profile)
        self.assertEqual(profile["level_estimate"], "beginner")

    def test_update_style_profile_skips_empty_values(self):
        memory.update_style_profile(level_estimate="advanced")
        memory.update_style_profile(level_estimate="")
        profile = memory.get_style_profile()
        self.assertEqual(profile["level_estimate"], "advanced")

    def test_get_style_profile_default(self):
        profile = memory.get_style_profile()
        self.assertEqual(profile["id"], 1)
        self.assertEqual(profile["level_estimate"], "beginner")
        self.assertEqual(profile["grammar_notes"], "")

    def test_build_context_block_format(self):
        memory.add_message("s1", "user", "こんにちは")
        memory.update_style_profile(level_estimate="elementary", grammar_notes="N5 grammar")
        memory.log_mistake("これ", "それ")
        
        block = memory.build_context_block()
        self.assertIn("=== Student Profile ===", block)
        self.assertIn("Estimated level: elementary", block)
        self.assertIn("Grammar notes: N5 grammar", block)
        self.assertIn("=== Recurring Mistakes ===", block)
        self.assertIn("Said: 「これ」", block)

    # ── Vocabulary Tests ──────────────────────────────────────────────────────

    def test_upsert_vocabulary_new_word(self):
        memory.upsert_vocabulary("犬", "いぬ", "NOUN", produced=False, corrected=False)
        with memory._conn() as c:
            row = c.execute("SELECT * FROM vocabulary_log WHERE word='犬'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["reading"], "いぬ")
        self.assertEqual(row["pos"], "NOUN")
        self.assertEqual(row["times_seen"], 1)
        self.assertEqual(row["times_produced"], 0)
        self.assertEqual(row["times_corrected"], 0)

    def test_upsert_vocabulary_existing_increments_seen(self):
        memory.upsert_vocabulary("本", "ほん", "NOUN")
        memory.upsert_vocabulary("本", "ほん", "NOUN")
        with memory._conn() as c:
            row = c.execute("SELECT times_seen FROM vocabulary_log WHERE word='本'").fetchone()
        self.assertEqual(row["times_seen"], 2)

    def test_upsert_vocabulary_produced_flag(self):
        memory.upsert_vocabulary("走る", "はしる", "VERB", produced=True)
        with memory._conn() as c:
            row = c.execute("SELECT times_seen, times_produced FROM vocabulary_log WHERE word='走る'").fetchone()
        self.assertEqual(row["times_seen"], 1)
        self.assertEqual(row["times_produced"], 1)

    def test_upsert_vocabulary_corrected_flag(self):
        memory.upsert_vocabulary("言う", "いう", "VERB", corrected=True)
        with memory._conn() as c:
            row = c.execute("SELECT times_corrected FROM vocabulary_log WHERE word='言う'").fetchone()
        self.assertEqual(row["times_corrected"], 1)

    def test_get_known_vocabulary_threshold(self):
        for _ in range(3):
            memory.upsert_vocabulary("知る", "しる", "VERB")
        memory.upsert_vocabulary("忘れる", "わすれる", "VERB")
        
        known = memory.get_known_vocabulary(min_exposures=3)
        self.assertEqual(len(known), 1)
        self.assertEqual(known[0]["word"], "知る")

    def test_get_weak_vocabulary_never_produced(self):
        # Seen >= 2, produced == 0
        memory.upsert_vocabulary("読む", "よむ", "VERB", produced=False)
        memory.upsert_vocabulary("読む", "よむ", "VERB", produced=False)
        
        weak = memory.get_weak_vocabulary()
        words = [w["word"] for w in weak]
        self.assertIn("読む", words)

    def test_get_weak_vocabulary_corrected(self):
        # Corrected > 0, seen only once
        memory.upsert_vocabulary("間違い", "まちがい", "NOUN", corrected=True)
        weak = memory.get_weak_vocabulary()
        words = [w["word"] for w in weak]
        self.assertIn("間違い", words)

    def test_get_total_vocabulary_count(self):
        self.assertEqual(memory.get_total_vocabulary_count(), 0)
        memory.upsert_vocabulary("一", "いち")
        memory.upsert_vocabulary("二", "に")
        memory.upsert_vocabulary("一", "いち")  # duplicate word+reading
        self.assertEqual(memory.get_total_vocabulary_count(), 2)

    def test_get_vocabulary_known_count(self):
        for _ in range(4):
            memory.upsert_vocabulary("木", "き")
        for _ in range(2):
            memory.upsert_vocabulary("森", "もり")
        self.assertEqual(memory.get_vocabulary_known_count(min_exposures=3), 1)

    def test_get_recent_vocabulary_order(self):
        memory.upsert_vocabulary("Older", "おるだー")
        memory.upsert_vocabulary("Newer", "にゅーあー")
        recent = memory.get_recent_vocabulary(limit=2)
        self.assertEqual(recent[0]["word"], "Newer")
        self.assertEqual(recent[1]["word"], "Older")

    # ── Daily Stats & Streaks Tests ───────────────────────────────────────────

    def test_log_daily_activity_creates_row(self):
        memory.log_daily_activity(messages=2, words=5, minutes=3.5, xp=15)
        today = datetime.now().strftime("%Y-%m-%d")
        with memory._conn() as c:
            row = c.execute("SELECT * FROM daily_stats WHERE date=?", (today,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["messages_sent"], 2)
        self.assertEqual(row["words_learned"], 5)
        self.assertAlmostEqual(row["minutes_active"], 3.5)
        self.assertEqual(row["xp_earned"], 15)

    def test_log_daily_activity_upsert_adds(self):
        memory.log_daily_activity(messages=1, words=2, xp=5)
        memory.log_daily_activity(messages=3, words=4, xp=10)
        today = datetime.now().strftime("%Y-%m-%d")
        with memory._conn() as c:
            row = c.execute("SELECT * FROM daily_stats WHERE date=?", (today,)).fetchone()
        self.assertEqual(row["messages_sent"], 4)
        self.assertEqual(row["words_learned"], 6)
        self.assertEqual(row["xp_earned"], 15)

    def test_get_streak_no_activity(self):
        self.assertEqual(memory.get_streak(), 0)

    def test_get_streak_consecutive_days(self):
        today = datetime.now().date()
        with memory._conn() as c:
            for i in range(3):
                d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                c.execute(
                    "INSERT INTO daily_stats (date, messages_sent) VALUES (?, 1)", (d,)
                )
        self.assertEqual(memory.get_streak(), 3)

    def test_get_streak_gap_breaks(self):
        today = datetime.now().date()
        with memory._conn() as c:
            # Active today, active 2 days ago, but missing yesterday
            c.execute("INSERT INTO daily_stats (date, messages_sent) VALUES (?, 1)",
                      (today.strftime("%Y-%m-%d"),))
            c.execute("INSERT INTO daily_stats (date, messages_sent) VALUES (?, 1)",
                      ((today - timedelta(days=2)).strftime("%Y-%m-%d"),))
        self.assertEqual(memory.get_streak(), 1)

    def test_get_streak_today_plus_yesterday(self):
        today = datetime.now().date()
        with memory._conn() as c:
            c.execute("INSERT INTO daily_stats (date, messages_sent) VALUES (?, 1)",
                      (today.strftime("%Y-%m-%d"),))
            c.execute("INSERT INTO daily_stats (date, messages_sent) VALUES (?, 1)",
                      ((today - timedelta(days=1)).strftime("%Y-%m-%d"),))
        self.assertEqual(memory.get_streak(), 2)

    def test_get_streak_only_yesterday(self):
        # Grace period: student hasn't logged today yet, but logged yesterday
        yesterday = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        with memory._conn() as c:
            c.execute("INSERT INTO daily_stats (date, messages_sent) VALUES (?, 1)", (yesterday,))
        self.assertEqual(memory.get_streak(), 1)

    def test_get_today_xp_no_data(self):
        self.assertEqual(memory.get_today_xp(), 0)

    def test_get_today_xp_with_data(self):
        memory.log_daily_activity(xp=42)
        self.assertEqual(memory.get_today_xp(), 42)

    def test_get_daily_stats_history(self):
        today = datetime.now().date()
        with memory._conn() as c:
            c.execute("INSERT INTO daily_stats (date, messages_sent) VALUES (?, 5)",
                      ((today - timedelta(days=1)).strftime("%Y-%m-%d"),))
            c.execute("INSERT INTO daily_stats (date, messages_sent) VALUES (?, 10)",
                      (today.strftime("%Y-%m-%d"),))
        history = memory.get_daily_stats_history(days=7)
        self.assertEqual(len(history), 2)
        # Should be sorted chronologically (reversed from DESC)
        self.assertEqual(history[0]["messages_sent"], 5)
        self.assertEqual(history[1]["messages_sent"], 10)

    # ── Concurrency & Edge Cases ──────────────────────────────────────────────

    def test_concurrent_upsert_vocabulary(self):
        def worker():
            for _ in range(20):
                memory.upsert_vocabulary("競合", "きょうごう", "NOUN")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with memory._conn() as c:
            row = c.execute("SELECT times_seen FROM vocabulary_log WHERE word='競合'").fetchone()
        self.assertEqual(row["times_seen"], 80)

    def test_init_db_idempotent(self):
        memory.add_message("s1", "user", "テスト")
        # Second call to init_db should NOT wipe data or raise errors
        memory.init_db()
        msgs = memory.get_recent_messages(10)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "テスト")


if __name__ == "__main__":
    unittest.main()
