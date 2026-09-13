# tests/test_knowledge.py — Thorough test suite for knowledge.py (Vocab & Grammar Extraction)

import unittest
from unittest.mock import patch, MagicMock

import knowledge
import memory
from tests.conftest import InMemoryDBTestCase, create_mock_doc


class TestKnowledge(InMemoryDBTestCase):
    """
    Comprehensive tests for knowledge.py covering Japanese character detection,
    GiNZA vocabulary extraction with morphological analysis, fallback regex extraction,
    grammar pattern detection, user/tutor message processing pipeline, and knowledge summary.
    """

    # ── Japanese Detection Tests ──────────────────────────────────────────────

    def test_contains_japanese_hiragana(self):
        self.assertTrue(knowledge._contains_japanese("こんにちは"))

    def test_contains_japanese_katakana(self):
        self.assertTrue(knowledge._contains_japanese("カタカナ"))

    def test_contains_japanese_kanji(self):
        self.assertTrue(knowledge._contains_japanese("漢字"))

    def test_contains_japanese_english_only(self):
        self.assertFalse(knowledge._contains_japanese("Hello World! 123"))

    def test_contains_japanese_mixed(self):
        self.assertTrue(knowledge._contains_japanese("Hello こんにちは"))

    # ── GiNZA Extraction Tests (Mocked NLP) ───────────────────────────────────

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_filters_pos(self, mock_process):
        # NOUN (>=2 chars) kept, ADP filtered, VERB (>=2 chars) kept
        mock_process.return_value = create_mock_doc([
            {"text": "料理", "pos": "NOUN", "lemma": "料理", "reading": "リョウリ"},
            {"text": "が", "pos": "ADP", "lemma": "が"},
            {"text": "走る", "pos": "VERB", "lemma": "走る", "reading": "ハシル"},
        ])

        words = knowledge.extract_words_ginza("料理が走る")
        self.assertEqual(len(words), 2)
        pos_list = [w["pos"] for w in words]
        self.assertIn("NOUN", pos_list)
        self.assertIn("VERB", pos_list)

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_uses_lemma(self, mock_process):
        mock_process.return_value = create_mock_doc([
            {"text": "食べました", "pos": "VERB", "lemma": "食べる", "reading": "タベル"},
        ])
        words = knowledge.extract_words_ginza("食べました")
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["word"], "食べる")

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_skips_stop_words(self, mock_process):
        mock_process.return_value = create_mock_doc([
            {"text": "私", "pos": "PRON", "lemma": "私"},  # In stop words
            {"text": "リンゴ", "pos": "NOUN", "lemma": "リンゴ", "reading": "リンゴ"},
        ])
        words = knowledge.extract_words_ginza("私リンゴ")
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["word"], "リンゴ")

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_skips_short_words(self, mock_process):
        # Words < 2 characters are skipped
        mock_process.return_value = create_mock_doc([
            {"text": "手", "pos": "NOUN", "lemma": "手"},
            {"text": "寿司", "pos": "NOUN", "lemma": "寿司", "reading": "スシ"},
        ])
        words = knowledge.extract_words_ginza("手寿司")
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["word"], "寿司")

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_skips_non_japanese(self, mock_process):
        mock_process.return_value = create_mock_doc([
            {"text": "Python", "pos": "PROPN", "lemma": "Python"},
            {"text": "プログラミング", "pos": "NOUN", "lemma": "プログラミング"},
        ])
        words = knowledge.extract_words_ginza("Pythonプログラミング")
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["word"], "プログラミング")

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_deduplicates(self, mock_process):
        mock_process.return_value = create_mock_doc([
            {"text": "料理", "pos": "NOUN", "lemma": "料理"},
            {"text": "料理", "pos": "NOUN", "lemma": "料理"},
        ])
        words = knowledge.extract_words_ginza("料理料理")
        self.assertEqual(len(words), 1)

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_fallback_on_error(self, mock_process):
        mock_doc = MagicMock()
        mock_doc.__iter__.side_effect = Exception("GiNZA boom")
        mock_process.return_value = mock_doc

        words = knowledge.extract_words_ginza("ラーメンを食べます")
        # Should fallback to regex extraction without raising
        self.assertTrue(len(words) > 0)
        self.assertTrue(any("ラーメン" in w["word"] or "食べ" in w["word"] for w in words))

    @patch("nlp_core.process_text_safely")
    def test_extract_words_ginza_reading_extraction(self, mock_process):
        mock_process.return_value = create_mock_doc([
            {"text": "料理", "pos": "NOUN", "lemma": "料理", "reading": "リョウリ"},
        ])
        words = knowledge.extract_words_ginza("料理")
        self.assertEqual(words[0]["reading"], "リョウリ")

    # ── Fallback Regex Extraction Tests ───────────────────────────────────────

    def test_extract_words_fallback_kanji(self):
        words = knowledge.extract_words_fallback("食べ物 料理")
        word_strings = [w["word"] for w in words]
        self.assertIn("食べ物", word_strings)

    def test_extract_words_fallback_katakana(self):
        words = knowledge.extract_words_fallback("朝コーヒーを飲みました")
        word_strings = [w["word"] for w in words]
        self.assertIn("コーヒー", word_strings)

    def test_extract_words_fallback_hiragana(self):
        # Hiragana >= 3 chars
        words = knowledge.extract_words_fallback("おはようございます")
        self.assertTrue(len(words) > 0)

    def test_extract_words_fallback_deduplicates(self):
        words = knowledge.extract_words_fallback("東京 東京 東京")
        self.assertEqual(len([w for w in words if w["word"] == "東京"]), 1)

    # ── Grammar Pattern Detection Tests ───────────────────────────────────────

    def test_grammar_pattern_te_form(self):
        patterns = knowledge.extract_grammar_patterns("日本語を教えてください")
        self.assertIn("て-form request", patterns)

    def test_grammar_pattern_progressive(self):
        patterns = knowledge.extract_grammar_patterns("今勉強しています")
        self.assertIn("progressive/state", patterns)

    def test_grammar_pattern_multiple(self):
        patterns = knowledge.extract_grammar_patterns("日本に行きたいです。富士山を見たことがある。")
        self.assertIn("want to (tai-form)", patterns)
        self.assertIn("experience (koto ga aru)", patterns)

    def test_grammar_pattern_none(self):
        patterns = knowledge.extract_grammar_patterns("こんにちは。")
        self.assertEqual(patterns, [])

    # ── Processing Pipeline Tests ─────────────────────────────────────────────

    @patch("knowledge.extract_words_ginza")
    def test_process_user_message_tracks_produced(self, mock_extract):
        mock_extract.return_value = [
            {"word": "寿司", "reading": "すし", "pos": "NOUN"}
        ]
        knowledge.process_user_message("寿司を食べました")

        with memory._conn() as c:
            row = c.execute("SELECT * FROM vocabulary_log WHERE word='寿司'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["times_produced"], 1)
        self.assertEqual(row["times_seen"], 1)

    @patch("knowledge.extract_words_ginza")
    def test_process_user_message_logs_activity_xp(self, mock_extract):
        mock_extract.return_value = [
            {"word": "花", "reading": "はな", "pos": "NOUN"}
        ]
        knowledge.process_user_message("花が咲く")
        self.assertGreater(memory.get_today_xp(), 0)

    def test_process_user_message_english_skipped(self):
        knowledge.process_user_message("Hello there, how are you?")
        self.assertEqual(memory.get_total_vocabulary_count(), 0)
        self.assertEqual(memory.get_today_xp(), 0)

    @patch("knowledge.extract_words_ginza")
    def test_process_tutor_message_tracks_seen_not_produced(self, mock_extract):
        mock_extract.return_value = [
            {"word": "先生", "reading": "せんせい", "pos": "NOUN"}
        ]
        knowledge.process_tutor_message("先生が話しました --- 🇬🇧 Teacher spoke")

        with memory._conn() as c:
            row = c.execute("SELECT * FROM vocabulary_log WHERE word='先生'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["times_seen"], 1)
        self.assertEqual(row["times_produced"], 0)

    @patch("knowledge.extract_words_ginza")
    def test_process_tutor_message_only_japanese_part(self, mock_extract):
        mock_extract.return_value = []
        knowledge.process_tutor_message("日本語の文 --- 🇬🇧 English explanation with Japanese letters 猫")
        # Should be called with the text before '---'
        mock_extract.assert_called_once()
        called_arg = mock_extract.call_args[0][0]
        self.assertIn("日本語の文", called_arg)
        self.assertNotIn("English explanation", called_arg)

    # ── Knowledge Summary Tests ───────────────────────────────────────────────

    def test_build_knowledge_summary_format(self):
        summary = knowledge.build_knowledge_summary()
        self.assertIn("=== Student Knowledge ===", summary)
        self.assertIn("Total vocabulary encountered:", summary)
        self.assertIn("Words considered known", summary)
        self.assertIn("Weak/needs review:", summary)
        self.assertIn("=== Recurring Mistakes ===", summary)

    def test_build_knowledge_summary_empty_db(self):
        summary = knowledge.build_knowledge_summary()
        self.assertIn("None yet — student is a complete beginner", summary)
        self.assertIn("None yet", summary)

    def test_build_knowledge_summary_with_data(self):
        from config import VOCAB_KNOWN_THRESHOLD
        for _ in range(VOCAB_KNOWN_THRESHOLD + 1):
            memory.upsert_vocabulary("電車", "でんしゃ", "NOUN", produced=True)

        # Weak word: seen 2 times, never produced
        memory.upsert_vocabulary("飛行機", "ひこうき", "NOUN", produced=False)
        memory.upsert_vocabulary("飛行機", "ひこうき", "NOUN", produced=False)

        summary = knowledge.build_knowledge_summary()
        self.assertIn("電車", summary)
        self.assertIn("飛行機", summary)
        self.assertIn("never produced", summary)


if __name__ == "__main__":
    unittest.main()
