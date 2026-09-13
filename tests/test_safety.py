# tests/test_safety.py — Thorough test suite for safety.py (NLP Verification)

import unittest
from unittest.mock import patch, MagicMock
import safety
import nlp_core
from tests.conftest import create_mock_doc, MockToken


class TestSafety(unittest.TestCase):
    """
    Comprehensive tests for safety.py covering Japanese segment extraction,
    GiNZA structural analysis, uncertainty detection, suspicious pattern detection,
    and the full safety check pipeline.
    """

    def setUp(self):
        # Ensure NLP is initialized or mocked
        nlp_core._init_ginza()

    # ── Original Tests (Preserved) ────────────────────────────────────────────

    def test_extract_japanese_segments(self):
        text = "こんにちは --- 🇬🇧 Hello"
        segments = safety.extract_japanese_segments(text)
        self.assertTrue(len(segments) > 0)
        self.assertIn("こんにちは", segments[0])

    def test_ginza_check_safe_sentence(self):
        text = "私は学生です"
        warnings = safety._ginza_check(text)
        self.assertEqual(len(warnings), 0)

    def test_check_safety_pipeline(self):
        result = safety.check("私はりんごを食べます。 --- 🇬🇧 I eat apples.")
        self.assertFalse(result["flagged"])
        self.assertEqual(len(result["warnings"]), 0)

    # ── Japanese Segment Extraction Tests ─────────────────────────────────────

    def test_extract_segments_mixed_text(self):
        text = "Hello こんにちは world 日本語です"
        segments = safety.extract_japanese_segments(text)
        self.assertIn("こんにちは", segments)
        self.assertIn("日本語です", segments)

    def test_extract_segments_respects_separator(self):
        text = "美味しいです --- 🇬🇧 Delicious (おいしい)"
        segments = safety.extract_japanese_segments(text)
        self.assertEqual(segments, ["美味しいです"])

    def test_extract_segments_filters_short(self):
        # Characters with length <= 2 are excluded (len(m.group()) > 2)
        text = "はい 猫 です"  # "はい" is 2 chars, "猫" is 1 char, "です" is 2 chars
        segments = safety.extract_japanese_segments(text)
        self.assertEqual(segments, [])

    def test_extract_segments_pure_english(self):
        text = "Hello this is a complete English message without Japanese."
        segments = safety.extract_japanese_segments(text)
        self.assertEqual(segments, [])

    def test_extract_segments_multiple_runs(self):
        text = "Start おはようございます middle ありがとうございます end"
        segments = safety.extract_japanese_segments(text)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], "おはようございます")
        self.assertEqual(segments[1], "ありがとうございます")

    def test_extract_segments_katakana(self):
        text = "これはコンピューターです"
        segments = safety.extract_japanese_segments(text)
        self.assertIn("これはコンピューターです", segments)

    # ── GiNZA Structural Check Tests (Mocked NLP) ────────────────────────────

    @patch("nlp_core.process_text_safely")
    def test_ginza_check_valid_sentence(self, mock_process):
        # Sentence with NOUN and VERB
        mock_doc = create_mock_doc([
            {"text": "私", "pos": "PRON"},
            {"text": "は", "pos": "ADP"},
            {"text": "本", "pos": "NOUN"},
            {"text": "を", "pos": "ADP"},
            {"text": "読む", "pos": "VERB"},
        ])
        mock_process.return_value = mock_doc
        warnings = safety._ginza_check("私は本を読む")
        self.assertEqual(warnings, [])

    @patch("nlp_core.process_text_safely")
    def test_ginza_check_missing_verb(self, mock_process):
        # Sentence with >4 tokens, multiple nouns, but no VERB or AUX
        mock_doc = create_mock_doc([
            {"text": "東京", "pos": "PROPN"},
            {"text": "タワー", "pos": "NOUN"},
            {"text": "観光", "pos": "NOUN"},
            {"text": "名所", "pos": "NOUN"},
            {"text": "案内", "pos": "NOUN"},
        ])
        mock_process.return_value = mock_doc
        warnings = safety._ginza_check("東京タワー観光名所案内")
        self.assertTrue(any("Possible missing verb" in w for w in warnings))

    @patch("nlp_core.process_text_safely")
    def test_ginza_check_noun_chain(self, mock_process):
        # Sentence with 5+ consecutive NOUNs
        mock_doc = create_mock_doc([
            {"text": "昨日", "pos": "NOUN"},
            {"text": "日本", "pos": "PROPN"},
            {"text": "東京", "pos": "PROPN"},
            {"text": "大学", "pos": "NOUN"},
            {"text": "図書館", "pos": "NOUN"},
            {"text": "学生", "pos": "NOUN"},
            {"text": "行きました", "pos": "VERB"},
        ])
        mock_process.return_value = mock_doc
        warnings = safety._ginza_check("昨日日本東京大学図書館学生行きました")
        self.assertTrue(any("Suspicious noun chain" in w for w in warnings))

    @patch("nlp_core.process_text_safely")
    def test_ginza_check_short_sentence_skipped(self, mock_process):
        # Sentence with < 2 tokens is skipped
        mock_doc = create_mock_doc([{"text": "あ", "pos": "NOUN"}])
        mock_process.return_value = mock_doc
        warnings = safety._ginza_check("あ")
        self.assertEqual(warnings, [])

    @patch("nlp_core.process_text_safely")
    def test_ginza_check_nlp_unavailable(self, mock_process):
        mock_process.return_value = None
        warnings = safety._ginza_check("何かのテキスト")
        self.assertEqual(warnings, [])

    @patch("nlp_core.process_text_safely")
    def test_ginza_check_handles_exception(self, mock_process):
        mock_doc = MagicMock()
        mock_doc.sents = MagicMock(side_effect=RuntimeError("Corrupt spaCy parse"))
        mock_process.return_value = mock_doc
        warnings = safety._ginza_check("エラー発生テキスト")
        self.assertEqual(warnings, [])

    # ── Confidence Heuristics Tests ───────────────────────────────────────────

    def test_uncertainty_markers_detected(self):
        self.assertTrue(safety._has_uncertainty_markers("それは正しいかもしれません"))
        self.assertTrue(safety._has_uncertainty_markers("ちょっとわかりませんが"))
        self.assertTrue(safety._has_uncertainty_markers("そうだと思います"))

    def test_uncertainty_markers_absent(self):
        self.assertFalse(safety._has_uncertainty_markers("これは本です"))
        self.assertFalse(safety._has_uncertainty_markers("一緒に勉強しましょう"))

    def test_suspicious_patterns_particle_cluster(self):
        # e.g., "aはがをに" where a non-Japanese char is followed by 3+ particles
        self.assertTrue(safety._has_suspicious_patterns("wordはがをにtest"))

    def test_suspicious_patterns_long_kanji(self):
        # 8+ consecutive kanji
        self.assertTrue(safety._has_suspicious_patterns("日本国際電子技術研究所総長"))

    def test_suspicious_patterns_normal(self):
        self.assertFalse(safety._has_suspicious_patterns("私は毎日日本語を勉強しています"))

    # ── Full Safety Check Pipeline Tests ──────────────────────────────────────

    @patch("safety._ginza_check", return_value=[])
    def test_check_clean_response(self, mock_ginza):
        res = safety.check("今日はとても良い天気ですね。 --- 🇬🇧 It's nice weather today.")
        self.assertFalse(res["flagged"])
        self.assertEqual(res["warnings"], [])
        self.assertEqual(res["confidence"], "high")

    @patch("safety._ginza_check", return_value=[])
    def test_check_flagged_single_warning(self, mock_ginza):
        # Uncertainty marker triggers 1 warning -> medium confidence
        res = safety.check("これは正しいかもしれません --- 🇬🇧 Maybe correct")
        self.assertTrue(res["flagged"])
        self.assertEqual(len(res["warnings"]), 1)
        self.assertEqual(res["confidence"], "medium")

    @patch("safety._ginza_check", return_value=["Structural warning"])
    def test_check_multiple_warnings(self, mock_ginza):
        # 1 GiNZA warning + 1 uncertainty warning -> 2 warnings -> low confidence
        res = safety.check("これは正しいかもしれません --- 🇬🇧 Maybe")
        self.assertTrue(res["flagged"])
        self.assertGreaterEqual(len(res["warnings"]), 2)
        self.assertEqual(res["confidence"], "low")

    def test_check_no_japanese(self):
        res = safety.check("Just pure English response.")
        self.assertFalse(res["flagged"])
        self.assertEqual(res["confidence"], "high")
        self.assertEqual(res["warnings"], [])

    @patch("nlp_core.get_nlp", return_value=(MagicMock(), True))
    def test_ginza_status_available(self, mock_nlp):
        self.assertEqual(safety.ginza_status(), "GiNZA ✓")

    @patch("nlp_core.get_nlp", return_value=(None, False))
    def test_ginza_status_unavailable(self, mock_nlp):
        status = safety.ginza_status()
        self.assertIn("GiNZA ✗", status)


if __name__ == "__main__":
    unittest.main()
