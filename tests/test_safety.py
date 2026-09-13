import unittest
import safety
import nlp_core

class TestSafety(unittest.TestCase):
    def setUp(self):
        # Ensure NLP is initialized for safety tests
        nlp_core._init_ginza()

    def test_extract_japanese_segments(self):
        text = "こんにちは --- 🇬🇧 Hello"
        segments = safety.extract_japanese_segments(text)
        self.assertTrue(len(segments) > 0)
        self.assertIn("こんにちは", segments[0])

    def test_ginza_check_safe_sentence(self):
        # A valid sentence with a verb
        text = "私は学生です" # Or something simple
        warnings = safety._ginza_check(text)
        # Should be empty or no critical structural errors
        self.assertEqual(len(warnings), 0)

    def test_check_safety_pipeline(self):
        result = safety.check("私はりんごを食べます。 --- 🇬🇧 I eat apples.")
        self.assertFalse(result["flagged"])
        self.assertEqual(len(result["warnings"]), 0)

if __name__ == "__main__":
    unittest.main()
