"""Lightweight unit tests for numeral classification and Qwen thought parsing."""
import unittest

from run_experiments import QwenRunner, canonicalize_number, classify_numeral_system, extract_answer


class ParserTests(unittest.TestCase):
    def test_numeral_styles(self):
        self.assertEqual(classify_numeral_system("१२५"), "Devanagari")
        self.assertEqual(classify_numeral_system("125"), "Hindu-Arabic")
        self.assertEqual(classify_numeral_system("12५"), "Mixed")

    def test_numeric_normalization(self):
        self.assertEqual(canonicalize_number("१२५"), "125")
        self.assertEqual(canonicalize_number("1,00,000"), "100000")
        self.assertEqual(canonicalize_number("3/4"), "0.75")
        self.assertEqual(canonicalize_number("25%"), "25%")

    def test_answer_extraction_preserves_original_script(self):
        self.assertEqual(extract_answer("अंतिम उत्तर: ४०"), ("४०", "Devanagari", "40"))
        self.assertEqual(extract_answer("Final answer: 40"), ("40", "Hindu-Arabic", "40"))
        self.assertEqual(extract_answer(r"\boxed{१२५}"), ("१२५", "Devanagari", "125"))

    def test_thinking_delimiters(self):
        runner = QwenRunner.__new__(QwenRunner)
        self.assertEqual(
            runner._split_thinking("reasoning</think>\nanswer"),
            ("reasoning", "answer"),
        )
        self.assertEqual(
            runner._split_thinking("reasoning</thinking>\nanswer"),
            ("reasoning", "answer"),
        )
        self.assertEqual(
            runner._split_thinking("<think>reasoning</think>\nanswer"),
            ("reasoning", "answer"),
        )


if __name__ == "__main__":
    unittest.main()
