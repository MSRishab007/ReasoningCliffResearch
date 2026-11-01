"""Unit tests for answer parsing, numeral normalization and Qwen prompt setup."""
import unittest
from types import SimpleNamespace

import torch

from run_experiments import (
    QwenRunner,
    build_prompt,
    canonicalize_number,
    classify_numeral_system,
    extract_answer,
    classify_thinking_stop,
    resolve_answer_trigger_action,
)


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
        self.assertEqual(runner._split_thinking("reasoning</think>\nanswer"), ("reasoning", "answer"))
        self.assertEqual(runner._split_thinking("reasoning</thinking>\nanswer"), ("reasoning", "answer"))
        self.assertEqual(runner._split_thinking("<think>reasoning</think>\nanswer"), ("reasoning", "answer"))

    def test_question_only_prompt_is_exact(self):
        question = "राम के पास १२ आम हैं।"
        self.assertEqual(build_prompt("question_only", question), question)

    def test_answer_trigger_uses_same_question_only_prompt(self):
        question = "सीमा के पास २५ रुपये हैं।"
        self.assertEqual(build_prompt("question_only", question), build_prompt("question_only", question))

    def test_generate_output_tensor_is_supported(self):
        tensor = torch.tensor([[1, 2, 3]])
        self.assertIs(QwenRunner._extract_sequences(tensor), tensor)

    def test_generate_output_model_output_is_supported(self):
        tensor = torch.tensor([[4, 5, 6]])
        output = SimpleNamespace(sequences=tensor)
        self.assertIs(QwenRunner._extract_sequences(output), tensor)

    def test_natural_think_end_is_detected(self):
        self.assertEqual(
            classify_thinking_stop([11, 12, 99], 1024, 99, 2),
            "natural_think_end",
        )

    def test_thinking_limit_without_close_is_truncated(self):
        self.assertEqual(
            classify_thinking_stop([11, 12, 13], 3, 99, 2),
            "max_new_tokens_without_think_end",
        )

    def test_eos_before_think_end_is_not_misclassified_as_natural(self):
        self.assertEqual(
            classify_thinking_stop([11, 2], 1024, 99, 2),
            "eos_before_think_end",
        )

    def test_natural_trigger_continues_without_injection(self):
        self.assertEqual(resolve_answer_trigger_action("natural_think_end", False), (True, False))
        self.assertEqual(resolve_answer_trigger_action("natural_think_end", True), (True, False))

    def test_natural_mode_skips_answer_after_truncation(self):
        self.assertEqual(
            resolve_answer_trigger_action("max_new_tokens_without_think_end", False),
            (False, False),
        )

    def test_forced_control_injects_only_after_token_limit(self):
        self.assertEqual(
            resolve_answer_trigger_action("max_new_tokens_without_think_end", True),
            (True, True),
        )
        self.assertEqual(
            resolve_answer_trigger_action("eos_before_think_end", True),
            (False, False),
        )


if __name__ == "__main__":
    unittest.main()
