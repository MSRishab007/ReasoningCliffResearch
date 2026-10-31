"""Run inference-time reasoning experiments with Qwen3-4B in BF16 on GPU.

Experiments include question-only prompting, direct prompting, native Qwen
thinking, explicit CoT, structured CoT, English-pivot CoT, Plan-and-Solve,
Self-Consistency, Multi-Strategy Parallel Reasoning (MSPR), MSPR + verifier,
and a staged Qwen answer-trigger experiment.

GPU is mandatory: this runner deliberately refuses CPU execution or automatic
CPU offload. The full Hindi question, model thinking, final response, raw
output, numeral-script label, answer, token counts and end-to-end timing are
saved to CSV.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import torch

MODEL_ID_DEFAULT = "Qwen/Qwen3-4B"
DEFAULT_INPUT = "data/gsm8k_hi_200.csv"
DEFAULT_RESULTS = "results/qwen3_4b_bf16_gsm8k_hi_results.csv"
DEFAULT_SUMMARY = "results/qwen3_4b_bf16_summary_by_technique.csv"
DEFAULT_DTYPE_NAME = "bfloat16"

DEVANAGARI_DIGITS = "०१२३४५६७८९"
ASCII_DIGITS = "0123456789"
DIGIT_TRANSLATION = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)
NUMERIC_PATTERN = re.compile(
    r"(?<![\w])[-+]?(?:[0-9०-९]{1,3}(?:,[0-9०-९]{2,3})+|[0-9०-९]+)(?:\.[0-9०-९]+)?(?:/[0-9०-९]+)?%?"
)

CSV_FIELDS = [
    "question_id", "source_row_index", "dataset_id", "dataset_config", "dataset_split",
    "question_hi", "question_original", "reference_solution_hi",
    "reference_solution_original", "gold_answer_raw", "gold_answer_source",
    "gold_answer_normalized", "gold_numeral_system", "model", "model_dtype",
    "device", "gpu_name", "gpu_total_memory_gib", "technique_id", "technique_name",
    "thinking_enabled", "temperature", "top_p", "top_k", "max_new_tokens",
    "answer_max_new_tokens", "num_candidates", "prompt_text", "raw_model_output",
    "thinking_text", "final_response", "candidate_details_json", "verifier_raw_output",
    "verifier_thinking_text", "answer_trigger_text", "answer_trigger_injected",
    "thinking_stop_reason", "thinking_generation_time_seconds",
    "answer_generation_time_seconds", "extracted_answer_raw", "predicted_numeral_system",
    "predicted_answer_normalized", "is_correct", "answer_extraction_status",
    "input_tokens", "output_tokens", "thinking_tokens", "final_tokens",
    "latency_seconds", "model_generation_time_seconds",
    "candidate_generation_time_seconds", "verifier_generation_time_seconds",
    "technique_inference_time_seconds", "run_status", "error_message",
]


@dataclass
class Generation:
    raw: str
    thinking: str
    final: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    final_tokens: int
    latency_seconds: float
    temperature: float
    top_p: float
    top_k: int
    thinking_generation_time_seconds: float = 0.0
    answer_generation_time_seconds: float = 0.0
    answer_trigger_text: str = ""
    answer_trigger_injected: bool = False
    thinking_stop_reason: str = "not_applicable"


class QwenRunner:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        answer_max_new_tokens: int = 512,
        gpu_id: int = 0,
        min_free_vram_gib: float = 9.0,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.answer_max_new_tokens = answer_max_new_tokens
        self.gpu_id = gpu_id

        # Mandatory GPU + native BF16 checks. Never silently fall back to CPU.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and check the NVIDIA driver."
            )
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {gpu_id} was requested, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(gpu_id)
        if not torch.cuda.is_bf16_supported():
            name = torch.cuda.get_device_name(gpu_id)
            capability = torch.cuda.get_device_capability(gpu_id)
            raise RuntimeError(
                f"GPU '{name}' (compute capability {capability}) does not report native BF16 support. "
                "This run is configured for BF16 only; CPU/FP16 fallback is disabled."
            )

        props = torch.cuda.get_device_properties(gpu_id)
        total_gib = props.total_memory / (1024 ** 3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
        free_gib = free_bytes / (1024 ** 3)
        print(f"GPU selected: cuda:{gpu_id} | {props.name}")
        print(f"GPU memory: {free_gib:.2f} GiB free / {total_gib:.2f} GiB total")
        if total_gib < 14.0:
            raise RuntimeError(
                f"The selected GPU has {total_gib:.2f} GiB total memory. "
                "This experiment expects a 16-GB-class GPU for Qwen3-4B BF16."
            )
        if free_gib < min_free_vram_gib:
            raise RuntimeError(
                f"Only {free_gib:.2f} GiB VRAM is free; at least {min_free_vram_gib:.2f} GiB is required. "
                "Close GPU-heavy applications or lower the configured minimum deliberately."
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        print(f"Loading model in BF16 directly onto cuda:{gpu_id}: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_id},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        first_parameter = next(self.model.parameters())
        if first_parameter.device.type != "cuda":
            raise RuntimeError(
                f"GPU-only assertion failed: model parameter landed on {first_parameter.device}."
            )
        if first_parameter.dtype != torch.bfloat16:
            raise RuntimeError(
                f"BF16 assertion failed: model parameter dtype is {first_parameter.dtype}, not torch.bfloat16."
            )
        self.device = torch.device(f"cuda:{gpu_id}")
        self.gpu_name = props.name
        self.gpu_total_memory_gib = round(total_gib, 3)
        print(f"Verified: model device={first_parameter.device}, dtype={first_parameter.dtype}")
        print(f"Allocated VRAM after load: {torch.cuda.memory_allocated(gpu_id) / (1024 ** 3):.2f} GiB")

        think_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(think_ids) != 1:
            raise RuntimeError(
                f"Expected Qwen </think> to be one tokenizer ID, got {think_ids}. "
                "Check that the selected tokenizer is the official Qwen3 tokenizer."
            )
        self.think_end_token_id = think_ids[0]

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _strip_terminal_chat_tokens(text: str) -> str:
        return re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>|<\|fim_suffix\|>)+\s*$", "", text).strip()

    def _split_thinking(self, raw: str) -> tuple[str, str]:
        closing = re.search(r"</(?:think|thinking)>", raw, flags=re.IGNORECASE)
        if closing:
            thinking = raw[:closing.start()]
            final = raw[closing.end():]
            thinking = re.sub(r"^\s*<(?:think|thinking)>\s*", "", thinking, flags=re.I)
            return self._strip_terminal_chat_tokens(thinking), self._strip_terminal_chat_tokens(final)
        opening = re.search(r"<(?:think|thinking)>", raw, flags=re.IGNORECASE)
        if opening:
            return self._strip_terminal_chat_tokens(raw[opening.end():]), ""
        return "", self._strip_terminal_chat_tokens(raw)

    def _format_inputs(self, prompt: str, enable_thinking: bool) -> dict[str, torch.Tensor]:
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        encoded = self.tokenizer(formatted, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _sample_kwargs(self, enable_thinking: bool, seed: int) -> dict[str, Any]:
        self._set_seed(seed)
        return {
            "do_sample": True,
            "temperature": 0.6 if enable_thinking else 0.7,
            "top_p": 0.95 if enable_thinking else 0.8,
            "top_k": 20,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }

    def _timed_generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        sample_kwargs: dict[str, Any],
        eos_token_id: Any = None,
    ) -> tuple[torch.Tensor, float]:
        torch.cuda.synchronize(self.gpu_id)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id,
                **sample_kwargs,
            )
        torch.cuda.synchronize(self.gpu_id)
        return output.sequences, time.perf_counter() - start

    def generate(
        self,
        prompt: str,
        enable_thinking: bool,
        seed: int,
        max_new_tokens: Optional[int] = None,
    ) -> Generation:
        kwargs = self._sample_kwargs(enable_thinking, seed)
        inputs = self._format_inputs(prompt, enable_thinking)
        input_tokens = int(inputs["input_ids"].shape[-1])
        sequences, elapsed = self._timed_generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            sample_kwargs=kwargs,
        )
        generated_ids = sequences[0, input_tokens:]
        raw = self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        thinking, final = self._split_thinking(raw)
        output_tokens = int(generated_ids.numel())
        thinking_tokens = len(self.tokenizer.encode(thinking, add_special_tokens=False)) if thinking else 0
        final_tokens = len(self.tokenizer.encode(final, add_special_tokens=False)) if final else 0
        return Generation(
            raw=raw, thinking=thinking, final=final,
            input_tokens=input_tokens, output_tokens=output_tokens,
            thinking_tokens=thinking_tokens, final_tokens=final_tokens,
            latency_seconds=elapsed, temperature=kwargs["temperature"],
            top_p=kwargs["top_p"], top_k=kwargs["top_k"],
        )

    def generate_with_answer_trigger(self, prompt: str, seed: int) -> Generation:
        """Generate thinking to </think>, then continue after the Qwen answer delimiter.

        Qwen3 does not define a separate <answer> special token. Its documented
        format closes reasoning with </think>, followed by a blank-line boundary
        and then the assistant's normal answer. We stop at </think> and resume
        generation after the `\n\n` answer boundary. If the thought budget is
        exhausted before the model emits </think>, we inject the closing marker
        and record that intervention in the CSV.
        """
        kwargs = self._sample_kwargs(True, seed)
        inputs = self._format_inputs(prompt, enable_thinking=True)
        base_ids = inputs["input_ids"]
        base_mask = inputs.get("attention_mask", torch.ones_like(base_ids))
        input_tokens = int(base_ids.shape[-1])

        # Stop the reasoning phase as soon as Qwen emits its </think> token.
        phase1_sequences, phase1_time = self._timed_generate(
            input_ids=base_ids,
            attention_mask=base_mask,
            max_new_tokens=self.max_new_tokens,
            sample_kwargs=kwargs,
            eos_token_id=[self.think_end_token_id, self.tokenizer.eos_token_id],
        )
        phase1_ids = phase1_sequences[0, input_tokens:]
        phase1_raw = self.tokenizer.decode(
            phase1_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        closed_naturally = bool(phase1_ids.numel()) and int(phase1_ids[-1]) == self.think_end_token_id
        stop_reason = "think_end_token" if closed_naturally else "max_new_tokens_forced_close"

        # Explicit Qwen answer transition: closing tag + blank-line separator.
        trigger_text = "\n\n"
        if closed_naturally:
            continuation_prefix = phase1_ids
        else:
            close_tensor = torch.tensor([[self.think_end_token_id]], dtype=base_ids.dtype, device=self.device)
            continuation_prefix = torch.cat([phase1_ids.unsqueeze(0), close_tensor], dim=1)
        trigger_ids = self.tokenizer.encode(trigger_text, add_special_tokens=False)
        trigger_tensor = torch.tensor([trigger_ids], dtype=base_ids.dtype, device=self.device)
        phase2_input = torch.cat([base_ids, continuation_prefix, trigger_tensor], dim=1)
        phase2_mask = torch.ones_like(phase2_input)

        # Reset seed to keep the answer-stage sampling reproducible but distinct.
        kwargs_phase2 = self._sample_kwargs(True, seed + 1)
        phase2_sequences, phase2_time = self._timed_generate(
            input_ids=phase2_input,
            attention_mask=phase2_mask,
            max_new_tokens=self.answer_max_new_tokens,
            sample_kwargs=kwargs_phase2,
        )
        phase2_ids = phase2_sequences[0, phase2_input.shape[-1]:]
        phase2_raw = self.tokenizer.decode(
            phase2_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        _, final = self._split_thinking(phase2_raw)
        thinking, _ = self._split_thinking(phase1_raw)
        if not thinking:
            # Preserve incomplete thinking even if the model omitted its opening tag.
            thinking = re.sub(r"</think>\s*$", "", phase1_raw).strip()

        raw = phase1_raw + "\n\n" + phase2_raw
        total_output_tokens = int(phase1_ids.numel() + phase2_ids.numel())
        thinking_tokens = len(self.tokenizer.encode(thinking, add_special_tokens=False)) if thinking else 0
        final_tokens = len(self.tokenizer.encode(final, add_special_tokens=False)) if final else 0
        return Generation(
            raw=raw, thinking=thinking, final=final,
            input_tokens=input_tokens + int(phase2_input.shape[-1]),
            output_tokens=total_output_tokens,
            thinking_tokens=thinking_tokens, final_tokens=final_tokens,
            latency_seconds=phase1_time + phase2_time,
            temperature=kwargs["temperature"], top_p=kwargs["top_p"], top_k=kwargs["top_k"],
            thinking_generation_time_seconds=phase1_time,
            answer_generation_time_seconds=phase2_time,
            answer_trigger_text="</think>\n\n",
            answer_trigger_injected=not closed_naturally,
            thinking_stop_reason=stop_reason,
        )

def normalize_digits(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).translate(DIGIT_TRANSLATION)


def classify_numeral_system(text: str) -> str:
    text = str(text or "")
    has_devanagari = bool(re.search(r"[०-९]", text))
    has_ascii = bool(re.search(r"[0-9]", text))
    if has_devanagari and has_ascii:
        return "Mixed"
    if has_devanagari:
        return "Devanagari"
    if has_ascii:
        return "Hindu-Arabic"
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi-word-based"
    if text.strip():
        return "Other-text"
    return "Unparsed"


def canonicalize_number(raw_number: str) -> str:
    """Canonicalize integers, decimals, fractions and percentages safely."""
    value = normalize_digits(raw_number).strip().replace(",", "")
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1]
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            number = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
            decimal_value = Decimal(number.numerator) / Decimal(number.denominator)
        else:
            decimal_value = Decimal(value)
        if not decimal_value.is_finite():
            return ""
        normalized = format(decimal_value.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        if normalized in {"-0", "+0", ""}:
            normalized = "0"
        return normalized + ("%" if is_percent else "")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return ""


def _numeric_match(text: str) -> Optional[re.Match[str]]:
    # Match against the original text so numeral-style classification is preserved.
    matches = list(NUMERIC_PATTERN.finditer(text))
    return matches[-1] if matches else None


def extract_answer(final_response: str) -> tuple[str, str, str]:
    """Return (raw answer span, numeral style, canonical numeric answer)."""
    text = str(final_response or "").strip()
    if not text:
        return "", "Unparsed", ""

    # Prefer explicit boxed answers.
    boxed = list(re.finditer(r"\\boxed\s*\{([^{}]+)\}", text, flags=re.I))
    if boxed:
        raw = boxed[-1].group(1).strip()
        style = classify_numeral_system(raw)
        return raw, style, canonicalize_number(raw)

    # Prefer the suffix associated with an explicit final-answer marker.
    markers = [
        r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*",
        r"(?:अंतिम\s*उत्तर|उत्तर)\s*(?:है|:|=)?\s*",
        r"####\s*",
    ]
    suffix = text
    for marker in markers:
        found = list(re.finditer(marker, text, flags=re.I))
        if found:
            suffix = text[found[-1].end():]
            break

    match = _numeric_match(suffix)
    if not match:
        # Preserve a likely word-based answer for numeral-format reporting.
        word_answer = suffix.strip().splitlines()[0][:160]
        if re.search(r"[\u0900-\u097F]", word_answer):
            return word_answer, "Hindi-word-based", ""
        return "", "Unparsed", ""

    raw = match.group(0).strip()
    style = classify_numeral_system(raw)
    return raw, style, canonicalize_number(raw)


def get_gold_answer(row: dict[str, Any]) -> tuple[str, str, str]:
    raw = str(row.get("gold_answer_raw", "") or "").strip()
    style = classify_numeral_system(raw)
    normalized = canonicalize_number(raw)
    return raw, style, normalized


def majority_select(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    valid = [c for c in candidates if c.get("answer_normalized")]
    if not valid:
        return candidates[0], {}
    counts = Counter(c["answer_normalized"] for c in valid)
    max_count = max(counts.values())
    winning_values = {value for value, count in counts.items() if count == max_count}
    # Deterministic tie-break: earliest candidate whose answer is tied for the lead.
    selected = next(c for c in valid if c["answer_normalized"] in winning_values)
    return selected, dict(counts)


TECHNIQUES = {
    "question_only": "E1_Question_Only_Native_Thinking",
    "direct": "E2_Direct_Prompting_No_Thinking",
    "native": "E3_Qwen_Native_Thinking_With_Instruction",
    "cot": "E4_Explicit_CoT",
    "structured": "E5_Structured_CoT",
    "english_pivot": "E6_English_Pivot_CoT",
    "plan_solve": "E7_Plan_and_Solve",
    "self_consistency": "E8_Self_Consistency_N5",
    "mspr": "E9_MSPR_Majority",
    "mspr_verifier": "E10_MSPR_Verifier",
    "answer_trigger": "E11_Question_Only_Staged_Answer_Trigger",
}


def build_prompt(technique: str, question_hi: str) -> str:
    if technique == "question_only":
        # The user content is exactly the Hindi question: no task instruction,
        # no wrapper text, no translation or answer-format instruction.
        return question_hi
    if technique == "direct":
        return (
            "इस गणितीय प्रश्न को हल करें। कृपया सीधे अंतिम उत्तर दें। "
            "अंतिम उत्तर स्पष्ट रूप से लिखें।\n\nप्रश्न:\n" + question_hi
        )
    if technique == "native":
        return "इस प्रश्न का उत्तर दें। अंतिम उत्तर स्पष्ट रूप से लिखें।\n\nप्रश्न:\n" + question_hi
    if technique == "cot":
        return (
            "इस गणितीय प्रश्न को ध्यान से हल करें। अपने समाधान में आवश्यक गणितीय "
            "चरण दिखाएँ और अंत में अंतिम उत्तर स्पष्ट रूप से लिखें। उत्तर हिंदी में दें।\n\n"
            "प्रश्न:\n" + question_hi
        )
    if technique == "structured":
        return (
            "इस हिंदी गणितीय प्रश्न को हल करें। समाधान को इस संरचना में दें:\n"
            "1. दी गई जानकारी: प्रश्न में दी गई संख्याएँ और शर्तें लिखें।\n"
            "2. क्या ज्ञात करना है: लक्ष्य पहचानें।\n"
            "3. विधि: उपयुक्त गणितीय क्रिया या समीकरण चुनें।\n"
            "4. गणना: आवश्यक चरण हल करें।\n"
            "5. अंतिम उत्तर: उत्तर और उसकी इकाई स्पष्ट रूप से लिखें।\n\n"
            "प्रश्न:\n" + question_hi
        )
    if technique == "english_pivot":
        return (
            "The following word problem is written in Hindi. First understand it and "
            "translate its meaning into English internally. Perform the mathematical "
            "reasoning in English. Do not assume any extra facts. Give the final answer "
            "in Hindi, preserving the requested unit. Show the key calculation and mark "
            "the final answer clearly.\n\nHindi question:\n" + question_hi
        )
    if technique == "self_consistency":
        return (
            "इस गणितीय प्रश्न को स्वतंत्र रूप से हल करें। आवश्यक गणितीय चरण दिखाएँ, "
            "गणनाओं की जाँच करें और अंतिम उत्तर स्पष्ट रूप से लिखें। उत्तर हिंदी में दें।\n\n"
            "प्रश्न:\n" + question_hi
        )
    raise ValueError(f"No single-prompt template for technique: {technique}")


def run_single(runner: QwenRunner, prompt: str, thinking: bool, seed: int) -> dict[str, Any]:
    generation = runner.generate(prompt, enable_thinking=thinking, seed=seed)
    answer_raw, numeral_style, answer_normalized = extract_answer(generation.final)
    return {
        "raw": generation.raw,
        "thinking": generation.thinking,
        "final": generation.final,
        "answer_raw": answer_raw,
        "numeral_system": numeral_style,
        "answer_normalized": answer_normalized,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "thinking_tokens": generation.thinking_tokens,
        "final_tokens": generation.final_tokens,
        "latency_seconds": generation.latency_seconds,
        "thinking_generation_time_seconds": generation.thinking_generation_time_seconds,
        "answer_generation_time_seconds": generation.answer_generation_time_seconds,
        "answer_trigger_text": generation.answer_trigger_text,
        "answer_trigger_injected": generation.answer_trigger_injected,
        "thinking_stop_reason": generation.thinking_stop_reason,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "prompt": prompt,
    }


def run_answer_triggered(runner: QwenRunner, prompt: str, seed: int) -> dict[str, Any]:
    generation = runner.generate_with_answer_trigger(prompt, seed)
    answer_raw, numeral_style, answer_normalized = extract_answer(generation.final)
    return {
        "raw": generation.raw,
        "thinking": generation.thinking,
        "final": generation.final,
        "answer_raw": answer_raw,
        "numeral_system": numeral_style,
        "answer_normalized": answer_normalized,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "thinking_tokens": generation.thinking_tokens,
        "final_tokens": generation.final_tokens,
        "latency_seconds": generation.latency_seconds,
        "thinking_generation_time_seconds": generation.thinking_generation_time_seconds,
        "answer_generation_time_seconds": generation.answer_generation_time_seconds,
        "answer_trigger_text": generation.answer_trigger_text,
        "answer_trigger_injected": generation.answer_trigger_injected,
        "thinking_stop_reason": generation.thinking_stop_reason,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "prompt": prompt,
    }


def candidate_details(candidates: list[dict[str, Any]]) -> str:
    safe = []
    for idx, candidate in enumerate(candidates, start=1):
        safe.append({
            "candidate_id": idx,
            "answer_raw": candidate.get("answer_raw", ""),
            "numeral_system": candidate.get("numeral_system", "Unparsed"),
            "answer_normalized": candidate.get("answer_normalized", ""),
            "thinking": candidate.get("thinking", ""),
            "final": candidate.get("final", ""),
            "raw": candidate.get("raw", ""),
            "thinking_generation_time_seconds": candidate.get("thinking_generation_time_seconds", 0.0),
            "answer_generation_time_seconds": candidate.get("answer_generation_time_seconds", 0.0),
            "answer_trigger_text": candidate.get("answer_trigger_text", ""),
            "answer_trigger_injected": candidate.get("answer_trigger_injected", False),
            "thinking_stop_reason": candidate.get("thinking_stop_reason", "not_applicable"),
        })
    return json.dumps(safe, ensure_ascii=False)


def run_plan_and_solve(runner: QwenRunner, question: str, seed: int) -> tuple[list[dict[str, Any]], str]:
    plan_prompt = (
        "इस हिंदी गणितीय प्रश्न को हल करने की संक्षिप्त योजना बनाएँ। "
        "अभी अंतिम गणना या उत्तर न दें। केवल आवश्यक चरणों की योजना दें।\n\nप्रश्न:\n" + question
    )
    plan = run_single(runner, plan_prompt, True, seed)
    plan_text = "\n".join(part for part in (plan["thinking"], plan["final"]) if part).strip()
    solve_prompt = (
        "अब नीचे दिए गए प्रश्न को हल करें। पहले दी गई योजना का उपयोग करें, लेकिन यदि "
        "योजना में गलती हो तो उसे सुधारें। आवश्यक गणना दिखाएँ और अंतिम उत्तर हिंदी में दें।\n\n"
        f"प्रश्न:\n{question}\n\nयोजना:\n{plan_text}"
    )
    solution = run_single(runner, solve_prompt, True, seed + 1)
    prompts = json.dumps({"plan_prompt": plan_prompt, "solve_prompt": solve_prompt}, ensure_ascii=False)
    return [plan, solution], prompts


def run_mspr(runner: QwenRunner, question: str, seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    strategies = [
        ("arithmetic", "संख्यात्मक गणना और सीधे अंकगणितीय क्रियाओं से हल करें।"),
        ("equation", "यदि उपयुक्त हो तो चर और समीकरण बनाकर हल करें; यदि समीकरण उपयुक्त न हो तो दूसरी वैध विधि चुनें।"),
        ("decomposition", "समस्या को छोटे उप-चरणों में बाँटकर हल करें और फिर परिणाम जोड़ें या संयोजित करें।"),
        ("alternative", "पहले से अलग एक स्वतंत्र गणितीय तरीका खोजकर हल करें; यदि अलग तरीका संभव न हो तो स्वतंत्र रूप से जाँच करें।"),
    ]
    outputs: list[dict[str, Any]] = []
    prompts: list[str] = []
    for i, (name, instruction) in enumerate(strategies):
        prompt = (
            f"आप इस समस्या को '{name}' रणनीति से हल कर रहे हैं। {instruction} "
            "सभी गणनाएँ जाँचें और अंतिम उत्तर हिंदी में स्पष्ट रूप से लिखें।\n\n"
            f"प्रश्न:\n{question}"
        )
        prompts.append(prompt)
        outputs.append(run_single(runner, prompt, True, seed + i))
    return outputs, prompts


def parse_verifier_choice(text: str, candidate_count: int) -> Optional[int]:
    patterns = [
        r'"best_candidate_id"\s*:\s*(\d+)',
        r"best_candidate_id\s*[:=]\s*(\d+)",
        r"best\s+candidate\s*(?:is|:|#)?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            choice = int(match.group(1))
            if 1 <= choice <= candidate_count:
                return choice - 1
    return None


def run_verifier(
    runner: QwenRunner,
    question: str,
    candidates: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    candidate_block = []
    for i, candidate in enumerate(candidates, start=1):
        reasoning_text = "\n".join(
            part for part in (candidate.get("thinking", ""), candidate.get("final", "")) if part
        )
        candidate_block.append(
            f"CANDIDATE {i}\nReasoning and response:\n{reasoning_text}\n"
            f"Extracted final answer: {candidate['answer_raw'] or 'not extracted'}"
        )
    prompt = (
        "आप गणितीय समाधान के verifier हैं। मूल प्रश्न को स्वतंत्र रूप से समझें और प्रत्येक "
        "candidate की गणितीय वैधता जाँचें। केवल भाषा या शैली के आधार पर निर्णय न लें। "
        "गलत गणना, गलत operation, छूटी हुई शर्त और गलत अंतिम उत्तर को दंडित करें। "
        "अंत में केवल JSON दें, इस schema के अनुसार: "
        '{"best_candidate_id": 1, "candidate_scores": '
        '[{"candidate_id": 1, "score": 0, "reason": "..."}]}. '
        "score 0 से 2 तक हो: 0=गलत/अपूर्ण, 1=आंशिक/अनिश्चित, 2=गणितीय रूप से सही।\n\n"
        f"मूल हिंदी प्रश्न:\n{question}\n\n" + "\n\n".join(candidate_block)
    )
    return run_single(runner, prompt, True, seed)


def aggregate_generation_stats(generations: list[dict[str, Any]]) -> dict[str, Any]:
    # Keep one-call generations readable as plain text; serialize multiple calls
    # as JSON arrays so no candidate's reasoning/output is lost in the CSV.
    if len(generations) == 1:
        raw_output: str = generations[0]["raw"]
        thinking_text: str = generations[0]["thinking"]
        final_response: str = generations[0]["final"]
    else:
        raw_output = json.dumps([g["raw"] for g in generations], ensure_ascii=False)
        thinking_text = json.dumps([g["thinking"] for g in generations], ensure_ascii=False)
        final_response = json.dumps([g["final"] for g in generations], ensure_ascii=False)
    return {
        "raw_model_output": raw_output,
        "thinking_text": thinking_text,
        "final_response": final_response,
        "input_tokens": sum(g["input_tokens"] for g in generations),
        "output_tokens": sum(g["output_tokens"] for g in generations),
        "thinking_tokens": sum(g["thinking_tokens"] for g in generations),
        "final_tokens": sum(g["final_tokens"] for g in generations),
        "latency_seconds": sum(g["latency_seconds"] for g in generations),
        "thinking_generation_time_seconds": sum(g.get("thinking_generation_time_seconds", 0.0) for g in generations),
        "answer_generation_time_seconds": sum(g.get("answer_generation_time_seconds", 0.0) for g in generations),
        "answer_trigger_text": json.dumps([g.get("answer_trigger_text", "") for g in generations if g.get("answer_trigger_text")], ensure_ascii=False) if any(g.get("answer_trigger_text") for g in generations) else "",
        "answer_trigger_injected": any(g.get("answer_trigger_injected", False) for g in generations),
        "thinking_stop_reason": json.dumps([g.get("thinking_stop_reason", "not_applicable") for g in generations], ensure_ascii=False) if any(g.get("thinking_stop_reason", "not_applicable") != "not_applicable" for g in generations) else "not_applicable",
    }


def load_questions(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No questions found in {path}. Run prepare_dataset.py first.")
    for row in rows:
        if not row.get("question_hi", "").strip():
            raise ValueError(f"Missing full Hindi question in row {row.get('question_id')}")
    return rows


def make_result_row(
    source: dict[str, Any], technique_id: str, technique_name: str,
    model_id: str, generations: list[dict[str, Any]], selected: dict[str, Any],
    thinking_enabled: bool, num_candidates: int, prompt_text: str,
    verifier: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    gold_raw, gold_style, gold_normalized = get_gold_answer(source)
    predicted_raw = selected.get("answer_raw", "")
    predicted_style = selected.get("numeral_system", "Unparsed")
    predicted_normalized = selected.get("answer_normalized", "")
    is_correct: Any = None
    if gold_normalized and predicted_normalized:
        is_correct = gold_normalized == predicted_normalized
    elif gold_normalized and not predicted_normalized:
        is_correct = False

    stats = aggregate_generation_stats(generations)
    verifier = verifier or {}
    all_generations = generations + ([verifier] if verifier else [])
    raw_model_output = stats["raw_model_output"]
    thinking_text = stats["thinking_text"]
    final_response = stats["final_response"]
    verifier_raw = verifier.get("raw", "")
    verifier_thinking = verifier.get("thinking", "")
    if verifier:
        raw_model_output = json.dumps({
            "candidate_generations": [g["raw"] for g in generations],
            "verifier_generation": verifier_raw,
        }, ensure_ascii=False)
        thinking_text = json.dumps({
            "candidate_thinking": [g["thinking"] for g in generations],
            "verifier_thinking": verifier_thinking,
        }, ensure_ascii=False)
        final_response = json.dumps({
            "candidate_final_responses": [g["final"] for g in generations],
            "verifier_final_response": verifier.get("final", ""),
        }, ensure_ascii=False)

    temp_values = [g["temperature"] for g in all_generations]
    top_p_values = [g["top_p"] for g in all_generations]
    top_k_values = [g["top_k"] for g in all_generations]
    return {
        "question_id": source.get("question_id", ""),
        "source_row_index": source.get("source_row_index", ""),
        "dataset_id": source.get("dataset_id", "sarvamai/gsm8k-indic"),
        "dataset_config": source.get("dataset_config", "hi"),
        "dataset_split": source.get("dataset_split", "test"),
        "question_hi": source.get("question_hi", ""),
        "question_original": source.get("question_original", ""),
        "reference_solution_hi": source.get("reference_solution_hi", ""),
        "reference_solution_original": source.get("reference_solution_original", ""),
        "gold_answer_raw": gold_raw,
        "gold_answer_source": source.get("gold_answer_source", ""),
        "gold_answer_normalized": gold_normalized,
        "gold_numeral_system": gold_style,
        "model": model_id,
        "model_dtype": "bfloat16",
        "device": "cuda",
        "gpu_name": "",
        "gpu_total_memory_gib": "",
        "technique_id": technique_id,
        "technique_name": technique_name,
        "thinking_enabled": thinking_enabled,
        "temperature": statistics.mean(temp_values) if temp_values else "",
        "top_p": statistics.mean(top_p_values) if top_p_values else "",
        "top_k": top_k_values[0] if top_k_values else "",
        "max_new_tokens": "CLI-configured",
        "answer_max_new_tokens": "CLI-configured",
        "num_candidates": num_candidates,
        "prompt_text": prompt_text,
        "raw_model_output": raw_model_output,
        "thinking_text": thinking_text,
        "final_response": final_response,
        "candidate_details_json": candidate_details(generations),
        "verifier_raw_output": verifier_raw,
        "verifier_thinking_text": verifier_thinking,
        "answer_trigger_text": stats["answer_trigger_text"],
        "answer_trigger_injected": stats["answer_trigger_injected"],
        "thinking_stop_reason": stats["thinking_stop_reason"],
        "thinking_generation_time_seconds": stats["thinking_generation_time_seconds"],
        "answer_generation_time_seconds": stats["answer_generation_time_seconds"],
        "extracted_answer_raw": predicted_raw,
        "predicted_numeral_system": predicted_style,
        "predicted_answer_normalized": predicted_normalized,
        "is_correct": is_correct,
        "answer_extraction_status": "parsed_numeric" if predicted_normalized else (
            "word_based" if predicted_style == "Hindi-word-based" else "unparsed"
        ),
        "input_tokens": stats["input_tokens"] + (verifier.get("input_tokens", 0) if verifier else 0),
        "output_tokens": stats["output_tokens"] + (verifier.get("output_tokens", 0) if verifier else 0),
        "thinking_tokens": stats["thinking_tokens"] + (verifier.get("thinking_tokens", 0) if verifier else 0),
        "final_tokens": stats["final_tokens"] + (verifier.get("final_tokens", 0) if verifier else 0),
        # Generation time is the sum of model.generate() durations across all
        # candidate calls and the optional verifier call.
        "latency_seconds": stats["latency_seconds"] + (verifier.get("latency_seconds", 0.0) if verifier else 0.0),
        "model_generation_time_seconds": stats["latency_seconds"] + (verifier.get("latency_seconds", 0.0) if verifier else 0.0),
        "candidate_generation_time_seconds": stats["latency_seconds"],
        "verifier_generation_time_seconds": verifier.get("latency_seconds", 0.0) if verifier else 0.0,
        # Set by main() after the complete strategy (generation + aggregation +
        # answer extraction/result assembly) has finished for this question.
        "technique_inference_time_seconds": "",
        "run_status": "ok",
        "error_message": "",
    }


def run_technique(
    runner: QwenRunner, row: dict[str, Any], technique: str, row_seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], bool, int, str, Optional[dict[str, Any]]]:
    question = row["question_hi"]
    verifier = None
    if technique == "question_only":
        prompt = build_prompt("question_only", question)
        outputs = [run_single(runner, prompt, True, row_seed)]
        return outputs, outputs[0], True, 1, prompt, verifier
    if technique == "answer_trigger":
        # Same content as question_only; the only experimental difference is
        # stopping at </think> and continuing after Qwen's answer boundary.
        prompt = build_prompt("question_only", question)
        outputs = [run_answer_triggered(runner, prompt, row_seed)]
        return outputs, outputs[0], True, 1, prompt, verifier
    if technique == "direct":
        prompt = build_prompt("direct", question)
        outputs = [run_single(runner, prompt, False, row_seed)]
        return outputs, outputs[0], False, 1, prompt, verifier
    if technique == "native":
        prompt = build_prompt("native", question)
        outputs = [run_single(runner, prompt, True, row_seed)]
        return outputs, outputs[0], True, 1, prompt, verifier
    if technique in {"cot", "structured", "english_pivot"}:
        prompt = build_prompt(technique, question)
        outputs = [run_single(runner, prompt, True, row_seed)]
        return outputs, outputs[0], True, 1, prompt, verifier
    if technique == "plan_solve":
        outputs, solve_prompt = run_plan_and_solve(runner, question, row_seed)
        return outputs, outputs[-1], True, 2, solve_prompt, verifier
    if technique == "self_consistency":
        prompt = build_prompt("self_consistency", question)
        outputs = [run_single(runner, prompt, True, row_seed + i) for i in range(5)]
        selected, _ = majority_select(outputs)
        return outputs, selected, True, 5, prompt, verifier
    if technique in {"mspr", "mspr_verifier"}:
        outputs, prompts = run_mspr(runner, question, row_seed)
        selected, _ = majority_select(outputs)
        if technique == "mspr_verifier":
            verifier = run_verifier(runner, question, outputs, row_seed + 10)
            choice = parse_verifier_choice(verifier["final"], len(outputs))
            if choice is not None:
                selected = outputs[choice]
        return outputs, selected, True, len(outputs), json.dumps(prompts, ensure_ascii=False), verifier
    raise ValueError(f"Unknown technique: {technique}")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    # Write the UTF-8 BOM only once, not every time a row is appended.
    encoding = "utf-8" if exists else "utf-8-sig"
    with path.open("a", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    import pandas as pd
    df = pd.read_csv(results_path, encoding="utf-8-sig")
    if df.empty:
        return
    df["is_correct_num"] = df["is_correct"].map({True: 1, False: 0, "True": 1, "False": 0})

    # Backward-compatible defaults allow a summary to be regenerated from older
    # result CSVs that predate the detailed timing columns.
    timing_columns = [
        "latency_seconds", "model_generation_time_seconds",
        "candidate_generation_time_seconds", "verifier_generation_time_seconds",
        "technique_inference_time_seconds", "thinking_generation_time_seconds",
        "answer_generation_time_seconds",
    ]
    for column in timing_columns:
        if column not in df.columns:
            df[column] = pd.NA

    # Older result files may have latency_seconds but not its clearer alias.
    df["model_generation_time_seconds"] = df["model_generation_time_seconds"].fillna(
        df["latency_seconds"]
    )

    summary = df.groupby(["model", "technique_id", "technique_name"], dropna=False).agg(
        questions_run=("question_id", "count"),
        accuracy=("is_correct_num", "mean"),
        # Generation-only time: sum of model.generate() calls.
        avg_latency_seconds=("model_generation_time_seconds", "mean"),
        avg_model_generation_time_seconds=("model_generation_time_seconds", "mean"),
        total_model_generation_time_seconds=("model_generation_time_seconds", "sum"),
        total_candidate_generation_time_seconds=("candidate_generation_time_seconds", "sum"),
        total_verifier_generation_time_seconds=("verifier_generation_time_seconds", "sum"),
        # Complete technique wall time per question, then accumulated over questions.
        avg_technique_inference_time_seconds=("technique_inference_time_seconds", "mean"),
        total_technique_inference_time_seconds=("technique_inference_time_seconds", "sum"),
        median_technique_inference_time_seconds=("technique_inference_time_seconds", "median"),
        max_technique_inference_time_seconds=("technique_inference_time_seconds", "max"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_thinking_tokens=("thinking_tokens", "mean"),
        avg_thinking_generation_time_seconds=("thinking_generation_time_seconds", "mean"),
        total_thinking_generation_time_seconds=("thinking_generation_time_seconds", "sum"),
        avg_answer_generation_time_seconds=("answer_generation_time_seconds", "mean"),
        total_answer_generation_time_seconds=("answer_generation_time_seconds", "sum"),
        forced_thinking_closures=("answer_trigger_injected", lambda s: int((s.astype(str).str.lower() == "true").sum())),
        devanagari_outputs=("predicted_numeral_system", lambda s: int((s == "Devanagari").sum())),
        hindu_arabic_outputs=("predicted_numeral_system", lambda s: int((s == "Hindu-Arabic").sum())),
        mixed_outputs=("predicted_numeral_system", lambda s: int((s == "Mixed").sum())),
        unparsed_outputs=("predicted_numeral_system", lambda s: int((s == "Unparsed").sum())),
    ).reset_index()
    summary["accuracy_percent"] = summary["accuracy"] * 100
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Sample CSV made by prepare_dataset.py")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=MODEL_ID_DEFAULT)
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum tokens for the thinking/candidate generation phase")
    parser.add_argument("--answer-max-new-tokens", type=int, default=512, help="Maximum tokens for the staged answer-trigger phase")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index; CPU execution is disabled")
    parser.add_argument("--min-free-vram-gib", type=float, default=9.0, help="Minimum free VRAM required before loading BF16 model")
    parser.add_argument("--max-questions", type=int, default=0, help="0 means all questions; use 1-3 for a smoke test")
    parser.add_argument("--techniques", default="all", help="all or comma-separated IDs: " + ",".join(TECHNIQUES))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", help="Skip question/technique rows already present in output CSV")
    args = parser.parse_args()

    questions = load_questions(args.input)
    if args.max_questions > 0:
        questions = questions[:args.max_questions]

    selected_techniques = list(TECHNIQUES) if args.techniques == "all" else [x.strip() for x in args.techniques.split(",") if x.strip()]
    unknown = set(selected_techniques).difference(TECHNIQUES)
    if unknown:
        raise ValueError(f"Unknown techniques: {sorted(unknown)}. Valid: {list(TECHNIQUES)}")

    output_path = Path(args.output)
    completed: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8-sig") as f:
            for existing in csv.DictReader(f):
                completed.add((existing.get("question_id", ""), existing.get("technique_id", "")))
        print(f"Resume enabled: found {len(completed)} completed question/technique rows")
    elif output_path.exists() and not args.resume:
        output_path.unlink()

    runner = QwenRunner(
        args.model,
        args.max_new_tokens,
        answer_max_new_tokens=args.answer_max_new_tokens,
        gpu_id=args.gpu_id,
        min_free_vram_gib=args.min_free_vram_gib,
    )
    total = len(questions) * len(selected_techniques)
    progress = 0
    for q_index, source in enumerate(questions):
        question_id = source["question_id"]
        for t_index, technique in enumerate(selected_techniques):
            progress += 1
            if (question_id, technique) in completed:
                print(f"[{progress}/{total}] SKIP {question_id} / {technique}")
                continue
            print(f"[{progress}/{total}] {question_id} / {TECHNIQUES[technique]}")
            row_seed = args.seed + q_index * 1000 + t_index * 100
            # End-to-end wall time for this question/technique. This includes all
            # model calls, prompt orchestration, candidate aggregation, optional
            # verification, and final answer extraction/result assembly. It excludes
            # one-time model loading, dataset loading, and CSV disk writing.
            technique_start = time.perf_counter()
            try:
                generations, selected, thinking_enabled, n_candidates, prompt_text, verifier = run_technique(
                    runner, source, technique, row_seed
                )
                result = make_result_row(
                    source=source,
                    technique_id=technique,
                    technique_name=TECHNIQUES[technique],
                    model_id=args.model,
                    generations=generations,
                    selected=selected,
                    thinking_enabled=thinking_enabled,
                    num_candidates=n_candidates,
                    prompt_text=prompt_text,
                    verifier=verifier,
                )
                result["device"] = str(runner.device)
                result["gpu_name"] = runner.gpu_name
                result["gpu_total_memory_gib"] = runner.gpu_total_memory_gib
                result["max_new_tokens"] = args.max_new_tokens
                result["answer_max_new_tokens"] = args.answer_max_new_tokens
                result["technique_inference_time_seconds"] = round(
                    time.perf_counter() - technique_start, 6
                )
            except Exception as exc:
                elapsed_technique = round(time.perf_counter() - technique_start, 6)
                print(f"ERROR: {type(exc).__name__}: {exc}")
                result = {field: "" for field in CSV_FIELDS}
                result.update({
                    "question_id": question_id,
                    "source_row_index": source.get("source_row_index", ""),
                    "dataset_id": source.get("dataset_id", "sarvamai/gsm8k-indic"),
                    "dataset_config": "hi",
                    "dataset_split": "test",
                    "question_hi": source.get("question_hi", ""),
                    "question_original": source.get("question_original", ""),
                    "reference_solution_hi": source.get("reference_solution_hi", ""),
                    "reference_solution_original": source.get("reference_solution_original", ""),
                    "gold_answer_raw": source.get("gold_answer_raw", ""),
                    "model": args.model,
                    "model_dtype": "bfloat16",
                    "device": str(runner.device),
                    "gpu_name": runner.gpu_name,
                    "gpu_total_memory_gib": runner.gpu_total_memory_gib,
                    "max_new_tokens": args.max_new_tokens,
                    "answer_max_new_tokens": args.answer_max_new_tokens,
                    "technique_id": technique,
                    "technique_name": TECHNIQUES[technique],
                    "technique_inference_time_seconds": elapsed_technique,
                    "run_status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                })
            append_csv(output_path, result)
            if result.get("technique_inference_time_seconds") != "":
                print(
                    f"    technique time: {float(result['technique_inference_time_seconds']):.2f}s | "
                    f"model generation: {float(result.get('model_generation_time_seconds') or 0):.2f}s"
                )

    write_summary(output_path, Path(args.summary))
    print("\nExperiment run complete.")
    print(f"Detailed results: {output_path.resolve()}")
    print(f"Technique summary: {Path(args.summary).resolve()}")


if __name__ == "__main__":
    main()
