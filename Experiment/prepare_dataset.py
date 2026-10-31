"""Prepare a reproducible 200-question Hindi GSM8K sample from Sarvam AI."""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "sarvamai/gsm8k-indic"
CONFIG = "hi"
SPLIT = "test"
SEED = 42
SAMPLE_SIZE = 200

DEVANAGARI_DIGITS = "०१२३४५६७८९"
ASCII_DIGITS = "0123456789"
DIGIT_TRANSLATION = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)


def normalize_digits(text: str) -> str:
    return str(text).translate(DIGIT_TRANSLATION)


def find_final_segment(text: str) -> str:
    """Return the text after GSM8K's #### marker, when present."""
    if not text:
        return ""
    matches = list(re.finditer(r"####\s*(.+?)\s*$", str(text), flags=re.S))
    return matches[-1].group(1).strip() if matches else str(text).strip()


def extract_gold_answer(hindi_solution: str, original_solution: str) -> tuple[str, str]:
    """Prefer the canonical original GSM8K final answer; retain Hindi fallback."""
    for source_name, solution in (
        ("original_answer", original_solution),
        ("answer", hindi_solution),
    ):
        segment = find_final_segment(solution)
        # GSM8K final answers are usually a single number after ####.
        normalized = normalize_digits(segment)
        number = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?%?", normalized)
        if number:
            return number.group(0).strip(), source_name
    return "", "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/gsm8k_hi_200.csv")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Loading {DATASET_ID} / config={CONFIG} / split={SPLIT} ...")
    dataset = load_dataset(DATASET_ID, CONFIG, split=SPLIT)
    required = {"question", "answer", "original_question", "original_answer"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Unexpected Sarvam dataset schema. Missing {sorted(missing)}; "
            f"available columns: {dataset.column_names}"
        )

    if args.sample_size > len(dataset):
        raise ValueError(f"Requested {args.sample_size} rows, but split has {len(dataset)}")

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(dataset)), args.sample_size))

    rows: list[dict[str, Any]] = []
    for idx in selected_indices:
        item = dataset[idx]
        gold_raw, gold_source = extract_gold_answer(
            item.get("answer", ""), item.get("original_answer", "")
        )
        if not item.get("question"):
            raise ValueError(f"Empty Hindi question at source row {idx}")
        if not gold_raw:
            print(f"WARNING: could not extract gold answer at source row {idx}")

        rows.append({
            "question_id": f"SARVAM_GSM8K_HI_{idx:04d}",
            "source_row_index": idx,
            "dataset_id": DATASET_ID,
            "dataset_config": CONFIG,
            "dataset_split": SPLIT,
            "question_hi": item["question"],
            "question_original": item.get("original_question", ""),
            "reference_solution_hi": item.get("answer", ""),
            "reference_solution_original": item.get("original_answer", ""),
            "gold_answer_raw": gold_raw,
            "gold_answer_source": gold_source,
            "sample_seed": args.seed,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} questions to: {output.resolve()}")
    print("Hindi question field: question")
    print("Hindi reference solution field: answer")
    print("Canonical gold answer prefers original_answer, with Hindi fallback.")
    print("Review the sample CSV before running model inference.")


if __name__ == "__main__":
    main()
