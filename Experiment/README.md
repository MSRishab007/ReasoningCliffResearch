# Hindi Math Reasoning with Qwen3-1.7B

Inference-time reasoning experiments on the Hindi configuration of Sarvam AI's GSM8K-Indic dataset.

## Dataset

- Hugging Face dataset: `sarvamai/gsm8k-indic`
- Configuration: `hi`
- Split: `test` (1,319 rows)
- Hindi question column: `question`
- Hindi reference-solution column: `answer`
- Original-language columns: `original_question`, `original_answer`
- Sample size: 200 (default)
- Sampling seed: 42 (default)

The Hindi question and solution are used for model evaluation. The original-language question/solution are retained as reference metadata and are **not** supplied to the model.

## Implemented techniques

| ID | Technique | Inference calls per question |
|---|---|---:|
| E1 | Direct prompting (thinking disabled) | 1 |
| E2 | Qwen native thinking | 1 |
| E3 | Explicit Chain-of-Thought | 1 |
| E4 | Structured CoT | 1 |
| E5 | English-pivot CoT | 1 |
| E6 | Plan-and-Solve | 2 |
| E7 | Self-Consistency (5 samples) | 5 |
| E8 | Multi-Strategy Parallel Reasoning (4 strategies) | 4 |
| E9 | MSPR + model-based verifier | 5 |

All techniques use the same unmodified Qwen3-1.7B checkpoint. No fine-tuning is performed.

## Qwen thinking parser

Official Qwen3 uses `<think> ... </think>`. Depending on the chat template, the opening marker may be part of the prompt, so generated output may contain the thinking text followed by `</think>` without a generated `<think>` prefix. The parser handles this form and also accepts `</thinking>` for compatibility with alternate wrappers. Both raw generation and the extracted thinking text are saved.

## Install

Create and activate a virtual environment, then install a PyTorch build appropriate to your hardware. Then:

```bash
pip install -r requirements.txt
```

## Step 1: Prepare a fixed 200-question sample

```bash
python prepare_dataset.py
```

This creates:

```text
data/gsm8k_hi_200.csv
```

Review this file. It contains the complete Hindi question, the Hindi reference solution, the original question/solution, and the extracted gold answer.

## Step 2: Run a small smoke test

Run one question and two techniques first to confirm model loading, generation, and CSV output:

```bash
python run_experiments.py --max-questions 1 --techniques direct,native
```

Then run all nine techniques on the first few questions:

```bash
python run_experiments.py --max-questions 3
```

## Step 3: Run all 200 questions

```bash
python run_experiments.py
```

To run selected techniques:

```bash
python run_experiments.py --techniques cot,structured,self_consistency,mspr,mspr_verifier
```

To continue an interrupted run without duplicating completed question/technique pairs:

```bash
python run_experiments.py --resume
```

Use `--max-new-tokens` to adjust the generation limit. The default is 2,048 generated tokens per model call.

## Output files

- `results/qwen3_1_7b_gsm8k_hi_results.csv`: one row per question per technique.
- `results/summary_by_technique.csv`: aggregated accuracy, timing, token use and numeral-style counts.

Important detailed CSV columns:

- `question_hi`: full Hindi question.
- `raw_model_output`: exact decoded generation including Qwen special markers.
- `thinking_text`: extracted thinking section(s); multiple candidate outputs are stored as JSON arrays.
- `final_response`: final answer section(s).
- `extracted_answer_raw`: answer span before numeral conversion.
- `predicted_numeral_system`: `Hindu-Arabic`, `Devanagari`, `Mixed`, `Hindi-word-based`, `Other-text`, or `Unparsed`.
- `predicted_answer_normalized`: normalized numeric answer used for exact matching.
- `gold_answer_raw` / `gold_answer_normalized`: reference answer in raw/canonical form.
- `is_correct`: numeric exact-match result after normalization.
- `candidate_details_json`: all candidate outputs and their extracted answers for multi-path methods.
- `latency_seconds` / `model_generation_time_seconds`: total model-generation time for this question and technique, summed across every candidate and optional verifier call.
- `candidate_generation_time_seconds`: generation time for candidate solutions only.
- `verifier_generation_time_seconds`: generation time for the verifier call, if used.
- `technique_inference_time_seconds`: complete end-to-end time for the technique on this question, including all calls, orchestration, voting/selection, verification, answer extraction and result assembly.

The normalizer converts Devanagari digits (०१२३४५६७८९) into ASCII digits (0123456789) before numerical comparison. It does not silently convert percentages into plain numbers or strip units from the stored answer text.

### Timing interpretation

The detailed CSV records complete per-question timing in `technique_inference_time_seconds`. For multi-call methods such as Plan-and-Solve, Self-Consistency, MSPR and MSPR + Verifier, this is the elapsed time for the entire technique—not just the final generation call. The summary CSV reports:

- `total_technique_inference_time_seconds`: sum of the per-question end-to-end technique times across the questions that ran.
- `avg_technique_inference_time_seconds`: average end-to-end time per question.
- `median_technique_inference_time_seconds` and `max_technique_inference_time_seconds`: distribution of per-question time.
- `total_model_generation_time_seconds`: sum of model generation durations across candidate and verifier calls.
- `total_candidate_generation_time_seconds` and `total_verifier_generation_time_seconds`: timing breakdown.

The technique total excludes model loading, dataset loading, and CSV file writes. It is the accumulated time spent executing that technique across its questions, rather than the overall wall-clock duration of the entire mixed-technique run.

## Notes on interpretation

1. `thinking_text` is the model-generated reasoning text, not a guarantee that it faithfully represents internal model computation.
2. The same model acts as the verifier in E9; this is self-verification, not an independently trained verifier.
3. Self-Consistency and MSPR use more model calls than single-path baselines. Compare both raw accuracy and token/latency cost.
4. The verifier can select a candidate only if it emits a parseable `best_candidate_id`; otherwise the code falls back to majority voting.
5. The script records the model's numeral style before normalizing the number, so the CSV can be used to study numeral-format preference separately from correctness.
6. The original English fields are kept in the CSV but are not included in prompts, including the English-pivot condition. The model must interpret the Hindi question itself.

## Suggested reproducibility settings

- Sample seed: 42
- Run seed: 2026
- Model: `Qwen/Qwen3-1.7B`
- Thinking mode temperature/top-p/top-k: 0.6 / 0.95 / 20
- Non-thinking temperature/top-p/top-k: 0.7 / 0.8 / 20
- Max new tokens: 2,048

For the Qwen3 thinking-mode settings and marker format, see the official model card: https://huggingface.co/Qwen/Qwen3-1.7B
