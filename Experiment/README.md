# Hindi Math Reasoning with Qwen3-4B (BF16, GPU-only)

Inference-time reasoning experiments on Sarvam AI's Hindi GSM8K dataset.

## Experimental setup

- Model: `Qwen/Qwen3-4B`
- Precision: BF16 (`torch.bfloat16`)
- Device: CUDA GPU only; CPU fallback and automatic CPU offload are disabled.
- Expected hardware: 16-GB-class GPU with native BF16 support and at least 9 GiB free VRAM before loading.
- Dataset: `sarvamai/gsm8k-indic`, config `hi`, split `test`.
- Sample: 200 Hindi questions, random seed 42.
- Fine-tuning: none.

The runner verifies CUDA availability, native BF16 support, GPU memory, model device, and model parameter dtype before inference begins. If any assertion fails, it stops rather than silently running on CPU or FP16.

## Techniques

| ID | CLI key | Technique |
|---|---|---|
| E1 | `question_only` | Hindi question as the entire user prompt; Qwen thinking mode on |
| E2 | `direct` | Explicit direct-answer instruction; thinking mode off |
| E3 | `native` | Minimal Hindi instruction with native Qwen thinking |
| E4 | `cot` | Explicit Chain-of-Thought |
| E5 | `structured` | Structured mathematical reasoning |
| E6 | `english_pivot` | Reason in English, answer in Hindi |
| E7 | `plan_solve` | Plan first, then solve |
| E8 | `self_consistency` | Five independent CoT candidates + majority voting |
| E9 | `mspr` | Four different mathematical approaches + majority voting |
| E10 | `mspr_verifier` | MSPR + a separate self-verifier call |
| E11 | `answer_trigger` | Exact Hindi question prompt; generate thought to `</think>`, then continue after Qwen's answer boundary |

### About the answer trigger

Qwen3's documented message format uses `<think> ... </think>` for reasoning, then a blank-line boundary (`\n\n`) and the normal assistant answer. There is not a separate official `<answer>` special token. E11 implements a staged version of that format:

1. Generate the thinking phase until the `</think>` token is emitted (or the configured thinking budget is reached).
2. If the model reaches the token limit without closing its thought, append a closing `</think>` marker and record this intervention.
3. Append the Qwen answer boundary (`\n\n`) and resume generation for the final answer.

The CSV records the trigger text, whether a closing marker had to be injected, the stop reason, thinking-generation time, and answer-generation time. E1 and E11 use the same user prompt content—the Hindi question only—so they provide a direct comparison between ordinary one-pass generation and staged answer triggering.

## Install

Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver first. Then:

```bash
pip install -r requirements.txt
```

## Step 1: Prepare the fixed 200-question Sarvam Hindi GSM8K sample

```bash
python prepare_dataset.py
```

Creates `data/gsm8k_hi_200.csv`. The complete Hindi question is stored in `question_hi`; Hindi and original reference solutions are retained. The source question IDs and the random seed are preserved.

## Step 2: Check the GPU and run a smoke test

First check the available CLI options:

```bash
python run_experiments.py --help
```

Then run only the question-only and staged-answer-trigger methods on one item:

```bash
python run_experiments.py --max-questions 1 --techniques question_only,answer_trigger
```

The runner prints the selected GPU, total/free VRAM, loaded model device and dtype. It stops if CUDA or BF16 is unavailable, if the GPU is not 16-GB-class, or if free VRAM is below 9 GiB.

Run a three-question smoke test for all techniques:

```bash
python run_experiments.py --max-questions 3
```

## Step 3: Run all 200 questions

```bash
python run_experiments.py
```

Optional parameters:

```bash
python run_experiments.py --gpu-id 0 --max-new-tokens 2048 --answer-max-new-tokens 512
```

If GPU applications use memory, close them before starting. The minimum free-memory guard can be adjusted deliberately with `--min-free-vram-gib`, but do not lower it unless you have checked the available memory.

## Output files

- `results/qwen3_4b_bf16_gsm8k_hi_results.csv`: one row per question per technique.
- `results/qwen3_4b_bf16_summary_by_technique.csv`: accuracy, timing, token and numeral-format summaries.

Important detailed CSV fields:

- `question_hi`: the full original Hindi question.
- `prompt_text`: exactly what was passed as user content. For `question_only` and `answer_trigger`, it is only the Hindi question text.
- `raw_model_output`: full decoded generation, including Qwen markers where present.
- `thinking_text`: extracted model-generated thinking content.
- `final_response`: extracted final answer content.
- `predicted_numeral_system`: `Hindu-Arabic`, `Devanagari`, `Mixed`, `Hindi-word-based`, `Other-text`, or `Unparsed`.
- `extracted_answer_raw`: answer as generated, before numeral normalization.
- `predicted_answer_normalized`: canonical form used for answer matching.
- `model_dtype`, `device`, `gpu_name`, `gpu_total_memory_gib`: record the actual execution configuration.
- `answer_trigger_text`, `answer_trigger_injected`, `thinking_stop_reason`: staged answer-trigger metadata.
- `thinking_generation_time_seconds`, `answer_generation_time_seconds`: timing breakdown for E11.
- `technique_inference_time_seconds`: complete per-question duration for the technique, including all generation calls and aggregation. Model loading, dataset loading and CSV writes are excluded.
- `total_technique_inference_time_seconds`: accumulated end-to-end technique time across the questions in the summary.

The numeral normalizer converts Devanagari digits (०१२३४५६७८९) to ASCII digits (0123456789) for correctness checking, while preserving the original answer and detected numeral system in separate columns.

## Reproducibility

- Dataset sample seed: 42
- Inference seed: 2026
- Model: `Qwen/Qwen3-4B`
- Precision: BF16
- Thinking mode temperature / top-p / top-k: 0.6 / 0.95 / 20
- Non-thinking mode temperature / top-p / top-k: 0.7 / 0.8 / 20
- Thinking token limit: 2048
- Staged final-answer token limit: 512

See the official [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B) and [Qwen3 thinking format documentation](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md).
