# Hindi Math Reasoning with Qwen3-4B (4-bit NF4 weights, BF16 compute, GPU-only)

Inference-time reasoning experiments on Sarvam AI's Hindi GSM8K dataset.

## Experimental setup

- Model: `Qwen/Qwen3-4B`
- Weight format: 4-bit NF4 using bitsandbytes, with double quantization.
- Compute dtype: BF16 (`torch.bfloat16`) for 4-bit linear-layer matrix operations and non-quantized model modules. This is NOT full-BF16 weight storage.
- Device: CUDA GPU only; CPU fallback and automatic CPU/disk offload are disabled.
- Target hardware: 6-GB-class RTX 3050 or better with native BF16 support and at least 4 GiB free VRAM before loading. The 4-bit setup is intended to fit the current 6-GB GPU, but this is a tight memory budget: browser/display use, context cache and generated tokens consume additional VRAM. Start with one question and a 512–1024 token cap; reduce `--max-new-tokens` to 512 if CUDA reports out-of-memory.
- Dataset: `sarvamai/gsm8k-indic`, config `hi`, split `test`.
- Sample: 200 Hindi questions, random seed 42.
- Fine-tuning: none.

The runner verifies CUDA availability, native BF16 support, GPU memory, presence of bitsandbytes `Linear4bit` layers, BF16 compute dtype on quantized layers, and absence of CPU/disk offload. It stops rather than silently running on CPU, FP16, or full-precision weights.

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
| E11 | `answer_trigger` | Exact Hindi question prompt; continue to answer only if Qwen naturally emits `</think>` |
| E12 | `answer_trigger_forced` | Separate control: if the 1,024-token thinking budget is exhausted, inject `</think>` and then request the answer |

### About natural and forced answer triggering

Qwen3's message format uses `<think> ... </think>` for reasoning, followed by a blank-line boundary (`\n\n`) and the normal assistant answer. There is no separate official `<answer>` special token.

E11 and E12 deliberately separate two behaviours:

**E11 — Natural answer trigger (`answer_trigger`)**

1. Send the Hindi question only and let Qwen generate its thinking.
2. Stop when Qwen itself emits the `</think>` token.
3. Continue the same assistant turn after the `\n\n` boundary to generate the final answer.
4. If the thinking token budget is exhausted before `</think>` appears, do not inject a closing token and do not run the answer phase. Save the incomplete thinking and mark `answer_generation_status=skipped_thinking_truncated`.

**E12 — Forced-closure control (`answer_trigger_forced`)**

1. Use the same Hindi-only prompt and the same thinking budget.
2. If Qwen naturally emits `</think>`, continue normally.
3. If the thinking budget is exhausted first, inject `</think>` and then continue after `\n\n` to request the answer. Mark `answer_trigger_injected=true` and `thinking_stop_reason=max_new_tokens_forced_close`.

This makes E11 a test of natural reasoning completion and E12 a separate forced-truncation control. Neither silently treats a forced closing marker as a natural model decision.

The CSV records `thinking_stop_reason`, `answer_generation_status`, `answer_trigger_injected`, `thinking_timing_scope`, and the separate thinking/answer generation durations. For one-call thinking methods, the thinking and answer phases cannot always be timed separately; when the final answer is produced in the same call, the phase-specific fields are blank and `thinking_timing_scope=single_call_combined_phases_not_separable`. If the one-call generation stops before the answer and the entire call is thinking, its elapsed time is saved as the thinking-phase duration.

## Install

Install CUDA-enabled PyTorch first. For the user's Windows x86-64 + Python 3.13 + NVIDIA driver 617.14 setup, the official PyTorch CUDA 13.0 index provides a matching `torch 2.14.1` wheel. In PowerShell, run:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Check the CUDA/BF16/bitsandbytes kernel before downloading the model:

```powershell
python check_gpu.py
```

Only proceed when it reports `NF4/BF16 test: PASS`. The latest bitsandbytes PyPI wheel supports Windows x86-64, Python 3.13 and NVIDIA CUDA GPUs. The GPU is an RTX 3050 (Ampere, compute capability 8.6); check BF16 support after installation. You do not need the full CUDA Toolkit just to use the prebuilt PyTorch wheel.

Verify CUDA and BF16 before loading the model:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not found'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else 0)"
```

Expected: CUDA available `True`, BF16 `True`, and approximately 6 GiB VRAM.

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
python run_experiments.py --max-questions 1 --techniques question_only,answer_trigger,answer_trigger_forced
```

The runner prints the selected GPU, total/free VRAM, loaded 4-bit layer count and BF16 compute setting. It stops if CUDA or BF16 is unavailable, if the GPU has less than 5.5 GiB total memory, if free VRAM is below 4 GiB, or if any model module is offloaded to CPU/disk.

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
python run_experiments.py --gpu-id 0 --max-new-tokens 1024 --answer-max-new-tokens 512 --min-free-vram-gib 4.0
```

If GPU applications use memory, close them before starting. The minimum free-memory guard can be adjusted deliberately with `--min-free-vram-gib`, but do not lower it unless you have checked the available memory.

## Resume an interrupted run

The runner appends one CSV row after each completed question/technique pair. If the process is interrupted, restart with the same input, output path, model, technique list and generation settings, and add `--resume`:

```powershell
python run_experiments.py --resume
```

For example, to resume a run that used a subset of techniques:

```powershell
python run_experiments.py --techniques question_only,answer_trigger,answer_trigger_forced --resume
```

With `--resume`, successful rows (`run_status=ok`) already present in the output CSV are skipped. Failed/unknown rows are retried. Without `--resume`, an existing output CSV is deleted at startup, so use `--resume` when continuing a previous run.

Resume is at the **question + technique** level, not inside a generation. If the program stops while a technique is still running (for example, after 3 of 5 Self-Consistency candidates), that entire question/technique is run again. Previously completed question/technique rows remain saved. Keep the same model and generation settings when resuming, and do not rename or delete the result CSV.

## Output files

- `results/qwen3_4b_nf4_bf16_gsm8k_hi_results_v2.csv`: one row per question per technique.
- `results/qwen3_4b_nf4_bf16_summary_by_technique_v2.csv`: accuracy, timing, token and numeral-format summaries.

The v2 output path is intentional: it keeps the earlier forced-closure implementation and 512-token pilot rows separate from the corrected natural-trigger experiment. Do not use the old v1 CSV as the v2 resume file.

Important detailed CSV fields:

- `question_hi`: the full original Hindi question.
- `prompt_text`: exactly what was passed as user content. For `question_only` and `answer_trigger`, it is only the Hindi question text.
- `raw_model_output`: full decoded generation, including Qwen markers where present.
- `thinking_text`: extracted model-generated thinking content.
- `final_response`: extracted final answer content.
- `predicted_numeral_system`: `Hindu-Arabic`, `Devanagari`, `Mixed`, `Hindi-word-based`, `Other-text`, or `Unparsed`.
- `extracted_answer_raw`: answer as generated, before numeral normalization.
- `predicted_answer_normalized`: canonical form used for answer matching.
- `model_dtype`, `weight_quantization`, `quantization_type`, `compute_dtype`, `bnb_double_quant`, `device`, `gpu_name`, `gpu_compute_capability`, `gpu_total_memory_gib`: record the actual execution configuration. The expected values are `4-bit`, `NF4`, and `bfloat16` compute.
- `answer_trigger_text`, `answer_trigger_injected`, `thinking_stop_reason`: staged answer-trigger metadata.
- `answer_generation_status`: whether the answer was generated in the same call, after a natural thinking close, after a forced close, or skipped because thinking was truncated.
- `thinking_timing_scope`: explains whether timing is an isolated thinking phase, answer-only call, or a combined one-call generation whose phases cannot be separated.
- `thinking_generation_time_seconds`, `answer_generation_time_seconds`: isolated phase durations for E11/E12. For one-call generations, phase timings are blank when they cannot be separated; an incomplete thinking-only call records its full duration as thinking time.
- `technique_inference_time_seconds`: complete per-question duration for the technique, including all generation calls and aggregation. Model loading, dataset loading and CSV writes are excluded.
- `total_technique_inference_time_seconds`: accumulated end-to-end technique time across the questions in the summary.

The numeral normalizer converts Devanagari digits (०१२३४५६७८९) to ASCII digits (0123456789) for correctness checking, while preserving the original answer and detected numeral system in separate columns.

## Reproducibility

- Dataset sample seed: 42
- Inference seed: 2026
- Model: `Qwen/Qwen3-4B`
- Weight quantization: 4-bit NF4 with double quantization
- Compute dtype: BF16
- Thinking mode temperature / top-p / top-k: 0.6 / 0.95 / 20
- Non-thinking mode temperature / top-p / top-k: 0.7 / 0.8 / 20
- Thinking token limit: 1024
- Staged final-answer token limit: 512
- Answer-trigger comparison: E11 natural close vs E12 forced closure at the token limit

See the official [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B) and [Qwen3 thinking format documentation](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md).
