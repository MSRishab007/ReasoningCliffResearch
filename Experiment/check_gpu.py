"""Verify CUDA, native BF16, and bitsandbytes NF4/BF16 before loading Qwen."""
from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False.")
        print("Install a CUDA-enabled PyTorch wheel and check the NVIDIA driver.")
        return 1

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    capability = torch.cuda.get_device_capability(device)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"VRAM: {total_bytes / 1024**3:.2f} GiB total; {free_bytes / 1024**3:.2f} GiB free")
    print(f"Native BF16 supported: {torch.cuda.is_bf16_supported()}")

    if not torch.cuda.is_bf16_supported():
        print("FAIL: this GPU/PyTorch combination does not report BF16 support.")
        return 1

    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
    except Exception as exc:
        print(f"FAIL: bitsandbytes could not be imported: {type(exc).__name__}: {exc}")
        return 1

    print(f"bitsandbytes: {getattr(bnb, '__version__', 'unknown')}")
    try:
        layer = Linear4bit(
            64,
            32,
            bias=True,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
        ).to(device)
        x = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            y = layer(x)
        torch.cuda.synchronize(device)
        if y.device.type != "cuda" or y.dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected test output device/dtype: {y.device}/{y.dtype}")
        print(f"NF4/BF16 test: PASS (output shape={tuple(y.shape)}, dtype={y.dtype})")
    except Exception as exc:
        print(f"FAIL: bitsandbytes NF4/BF16 CUDA smoke test failed: {type(exc).__name__}: {exc}")
        return 1

    if free_bytes / 1024**3 < 4.0:
        print("WARNING: less than 4 GiB VRAM is free. Close GPU-heavy applications before loading Qwen3-4B.")
    print("GPU environment checks passed. You can try the one-question Qwen smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
