# Phase 8 — QLoRA Fine-Tuning — Run Card

- Date: 2026-07-27
- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Precision: bf16 (bf16 LoRA)
- GPU tier: medium (24.0 GB)

## Hyperparameters

- LoRA: r=16, alpha=32, dropout=0.05
- Targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Epochs: 3.0 | lr: 0.0002 (cosine, warmup 3%)
- Batch: 2 x accum 8 = effective 16
- max_seq_len: 3072 | seed: 42
- Loss masked to the assistant turn only (compact system prompt).

## Data

- Train examples: 540 (truncated 0)
- Val examples: 60
- Train token length: {'min': 2030, 'p50': 2459, 'p95': 2822, 'max': 2973}
- Answer token length: {'min': 160, 'p50': 421, 'p95': 507, 'max': 587}

## Result

- Final train loss: 0.1325
- Best val loss: 0.1546
- Val loss trace: [0.1967, 0.1611, 0.1546]

## Environment (verified 2026-09-28 from `checkpoint-102/training_args.bin` and the machine)

- GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM (the "medium" tier above), Windows 10 Pro
- PyTorch 2.5.1+cu121 (CUDA 12.1) · transformers 5.14.1 · PEFT 0.19.1 · accelerate 1.14.0
- Training precision: **bf16 LoRA, no quantization** — `bf16=True`, `optim=adamw_torch`,
  gradient checkpointing on. The 4-bit NF4 (QLoRA) mode of `qlora_train.py` was **not**
  used for this adapter; `bitsandbytes` is not installed.
- Inference precision when served: base weights bfloat16, adapter weights float32,
  no quantization; 7.62 GiB VRAM allocated.
- Full record: `docs/HARDWARE_AND_ENVIRONMENT.md`

Load with the existing serving hook:
```python
from llm.serving.model_backends import QwenChatModel
model = QwenChatModel(adapter_path='<out_dir>/adapter')
```