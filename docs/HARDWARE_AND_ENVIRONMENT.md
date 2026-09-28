# Hardware & Software Environment

Canonical, verified record of the compute environment used for the Nephro-AI
language-model work. Every value below was read directly from the machine or
from a saved training artefact — nothing is estimated. Other documents link here.

**Verified on:** 2026-09-28

---

## Summary (the five most-asked values)

| Item | Value |
|---|---|
| **GPU model** | NVIDIA GeForce RTX 3090 |
| **VRAM** | 24 GB (24,576 MiB) |
| **PyTorch version** | 2.5.1+cu121 (CUDA 12.1, cuDNN 9.1.0) |
| **Inference precision** | bfloat16 base weights, no quantization; LoRA adapter weights float32 |
| **transformers version** | 5.14.1 |

These apply to **all three GPU workloads**: Phase-8 fine-tuning, the final Q1
six-arm evaluation, and the live `/generate-guidance` serving path. The
libraries were installed on 2026-07-23 (PEFT on 2026-07-26), before the
fine-tuning run on 2026-07-27, and have not been changed since.

---

## 1. Machine

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24,576 MiB, bf16-capable |
| NVIDIA driver | 560.94 |
| CUDA (PyTorch build) | 12.1 |
| cuDNN | 9.1.0 (90100) |
| CPU | 12th Gen Intel Core i9-12900 |
| System RAM | 31.8 GB |
| OS | Windows 10 Pro (10.0.19045) |
| Python | 3.10.11 |

## 2. Software versions (project `.venv`)

| Package | Version | Role |
|---|---|---|
| torch | 2.5.1+cu121 | Tensor runtime |
| transformers | 5.14.1 | Qwen3 loading and generation |
| peft | 0.19.1 | LoRA adapter training and loading |
| accelerate | 1.14.0 | `device_map="auto"` placement |
| sentence-transformers | 5.6.0 | Installed; **not** used for the served embedder (see note) |
| faiss | 1.14.3 | Vector search over the knowledge-base index |
| bitsandbytes | **not installed** | 4-bit quantization was not used anywhere |

> **Embedder note.** torch 2.5.1 is below the 2.6 floor required by a
> `torch.load` security restriction (CVE-2025-32434), so the
> `sentence-transformers` BGE-M3 path cannot load. BGE-M3 is therefore served
> through **Ollama** (`ollama-bge-m3`, 1024-dim). Upgrading torch was avoided
> deliberately to keep the verified Qwen3 + LoRA stack stable.

## 3. Inference (serving and evaluation)

Measured by loading the model exactly as the backend does
(`QwenChatModel(adapter_path="llm/finetune/adapters/v1/adapter")`):

| Property | Measured value |
|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 + LoRA `v1/adapter` |
| Load call | `torch_dtype="auto"`, `device_map="auto"` → resolves to the checkpoint's native `bfloat16` |
| Base weight dtype | **bfloat16** (all 398 base tensors) |
| Adapter weight dtype | float32 (all 504 LoRA tensors — PEFT's default on load) |
| Quantization | **None** (`is_quantized = False`) |
| Device placement | Entirely on `cuda:0` (no CPU offload) |
| Base parameters | 4.022 B |
| LoRA parameters | 33.03 M (0.81% of total) |
| VRAM after load | 7.62 GiB allocated / 7.74 GiB reserved |
| Model load time | ≈ 29 s |

**Decoding settings differ by use:**

| Use | Decoding | Parameters |
|---|---|---|
| Q1 evaluation (all arms) | **Greedy / deterministic** (`do_sample=False`) | seed 42 |
| Live serving (`/generate-guidance`) | **Greedy** (backend sets temperature 0.0, matching Arm F) | k = 4 retrieved excerpts |
| Both | — | `max_new_tokens = 700` |

Observed end-to-end latency on this machine (Q1, per request, including
retrieval and validation): 22–31 s for the full pipeline; the deployed
configuration (Arm F) averaged 30.9 s.

## 4. Fine-tuning (Phase 8)

Verified from `llm/finetune/adapters/v1/run_card.md` and the saved
`checkpoint-102/training_args.bin`:

| Property | Value |
|---|---|
| Hardware | The same RTX 3090 (24 GB), Windows — the trainer state stores the checkpoint path with Windows separators (`llm\finetune\adapters\v1\...`), and the VRAM auto-detect recorded the 24 GB "medium" tier |
| Date | 2026-07-27 |
| **Training precision** | **bf16 LoRA — no quantization** (run with `--precision bf16`) |
| Mixed precision | `bf16 = True`, `fp16 = False` |
| Optimizer | `adamw_torch` (the 4-bit path would have used `paged_adamw_8bit`) |
| Gradient checkpointing | On |
| LoRA | r = 16, α = 32, dropout 0.05, on `q/k/v/o/gate/up/down_proj` |
| Batch | 2 per device × 8 accumulation = 16 effective |
| Schedule | 3 epochs, lr 2×10⁻⁴, cosine, 3% warmup, seed 42 |
| Max sequence length | 3,072 |
| Steps | 102 |
| Final train loss / best val loss | 0.1325 / 0.1546 |

> **Terminology correction.** The phase, script (`qlora_train.py`) and several
> document titles use the name *QLoRA* because the script's default mode is
> 4-bit NF4. The adapter actually reported and deployed was trained with
> `--precision bf16`, i.e. **standard (non-quantized) LoRA**. The 24 GB card had
> enough headroom to hold the full bf16 base, which also avoids adding 4-bit
> quantization error to a small-data run. Describe the method as *LoRA
> fine-tuning in bf16* in any paper or thesis text.

## 5. Other components (not GPU-bound)

| Component | Environment |
|---|---|
| Model C (ML cascade) training | Kaggle, Python 3.12.13, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, imbalanced-learn 0.14.1 (CPU) |
| Model C serving | Python 3.10.11, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, imbalanced-learn 0.13.0, joblib 1.5.3 — scikit-learn matches training exactly; the imbalanced-learn difference is immaterial because resampling runs only at training time |
| Embeddings | BGE-M3 via Ollama (local REST), 1024-dim |
| Database | PostgreSQL 18.6, local |
| Phase 6 (30-profile MVP evaluation) | Google Colab GPU — GPU model not recorded in the saved artefacts |

## 6. How to re-verify

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch,transformers,peft;print(torch.__version__,torch.version.cuda,transformers.__version__,peft.__version__)"
```

Precision of the served model: load `QwenChatModel(adapter_path=...)` and
inspect `{p.dtype for p in model.model.parameters()}`; the live health endpoint
`GET /generate-guidance/health` confirms the adapter is attached
(`model_spec: qwen:Qwen/Qwen3-4B-Instruct-2507+lora:v1/adapter`).
