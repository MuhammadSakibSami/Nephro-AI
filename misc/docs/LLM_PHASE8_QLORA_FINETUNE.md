# LLM Phase 8 — QLoRA Fine-Tuning

**Status:** ✅ Pipeline built, validated offline, and **trained for real** on 2026-07-27 on
the RTX 3090 GPU box (adapter `adapters/v1`, final val loss 0.1546 — see §7). Evaluated in
Phase 9 and in the Q1 six-arm ablation (`results/q1_llm_evaluation_summary.md`); deployed.
**Plan reference:** `LLM_PLAN.md` → Phase 8
**Date implemented:** July 2026
**Depends on:** Phase 7 (`train.jsonl` / `val.jsonl`)

> **Precision note (added after the real run):** the actual training run used
> `--precision bf16` — a plain **LoRA** adapter on an **unquantized bf16** frozen
> base, not the script's `4bit`/QLoRA default. The 24 GB training GPU had enough
> headroom that 4-bit NF4 quantization wasn't needed, and bf16 avoids the extra
> quantization error NF4 would add on a small, 540-example training set. This
> doc still describes the general QLoRA mechanism below (the tool supports
> both), but the artifact actually produced is a bf16-trained LoRA adapter.
>
> **As actually run (v1 adapter) — verified from `checkpoint-102/training_args.bin`:**
>
> | Item | Value |
> |---|---|
> | GPU | NVIDIA GeForce RTX 3090, **24 GB VRAM** (local Windows workstation, not Colab) |
> | PyTorch / transformers / PEFT | **2.5.1+cu121** / **5.14.1** / 0.19.1 |
> | Training precision | bf16 base, no quantization (`--precision bf16`, `bf16=True`, `optim=adamw_torch`); adapter weights kept in float32 |
> | Inference precision (serving) | **bfloat16** base weights, no quantization; adapter weights float32; 7.62 GiB VRAM |
> | max_seq_len | 3072 (overridden from the tier default) |
>
> Full record: `docs/HARDWARE_AND_ENVIRONMENT.md`.

---

## 1. What this phase is (in plain language)

Phase 7 produced examples of the form *"given this kidney-risk result and these
guideline snippets, here is the ideal safe answer."* Phase 8 trains Qwen3-4B on
them so the model produces that answer **by itself** — from a **short prompt**,
with **no safety validator** patching it afterwards.

We do **not** retrain the whole 4B model. We freeze it and train small LoRA
adapter matrices on top (~0.5–1% of parameters) — that's what makes this fit on
a modest GPU regardless of precision. "QLoRA" specifically means doing this
with the frozen base compressed to 4-bit; our actual run kept the frozen base
in bf16 instead (no quantization — see the note above), because our training
GPU had enough memory that we didn't need the extra compression.

---

## 2. Why this shape (the reframe)

From Phase 6: base + RAG + guardrail is already near-ceiling on quality. The only
gap was ~10% of outputs needing the validator. So fine-tuning is **not** a quality
overhaul. Three measurable targets:

1. **Safety self-containment** — internalize the rules so `raw_unsafe` → ~0
   *without* the validator (Phase-6 base: 0.10).
2. **Consistency** — remove rare format misses and over-specific numbers.
3. **Prompt compression** — trained on the COMPACT prompt (~800 chars vs ~1800
   full) → cheaper inference.

Report honestly if raw-quality deltas are small; that is the expected outcome.

---

## 3. What was built (technical)

```
llm/finetune/
  __init__.py
  data.py                # chat-template encoding + assistant-only loss masking
  qlora_train.py         # LoRA SFT (4-bit NF4 or bf16 base via --precision); VRAM auto-detect; curves + run card
  evaluate_adapter.py    # acceptance check: raw_unsafe with the validator OFF
  adapters/              # trained adapters (gitignored)
misc/notebooks/
  phase8_qlora_finetune.ipynb   # Colab mirror of the CLI flow
```

### 3.1 Loss masking (`data.py`) — the correctness-critical part

Only the assistant turn should contribute loss; the prompt must not. The masking
is tokenizer-version independent:

```python
prompt_text = chat_template(messages[:-1], add_generation_prompt=True)
full_text   = chat_template(messages)          # includes assistant + <|im_end|>
labels      = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

`prompt_text` is always a literal prefix of `full_text` for the Qwen3 template,
so masking the first `len(prompt_ids)` tokens supervises exactly the answer —
**including its end-of-turn token**, so the model learns to stop. A
`RuntimeError` is raised if a future template breaks prefix-stability, rather
than silently training on a mis-aligned target.

Examples whose prompt alone fills the window (zero supervised tokens) are
**dropped**; truncated examples are **counted and warned about**.

### 3.2 Training (`qlora_train.py`)

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen3-4B-Instruct-2507` |
| Quantization | `--precision 4bit` (default): 4-bit **NF4**, double-quant, bf16 compute. `--precision bf16` (**what the real run used**): unquantized bf16 base, plain LoRA — no `bitsandbytes` quantization at all |
| LoRA | r=16, alpha=32, dropout 0.05 |
| Targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Schedule | 3 epochs, lr 2e-4, cosine, warmup 3% |
| Selection | best checkpoint by **val loss** (`load_best_model_at_end`) |
| Memory | gradient checkpointing + paged AdamW 8-bit (4-bit mode); **v1 run: gradient checkpointing + `adamw_torch`** |

**VRAM auto-detect** picks batch / accumulation / seq-len; every value is
overridable:

| Tier | VRAM | batch × accum | max_seq_len |
|---|---|---|---|
| large | ≥ 32 GB | 4 × 4 | 2048 |
| medium | 16–24 GB | 2 × 8 | 2048 |
| small | ≤ 12 GB | 1 × 16 | 1792 |

Phase-7 examples measured ~1400–1650 tokens (compact prompt + 4 golden + 2
distractor chunks + answer), so 2048 clears the p95 with headroom.

**Deviation from the plan, deliberate:** the plan names TRL's `SFTTrainer`; we
use `transformers.Trainer` over a pre-tokenized dataset. The training objective
is identical, but the assistant-only masking becomes explicit and unit-tested
instead of depending on TRL's shifting collator API. Worth one line in the paper.

### 3.3 Acceptance check (`evaluate_adapter.py`)

Generates on `val.jsonl` from the **COMPACT** prompt with the **validator OFF**,
then measures:

- **`raw_unsafe_rate`** — the headline (target ~0; Phase-6 base = 0.10)
- `format_ok_rate` — all 5 sections parse
- `mentions_risk_rate`, `mean_readability` (target ≤ 8)

`scan_unsafe` needs the patient's BP to know whether an emergency redirect was
required. Phase-7 rows don't store `patient_inputs`, so SBP/DBP are recovered
from the rendered PATIENT PROFILE block in the user turn (and `meta.patient_inputs`
is preferred if a future dataset provides it).

Run it **twice** — with and without `--adapter` — to get the base-vs-fine-tuned
pair for Phase 9.

---

## 4. How to run

### 4.1 Dry-run (no GPU, no model download) — validates the data pipeline
```bash
python -m llm.finetune.qlora_train --data-dir phase7_data --dry-run
```
Prints encoded counts, token-length percentiles, truncation count, and a
suggested `--max-seq-len`. **Confirm `truncated` is 0 before training.**

### 4.2 Train
```bash
python -m llm.finetune.qlora_train \
    --data-dir phase7_data \
    --out-dir llm/finetune/adapters/v1 \
    --epochs 3
```
Outputs in `--out-dir`: `adapter/` (+ tokenizer), `training_log.csv`,
`training_curve.png`, `run_card.md`.

### 4.3 Evaluate (both arms)
```bash
python -m llm.finetune.evaluate_adapter --data-dir phase7_data --tag base
python -m llm.finetune.evaluate_adapter --data-dir phase7_data \
    --adapter llm/finetune/adapters/v1/adapter --tag finetuned
```

### 4.4 Serve — no runtime changes
```python
from llm.serving.model_backends import QwenChatModel
model = QwenChatModel(adapter_path="llm/finetune/adapters/v1/adapter")
```

---

## 5. Offline validation result

Verified without a GPU (fake tokenizer + the 9-row Phase-7 smoke dataset):

| Check | Result |
|---|---|
| `mask_prompt_prefix` / `fit_to_length` pure functions | PASS |
| Assistant-only span starts at answer, ends at `<|im_end|>` | PASS |
| Prompt fully masked to `-100` | PASS |
| Prefix-stability guard raises on a broken template | PASS |
| Real Phase-7 rows encode; 0 dropped, 0 truncated @2048 | PASS (p50 1424, p95 1616 tokens; answers p50 163) |
| Zero-supervision examples dropped at a tiny window | PASS (9/9 dropped @16) |
| Empty `val.jsonl` tolerated (trains without eval + warns) | PASS |
| `evaluate_adapter` context rebuild on real rows | PASS (BP recovered 9/9; emergency row detected; all gold targets scan clean) |
| Padding collator shapes + pad masked to `-100` | SKIPPED (no torch in the validation env) |

(Validates plumbing and masking only — see §7 below for the real training run,
which has since completed with `--precision bf16` on a 24 GB GPU.)

> **Found during validation:** Phase 7 writes an *empty* `val.jsonl` on small runs
> because `int(n * val_frac)` rounds to 0 (e.g. n=9 → 0 val). That crashed the
> loader; it is now treated as "no validation set" with a warning. A real 600-row
> run yields ~60 val examples, so best-checkpoint selection works normally.

> **Token budget note:** the observed p95 of 1616 tokens is why the small-VRAM
> tier uses `max_seq_len=1792` rather than 1536 — 1536 would silently truncate
> answer tails. Always check `truncated == 0` in the dry-run.

---

## 6. Design decisions worth remembering

- **Mask by prompt-prefix length, not by string search.** No dependence on
  `response_template` matching or TRL version.
- **Supervise the end-of-turn token** so the fine-tuned model stops cleanly.
- **Best checkpoint on val loss**, not last — the base is already strong, so
  overfitting is the main risk.
- **Compact prompt at train time** = the prompt-compression objective; targets
  were generated under the full prompt in Phase 7.
- **Adapter, not merged weights.** Loads via the existing `adapter_path` hook;
  the base model is untouched and swappable. GGUF merge/export is Phase 10.
- **Evaluate with the validator OFF.** Leaving it on would hide the exact thing
  fine-tuning is supposed to fix.

---

## 7. Known open items

1. ~~Run the real training once Phase 7 finishes; commit `run_card.md` +
   `training_curve.png` for the paper.~~ **Done.** Trained with
   `--precision bf16` on a 24 GB GPU (`llm/finetune/adapters/v1/run_card.md`):
   3 epochs, val loss 0.1967 → 0.1611 → **0.1546**, 0 truncated examples.
2. **Record both arms** (base and fine-tuned) from `evaluate_adapter` — this is
   the Phase-9 Ablation-1 table. **Done** — see `misc/docs/LLM_PHASE9_ABLATION_EVAL.md`
   and the Q1 six-arm evaluation (`results/q1_llm_evaluation_summary.md`).
3. **Watch for overfitting** — if val loss rises after epoch 1–2, drop to
   `--epochs 2`; the best checkpoint is retained regardless.
4. **Self-distillation ceiling** — targets came largely from the base model, so
   expect internalization gains, not a quality jump. Consider the Gemini teacher
   subset if answers look homogeneous.
5. **Adapter versioning** — use `adapters/v1`, `v2`, … so Phase-9 runs stay
   traceable to a specific dataset + config.

---

## 8. Next steps

**Phase 9 — Evaluation & ablation:** fine-tuned vs the frozen Phase-6 base arm
(`results/phase6_*`) on the 30 held-out eval profiles — `raw_unsafe` ↓,
format/personalization deltas, prompt-length / inference-cost ↓. Plus RAG vs
no-RAG.

**Phase 10 — Serving:** `merge_and_export.py` (merge LoRA → base → GGUF →
Q4_K_M + Ollama `Modelfile`).

---

## 9. File map (quick reference)

```
llm/finetune/data.py                    # encoding + assistant-only masking (unit-tested)
llm/finetune/qlora_train.py             # python -m llm.finetune.qlora_train --help
llm/finetune/evaluate_adapter.py        # python -m llm.finetune.evaluate_adapter --help
llm/serving/model_backends.py           # QwenChatModel(adapter_path=...) load hook
misc/notebooks/phase8_qlora_finetune.ipynb
misc/docs/LLM_PHASE8_QLORA_FINETUNE.md  # this file
```
