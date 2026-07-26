"""LLM Component — QLoRA Fine-Tuning (LLM_PLAN.md Phase 8).

Fine-tunes Qwen3-4B-Instruct-2507 on the Phase-7 RAFT dataset so it emits safe,
grounded, correctly-structured guidance **directly from the COMPACT prompt** —
internalizing the format + safety rules the validator currently patches.

Setup (per the plan):
  * 4-bit NF4 (double-quant) base + gradient checkpointing
  * LoRA r=16, alpha=32, dropout 0.05 on
    q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  * loss masked to the assistant turn only (see `data.py`)
  * 3 epochs, lr 2e-4, cosine schedule, grad-accumulation
  * best checkpoint selected on val loss

Batch / seq-len defaults are chosen from detected GPU VRAM; every value is
overridable by flag.

Usage
-----
Validate the data pipeline with no GPU and no model download:
    python -m llm.finetune.qlora_train --data-dir phase7_data --dry-run

Train:
    python -m llm.finetune.qlora_train --data-dir phase7_data \
        --out-dir llm/finetune/adapters/v1

Outputs (in --out-dir): the LoRA adapter (+ tokenizer), `training_log.csv`,
`training_curve.png` (if matplotlib), and `run_card.md`.

Note on the trainer: the plan names TRL's `SFTTrainer`. We use
`transformers.Trainer` over a pre-tokenized dataset instead, because the
assistant-only masking is then explicit and unit-tested (`data.mask_prompt_prefix`)
rather than dependent on TRL's shifting collator API. The training objective is
identical.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data import (
    ListDataset,
    PaddedCollator,
    encode_dataset,
    load_examples,
    suggest_max_seq_len,
)

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# VRAM tier -> (per_device_batch, grad_accum, max_seq_len)
# Phase-7 examples run ~1400-1650 tokens (compact prompt + 4 golden + 2 distractor
# chunks + answer), so 2048 fits the p95 with headroom. Run --dry-run to confirm
# truncation is 0 for your actual dataset.
_TIERS = {
    "large":  (4, 4, 2048),    # >= 32 GB
    "medium": (2, 8, 2048),    # 16-24 GB
    "small":  (1, 16, 1792),   # <= 12 GB
    "cpu":    (1, 16, 1024),   # no CUDA (dry-run / debugging only)
}


def detect_tier() -> tuple[str, float]:
    """Pick a defaults tier from the visible GPU's VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", 0.0
        gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return "cpu", 0.0
    if gb >= 30:
        return "large", gb
    if gb >= 15:
        return "medium", gb
    return "small", gb


# ---------------------------------------------------------------------------
# Version-tolerant transformers helpers
# ---------------------------------------------------------------------------
def _make_training_args(**kw):
    """Build TrainingArguments across the eval_strategy/evaluation_strategy rename."""
    from transformers import TrainingArguments
    try:
        return TrainingArguments(**kw)
    except TypeError:
        renamed = dict(kw)
        if "eval_strategy" in renamed:
            renamed["evaluation_strategy"] = renamed.pop("eval_strategy")
        return TrainingArguments(**renamed)


def _make_trainer(model, args, train_ds, eval_ds, collator, tokenizer):
    """Build Trainer across the tokenizer/processing_class rename."""
    from transformers import Trainer
    kw = dict(model=model, args=args, train_dataset=train_ds,
              eval_dataset=eval_ds, data_collator=collator)
    try:
        return Trainer(**kw, processing_class=tokenizer)
    except TypeError:
        return Trainer(**kw, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(data_dir: str | Path = "phase7_data",
          out_dir: str | Path = "llm/finetune/adapters/v1",
          model_name: str = DEFAULT_MODEL,
          epochs: float = 3.0, lr: float = 2e-4,
          batch: Optional[int] = None, accum: Optional[int] = None,
          max_seq_len: Optional[int] = None,
          lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
          precision: str = "4bit", seed: int = 42,
          train_file: Optional[str] = None, val_file: Optional[str] = None,
          dry_run: bool = False, auto_seq_len: bool = False) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    train_path = Path(train_file) if train_file else data_dir / "train.jsonl"
    val_path = Path(val_file) if val_file else data_dir / "val.jsonl"

    tier, vram = detect_tier()
    t_batch, t_accum, t_seq = _TIERS[tier]
    batch = batch or t_batch
    accum = accum or t_accum
    max_seq_len = max_seq_len or t_seq

    print(f"GPU tier: {tier}" + (f" ({vram:.1f} GB)" if vram else " (no CUDA detected)"))
    print(f"Defaults: batch={batch} accum={accum} max_seq_len={max_seq_len} "
          f"(effective batch {batch * accum})")

    train_rows = load_examples(train_path)
    # Phase 7 writes an EMPTY val.jsonl on small runs (int(n * 0.1) == 0), so an
    # empty file means "no validation set", not an error.
    val_rows = load_examples(val_path, allow_empty=True) if val_path.exists() else []
    print(f"Loaded train={len(train_rows)} val={len(val_rows)} from {data_dir}")
    if not val_rows:
        print("[warn] No validation examples - training without eval; best-checkpoint "
              "selection is disabled. Use a larger Phase-7 run for a real val split.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Encoding train:")
    train_enc, train_stats = encode_dataset(tokenizer, train_rows, max_seq_len)

    # Optionally re-encode at a length that fits the observed p95.
    if auto_seq_len:
        suggested = suggest_max_seq_len(train_stats)
        if suggested != max_seq_len:
            print(f"--auto-seq-len: re-encoding at max_seq_len={suggested}")
            max_seq_len = suggested
            train_enc, train_stats = encode_dataset(tokenizer, train_rows, max_seq_len)

    val_enc, val_stats = ([], {})
    if val_rows:
        print("Encoding val:")
        val_enc, val_stats = encode_dataset(tokenizer, val_rows, max_seq_len)

    if dry_run:
        print("\n[dry-run] Data pipeline OK. No model was loaded, nothing trained.")
        print(f"[dry-run] Suggested --max-seq-len for this data: "
              f"{suggest_max_seq_len(train_stats)}")
        return {"train_stats": train_stats, "val_stats": val_stats, "dry_run": True}

    # --- Model -------------------------------------------------------------
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    load_kw: Dict[str, Any] = {}
    if precision == "4bit":
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    print(f"Loading {model_name} ({precision}) ...")
    # `torch_dtype` is the long-standing name; transformers >=4.56 renames it to
    # `dtype`. Try the old name first (matches the pinned >=4.44 range), then new.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": 0}, torch_dtype=torch.bfloat16, **load_kw)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": 0}, dtype=torch.bfloat16, **load_kw)

    model.config.use_cache = False
    if precision == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    ))
    model.print_trainable_parameters()

    # --- Trainer -----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    has_val = bool(val_enc)
    args = _make_training_args(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=max(1, batch),
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch" if has_val else "no",
        save_total_limit=2,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if precision == "4bit" else "adamw_torch",
        report_to=[],
        seed=seed,
    )

    collator = PaddedCollator(tokenizer.pad_token_id)
    trainer = _make_trainer(model, args, ListDataset(train_enc),
                            ListDataset(val_enc) if has_val else None,
                            collator, tokenizer)

    print(f"\nTraining: {epochs} epochs, lr={lr}, effective batch "
          f"{batch * accum}, LoRA r={lora_r}/alpha={lora_alpha}\n")
    trainer.train()

    # --- Save --------------------------------------------------------------
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nAdapter saved -> {adapter_dir}")

    history = list(trainer.state.log_history)
    _write_curves(history, out_dir)
    config = {
        "model_name": model_name, "precision": precision, "epochs": epochs,
        "lr": lr, "batch": batch, "accum": accum, "effective_batch": batch * accum,
        "max_seq_len": max_seq_len, "lora_r": lora_r, "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout, "lora_targets": LORA_TARGETS,
        "seed": seed, "gpu_tier": tier, "vram_gb": round(vram, 1),
    }
    _write_run_card(out_dir / "run_card.md", config, train_stats, val_stats, history)
    return {"adapter_dir": str(adapter_dir), "history": history, "config": config}


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _write_curves(history: List[Dict[str, Any]], out_dir: Path) -> None:
    rows = [h for h in history if "loss" in h or "eval_loss" in h]
    csv_path = out_dir / "training_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "train_loss", "eval_loss"])
        for h in rows:
            w.writerow([h.get("step", ""), round(float(h.get("epoch", 0)), 4),
                        h.get("loss", ""), h.get("eval_loss", "")])
    print(f"Training log -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[info] matplotlib not installed; skipping training_curve.png")
        return

    tr = [(h["step"], h["loss"]) for h in rows if "loss" in h and "step" in h]
    ev = [(h["step"], h["eval_loss"]) for h in rows if "eval_loss" in h and "step" in h]
    if not tr and not ev:
        return
    plt.figure(figsize=(7, 4))
    if tr:
        plt.plot([p[0] for p in tr], [p[1] for p in tr], label="train loss")
    if ev:
        plt.plot([p[0] for p in ev], [p[1] for p in ev], marker="o", label="val loss")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.title("Phase 8 — QLoRA fine-tuning")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    png = out_dir / "training_curve.png"
    plt.savefig(png, dpi=150); plt.close()
    print(f"Training curve -> {png}")


def _write_run_card(path: Path, config: Dict[str, Any], train_stats: Dict[str, Any],
                    val_stats: Dict[str, Any], history: List[Dict[str, Any]]) -> None:
    losses = [h["loss"] for h in history if "loss" in h]
    evals = [h["eval_loss"] for h in history if "eval_loss" in h]
    lines = [
        "# Phase 8 — QLoRA Fine-Tuning — Run Card", "",
        f"- Date: {datetime.now(timezone.utc).date().isoformat()}",
        f"- Base model: `{config['model_name']}`",
        f"- Precision: {config['precision']} "
        f"({'NF4 + double-quant' if config['precision'] == '4bit' else 'bf16 LoRA'})",
        f"- GPU tier: {config['gpu_tier']} ({config['vram_gb']} GB)",
        "",
        "## Hyperparameters", "",
        f"- LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}, "
        f"dropout={config['lora_dropout']}",
        f"- Targets: `{', '.join(config['lora_targets'])}`",
        f"- Epochs: {config['epochs']} | lr: {config['lr']} (cosine, warmup 3%)",
        f"- Batch: {config['batch']} x accum {config['accum']} "
        f"= effective {config['effective_batch']}",
        f"- max_seq_len: {config['max_seq_len']} | seed: {config['seed']}",
        "- Loss masked to the assistant turn only (compact system prompt).",
        "",
        "## Data", "",
        f"- Train examples: {train_stats.get('n_examples')} "
        f"(truncated {train_stats.get('n_truncated')})",
        f"- Val examples: {val_stats.get('n_examples', 0)}",
        f"- Train token length: {train_stats.get('token_len')}",
        f"- Answer token length: {train_stats.get('answer_len')}",
        "",
        "## Result", "",
        f"- Final train loss: {round(losses[-1], 4) if losses else 'n/a'}",
        f"- Best val loss: {round(min(evals), 4) if evals else 'n/a'}",
        f"- Val loss trace: {[round(e, 4) for e in evals] if evals else 'n/a'}",
        "",
        "Load with the existing serving hook:",
        "```python",
        "from llm.serving.model_backends import QwenChatModel",
        "model = QwenChatModel(adapter_path='<out_dir>/adapter')",
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Run card -> {path}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3-4B (Phase 8)")
    p.add_argument("--data-dir", default="phase7_data",
                   help="directory holding train.jsonl / val.jsonl")
    p.add_argument("--train-file", default=None)
    p.add_argument("--val-file", default=None)
    p.add_argument("--out-dir", default="llm/finetune/adapters/v1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=None, help="default: from VRAM tier")
    p.add_argument("--accum", type=int, default=None, help="default: from VRAM tier")
    p.add_argument("--max-seq-len", type=int, default=None, help="default: from VRAM tier")
    p.add_argument("--auto-seq-len", action="store_true",
                   help="size max_seq_len from the data's p95 token length")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="encode + report stats only; no model load, no training")
    a = p.parse_args(argv)

    train(data_dir=a.data_dir, out_dir=a.out_dir, model_name=a.model,
          epochs=a.epochs, lr=a.lr, batch=a.batch, accum=a.accum,
          max_seq_len=a.max_seq_len, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
          lora_dropout=a.lora_dropout, precision=a.precision, seed=a.seed,
          train_file=a.train_file, val_file=a.val_file, dry_run=a.dry_run,
          auto_seq_len=a.auto_seq_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
