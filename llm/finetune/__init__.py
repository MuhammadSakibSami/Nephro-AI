"""LLM Component — QLoRA fine-tuning (LLM_PLAN.md Phase 8).

Modules:
  * `data`              — load Phase-7 JSONL, apply the Qwen3 chat template,
                          mask loss to the assistant turn, report length stats.
  * `qlora_train`       — 4-bit NF4 + LoRA SFT on `train.jsonl` / `val.jsonl`.
  * `evaluate_adapter`  — Phase-8 acceptance check: generate from the COMPACT
                          prompt with NO validator and measure raw_unsafe.

Nothing here is imported by the serving path. The trained adapter is consumed by
the existing hook: `QwenChatModel(adapter_path=...)`.
"""
