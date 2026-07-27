"""LLM Component — Phase 9 Ablation (LLM_PLAN.md Phase 9).

Four arms over the same fresh, real-model-scored held-out profiles
(`phase9_profiles.py` — NOT the hand-authored Phase 6 set):

  1. base          - base Qwen3, FULL prompt,   NO retrieval, NO validator
  2. base_rag      - base Qwen3, FULL prompt,   RAG on,       NO validator
  3. base_rag_val  - base Qwen3, FULL prompt,   RAG on,       validator on   (Phase-6-equivalent, re-run fresh)
  4. finetuned     - fine-tuned adapter, COMPACT prompt, RAG on, validator on

Each arm is scored with `llm/eval/score.py`'s metrics (format, raw/final
unsafe, factor relevance, grounding, readability, personalization) plus a
prompt-token-count field for the inference-cost comparison (base uses the
full prompt; the fine-tuned arm uses the compact one — this is the deployed
"prompt compression" claim, not an apples-to-apples prompt-length test).

Run one arm at a time — keeps GPU memory to a single loaded model — then
combine once all 4 have run:

    python -m llm.eval.run_phase9 --arm base
    python -m llm.eval.run_phase9 --arm base_rag
    python -m llm.eval.run_phase9 --arm base_rag_val
    python -m llm.eval.run_phase9 --arm finetuned --adapter llm/finetune/adapters/v1/adapter
    python -m llm.eval.run_phase9 --combine
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..rag.retriever import Retriever
from ..serving.contract import LLMInput, parse_guidance
from ..serving.model_backends import get_model
from ..serving.prompts import build_compact_messages, build_messages
from ..serving.safety_validator import context_from_llm_input, enforce, validate
from .phase9_profiles import PHASE9_PROFILES_PATH, load_phase9_profiles
from .score import aggregate, score_record

ARMS = ["base", "base_rag", "base_rag_val", "finetuned"]

_ARM_CONFIG: Dict[str, Dict[str, Any]] = {
    "base":         dict(use_rag=False, use_validator=False, prompt="full"),
    "base_rag":     dict(use_rag=True,  use_validator=False, prompt="full"),
    "base_rag_val": dict(use_rag=True,  use_validator=True,  prompt="full"),
    "finetuned":    dict(use_rag=True,  use_validator=True,  prompt="compact"),
}

_RETRY_NOTE = ("\n\n(Reminder: reply using the EXACT output format with all five "
               "labeled sections. Do not diagnose, name or change any medication, "
               "or give numeric nutrient targets.)")

CSV_FIELDS = [
    "arm", "profile_id", "risk_class", "status", "n_attempts", "prompt_tokens",
    "format_ok", "raw_unsafe", "final_unsafe",
    "factor_relevance", "grounding_has_any", "grounding_valid_ratio",
    "readability_grade", "readability_ok", "personalization", "mentions_risk",
    "query", "raw_model_text", "final_safe_text",
]


@dataclass
class ArmOutput:
    status: str
    safe_text: str
    raw_model_text: str
    n_attempts: int
    model_cited_grounding: List[str] = field(default_factory=list)
    retrieved_chunk_ids: List[str] = field(default_factory=list)


def _prompt_token_count(model, messages: List[Dict[str, str]]) -> Optional[int]:
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return None
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tok(text)["input_ids"])
    except Exception:
        return None


def _prep_item(profile: Dict[str, Any], retriever: Retriever, model, k: int,
               use_rag: bool, prompt: str) -> Dict[str, Any]:
    inp = LLMInput.from_predict_response(profile)
    chunks = retriever.retrieve_for_llm_input(inp, k=k) if use_rag else []
    retrieved_ids = [c["chunk_id"] for c in chunks]
    build: Callable = build_compact_messages if prompt == "compact" else build_messages
    query = retriever.build_query_from_llm_input(inp) if use_rag else ""
    messages = build(inp, chunks)
    return dict(profile=profile, inp=inp, chunks=chunks, retrieved_ids=retrieved_ids,
                query=query, messages=messages, build=build,
                prompt_tokens=_prompt_token_count(model, messages),
                ctx=context_from_llm_input(inp))


def _arm_output_no_validator(raw: str, retrieved_ids: List[str]) -> ArmOutput:
    parsed = parse_guidance(raw)
    valid_grounding = [g for g in parsed.grounding if g in set(retrieved_ids)]
    # No validator ran: "final" IS the raw output (nothing was sanitized).
    return ArmOutput(status="unvalidated", safe_text=raw, raw_model_text=raw,
                      n_attempts=1, model_cited_grounding=valid_grounding,
                      retrieved_chunk_ids=retrieved_ids)


def _arm_output_validated(item: Dict[str, Any], raw: str, model, max_retries: int) -> ArmOutput:
    retrieved_ids = item["retrieved_ids"]
    vres = validate(raw, item["ctx"])
    if vres.recommendation == "accept":
        parsed = vres.parsed or parse_guidance(vres.safe_text)
        valid_grounding = [g for g in parsed.grounding if g in set(retrieved_ids)]
        return ArmOutput(status=vres.status, safe_text=vres.safe_text, raw_model_text=raw,
                          n_attempts=1, model_cited_grounding=valid_grounding,
                          retrieved_chunk_ids=retrieved_ids)

    # Rare path (severe/broken first attempt): fall back to the sequential
    # generate-validate-retry loop for just this profile.
    build = item["build"]
    captured = {"raw": "", "attempts": 0}

    def _generate(attempt: int) -> str:
        captured["attempts"] += 1
        msgs = build(item["inp"], item["chunks"])
        if attempt > 0:
            msgs[-1]["content"] += _RETRY_NOTE
        r = model.generate(msgs)
        captured["raw"] = r
        return r

    result = enforce(_generate, item["ctx"], max_retries=max_retries)
    parsed = result.parsed or parse_guidance(result.safe_text)
    valid_grounding = [g for g in parsed.grounding if g in set(retrieved_ids)]
    return ArmOutput(status=result.status, safe_text=result.safe_text,
                      raw_model_text=captured["raw"], n_attempts=captured["attempts"],
                      model_cited_grounding=valid_grounding, retrieved_chunk_ids=retrieved_ids)


def run(arm: str, profiles_path: Path = PHASE9_PROFILES_PATH,
        adapter: Optional[str] = None, model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        model_spec: str = "qwen", k: int = 4, max_retries: int = 1, n: Optional[int] = None,
        batch_size: int = 6, out_dir: str | Path = "phase9_eval") -> Dict[str, Any]:
    if arm not in _ARM_CONFIG:
        raise ValueError(f"Unknown arm {arm!r}; choose from {ARMS}")
    cfg = _ARM_CONFIG[arm]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_phase9_profiles(profiles_path)
    if n:
        profiles = profiles[:n]
    retriever = Retriever.load()
    model_kwargs = {"model_name": model_name} if model_spec == "qwen" else {}
    if arm == "finetuned" and model_spec == "qwen":
        model_kwargs["adapter_path"] = adapter
    model = get_model(model_spec, **model_kwargs)
    print(f"Arm: {arm} | model: {model.spec} | adapter: {adapter or 'none'} | "
          f"prompt: {cfg['prompt']} | RAG: {cfg['use_rag']} | validator: {cfg['use_validator']} | "
          f"profiles: {len(profiles)} | batch_size: {batch_size}\n")

    items = [_prep_item(p, retriever, model, k, cfg["use_rag"], cfg["prompt"]) for p in profiles]

    # Batched first-attempt generation — the GPU-bound step, so batching is
    # what actually cuts wall-clock (vs. one generate() call per profile).
    can_batch = hasattr(model, "generate_batch")
    raw_texts: List[str] = []
    if can_batch:
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            raw_texts.extend(model.generate_batch([it["messages"] for it in chunk]))
            print(f"  generated {min(i + batch_size, len(items))}/{len(items)}")
    else:
        for i, it in enumerate(items, 1):
            raw_texts.append(model.generate(it["messages"]))
            print(f"  generated {i}/{len(items)}")

    records: List[Dict[str, Any]] = []
    for i, (item, raw) in enumerate(zip(items, raw_texts), 1):
        profile = item["profile"]
        result = (_arm_output_validated(item, raw, model, max_retries) if cfg["use_validator"]
                   else _arm_output_no_validator(raw, item["retrieved_ids"]))
        scores = score_record(profile, result, item["ctx"])
        records.append({
            "arm": arm,
            "profile_id": profile["profile_id"],
            "risk_class": profile["risk_class"],
            "prompt_tokens": item["prompt_tokens"],
            "query": item["query"],
            "raw_model_text": result.raw_model_text,
            "final_safe_text": result.safe_text,
            **scores,
        })
        print(f"[{i:2}/{len(items)}] {profile['profile_id']} ({profile['risk_class']}) "
              f"status={result.status} fmt={scores['format_ok']} "
              f"raw_unsafe={scores['raw_unsafe']} final_unsafe={scores['final_unsafe']} "
              f"tokens={item['prompt_tokens']}")

    summary = aggregate(records)
    summary["arm"] = arm
    summary["model"] = model.spec
    summary["adapter"] = adapter or ""
    summary["prompt_variant"] = cfg["prompt"]
    summary["rag"] = cfg["use_rag"]
    summary["validator"] = cfg["use_validator"]
    prompt_tok_vals = [r["prompt_tokens"] for r in records if r["prompt_tokens"] is not None]
    summary["mean_prompt_tokens"] = round(sum(prompt_tok_vals) / len(prompt_tok_vals), 1) if prompt_tok_vals else None

    _write_csv(records, out_dir / f"phase9_{arm}_results.csv")
    (out_dir / f"phase9_{arm}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    examples = _pick_examples(records)
    _write_summary(out_dir / f"phase9_{arm}_summary.md", model, cfg, summary, examples)

    print("\n" + "=" * 60)
    print(f"AGGREGATE METRICS — arm={arm}")
    print("=" * 60)
    for kx, vx in summary.items():
        print(f"  {kx}: {vx}")
    print(f"\nOutputs -> {out_dir}/phase9_{arm}_*.csv|json|md")
    return {"summary": summary, "records": records}


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def _pick_examples(records: List[Dict[str, Any]]) -> Dict[str, Optional[Dict]]:
    """Good / validator-fixed / bad examples, deduplicated against each other
    (Phase 6's picker let the same record satisfy more than one bucket)."""
    used: set = set()

    def _first(pred) -> Optional[Dict]:
        for r in records:
            if r["profile_id"] in used:
                continue
            if pred(r):
                used.add(r["profile_id"])
                return r
        return None

    good = _first(lambda r: r["status"] in ("pass", "unvalidated") and not r["final_unsafe"]
                  and r["personalization"] == 2 and r["format_ok"]
                  and (r["factor_relevance"] in (None, 1.0)) and r["readability_ok"])
    fixed = _first(lambda r: r["status"] == "sanitized" or (r["raw_unsafe"] and not r["final_unsafe"]))
    bad = _first(lambda r: r["status"] == "fallback" or not r["format_ok"]
                 or r["raw_unsafe"] or r["personalization"] == 0 or not r["readability_ok"])
    return {"good": good, "validator_fixed": fixed, "bad": bad}


def _ex_block(title: str, r: Optional[Dict]) -> str:
    if not r:
        return f"### {title}\n_None in this run._\n"
    return (
        f"### {title} — `{r['profile_id']}` ({r['risk_class']}, status={r['status']})\n"
        f"- scores: personalization={r['personalization']}, "
        f"factor_relevance={r['factor_relevance']}, readability_grade={r['readability_grade']}, "
        f"format_ok={r['format_ok']}, raw_unsafe={r['raw_unsafe']}, final_unsafe={r['final_unsafe']}, "
        f"prompt_tokens={r['prompt_tokens']}\n\n"
        f"**Raw model output:**\n```\n{(r['raw_model_text'] or '').strip()[:900]}\n```\n\n"
        f"**Final (post-validator) output:**\n```\n{(r['final_safe_text'] or '').strip()[:900]}\n```\n"
    )


def _write_summary(path: Path, model, cfg: Dict[str, Any], summary: Dict[str, Any],
                    examples: Dict[str, Optional[Dict]]) -> None:
    is_mock = "mock" in str(model.spec).lower()
    lines = [
        f"# Phase 9 — Arm evaluation ({summary['arm']})", "",
        f"- Date: {datetime.now(timezone.utc).date().isoformat()}",
        f"- Model: `{model.spec}` | Adapter: `{summary['adapter'] or 'none'}`",
        f"- Prompt: **{cfg['prompt']}** | RAG: **{cfg['use_rag']}** | Validator: **{cfg['use_validator']}**",
        f"- Profiles: {summary['n']}",
    ]
    if is_mock:
        lines.append("\n> NOTE: this run used the offline `mock` model — numbers only validate the harness.")
    lines += ["", "## Aggregate metrics", "", "| Metric | Value |", "|---|---|"]
    skip = {"arm", "model", "adapter", "prompt_variant", "rag", "validator"}
    for kx, vx in summary.items():
        if kx not in skip:
            lines.append(f"| {kx} | {vx} |")
    lines += ["", "## Examples", "",
              _ex_block("Good output", examples["good"]),
              _ex_block("Validator-fixed output", examples["validator_fixed"]),
              _ex_block("Bad / weak output", examples["bad"])]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Combine: read the 4 per-arm summaries (no model loading) into one table.
# ---------------------------------------------------------------------------
_COMPARE_METRICS = [
    "format_adherence_rate", "raw_unsafe_rate", "final_unsafe_rate",
    "mean_factor_relevance", "grounding_any_rate", "mean_readability_grade",
    "readability_ok_rate", "mean_personalization", "mentions_risk_rate",
    "mean_prompt_tokens",
]


def combine(out_dir: str | Path = "phase9_eval") -> Dict[str, Any]:
    out_dir = Path(out_dir)
    summaries: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        p = out_dir / f"phase9_{arm}_summary.json"
        if p.exists():
            summaries[arm] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"[combine] missing {p} — run `--arm {arm}` first")

    lines = ["# Phase 9 — Ablation Comparison", "",
             "| Metric | " + " | ".join(ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]
    for metric in _COMPARE_METRICS:
        row = [str(summaries.get(a, {}).get(metric, "—")) for a in ARMS]
        lines.append(f"| {metric} | " + " | ".join(row) + " |")
    (out_dir / "phase9_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nOutputs -> {out_dir}/phase9_comparison.md")
    return summaries


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 9 ablation harness")
    p.add_argument("--arm", choices=ARMS, help="which arm to run")
    p.add_argument("--combine", action="store_true", help="build the comparison table from existing summaries")
    p.add_argument("--profiles", default=str(PHASE9_PROFILES_PATH))
    p.add_argument("--adapter", default=None, help="LoRA adapter path (finetuned arm only)")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--model-spec", default="qwen", help="qwen | mock (mock = offline harness check)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n", type=int, default=None, help="limit profiles (debug)")
    p.add_argument("--batch-size", type=int, default=6, help="profiles per generate_batch() call")
    p.add_argument("--out-dir", default="phase9_eval")
    a = p.parse_args(argv)

    if a.combine:
        combine(out_dir=a.out_dir)
        return 0
    if not a.arm:
        p.error("pass --arm <name> or --combine")
    run(arm=a.arm, profiles_path=Path(a.profiles), adapter=a.adapter, model_name=a.model,
        model_spec=a.model_spec, k=a.k, n=a.n, batch_size=a.batch_size, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
