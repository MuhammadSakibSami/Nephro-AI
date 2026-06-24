"""LLM Component — Phase 6 MVP Evaluation harness (LLM_PLAN.md Phase 6).

Runs the base-Qwen3 MVP over the 30 held-out profiles, scores every output, and
exports `phase6_results.csv` + `phase6_summary.md` (with good/bad/validator-fixed
examples and a data-driven fine-tuning go/no-go).

Colab (production, real Qwen3):
    from ckd_risk_app.llm.eval.run_phase6 import run
    run(model="qwen", out_dir="phase6_outputs")

Offline (harness validation, mock model):
    python -m ckd_risk_app.llm.eval.run_phase6 --model mock
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.profiles import held_out_profiles
from ..serving.contract import LLMInput
from ..serving.llm_runtime import GuidanceRuntime
from ..serving.safety_validator import context_from_llm_input
from .score import aggregate, score_record

# Data-driven go/no-go thresholds: if a signal fails, fine-tuning should target it.
_GONOGO = [
    ("format_adherence_rate", "<", 0.95, "Output format drift — model omits/garbles required sections"),
    ("raw_unsafe_rate", ">", 0.05, "Unsafe content in raw generations (pre-validator)"),
    ("fallback_rate", ">", 0.05, "Too many severe outputs forcing safe-fallback"),
    ("retry_rate", ">", 0.10, "Frequent retries needed to get an acceptable answer"),
    ("mean_factor_relevance", "<", 0.70, "Weak personalization to the patient's driving factors"),
    ("mean_personalization", "<", 1.50, "Generic answers — risk level / factors not addressed"),
    ("readability_ok_rate", "<", 0.80, "Reading level above grade 8 too often"),
    ("grounding_any_rate", "<", 0.60, "Answers not grounded in retrieved guideline chunks"),
]

CSV_FIELDS = [
    "profile_id", "risk_class", "status", "n_attempts",
    "format_ok", "raw_unsafe", "final_unsafe",
    "factor_relevance", "grounding_has_any", "grounding_valid_ratio",
    "readability_grade", "readability_ok", "personalization", "mentions_risk",
    "provenance", "query", "raw_model_text", "final_safe_text",
]


def run(model: str = "auto", k: int = 4, out_dir: str | Path = "phase6_outputs",
        max_retries: int = 1) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = held_out_profiles()
    rt = GuidanceRuntime(model=model, k=k, max_retries=max_retries)
    print(f"Model: {rt.model.spec} | retriever: {rt.retriever.embedder.spec} | "
          f"profiles: {len(profiles)}\n")

    records: List[Dict[str, Any]] = []
    for i, profile in enumerate(profiles, 1):
        result = rt.generate_guidance(profile)
        ctx = context_from_llm_input(LLMInput.from_predict_response(profile))
        scores = score_record(profile, result, ctx)
        records.append({
            "profile_id": profile["profile_id"],
            "risk_class": profile["risk_class"],
            "provenance": "; ".join(result.provenance),
            "query": result.query,
            "raw_model_text": result.raw_model_text,
            "final_safe_text": result.safe_text,
            **scores,
        })
        print(f"[{i:2}/{len(profiles)}] {profile['profile_id']} ({profile['risk_class']}) "
              f"status={result.status} fmt={scores['format_ok']} "
              f"raw_unsafe={scores['raw_unsafe']} final_unsafe={scores['final_unsafe']} "
              f"pers={scores['personalization']} fk={scores['readability_grade']}")

    summary = aggregate(records)
    _write_csv(records, out_dir / "phase6_results.csv")
    examples = _pick_examples(records)
    decision = _go_no_go(summary)
    _write_summary(out_dir / "phase6_summary.md", rt, summary, records, examples, decision)

    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    for kx, vx in summary.items():
        print(f"  {kx}: {vx}")
    print(f"\nDECISION: {decision['recommendation']}")
    for w in decision["weaknesses"]:
        print(f"  - {w}")
    print(f"\nOutputs -> {out_dir}/phase6_results.csv, {out_dir}/phase6_summary.md")
    return {"summary": summary, "records": records, "decision": decision}


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def _pick_examples(records: List[Dict[str, Any]]) -> Dict[str, Optional[Dict]]:
    good = next((r for r in records if r["status"] == "pass" and not r["final_unsafe"]
                 and r["personalization"] == 2 and r["format_ok"]
                 and (r["factor_relevance"] in (None, 1.0)) and r["readability_ok"]), None)
    fixed = next((r for r in records if r["status"] == "sanitized"
                  or (r["raw_unsafe"] and not r["final_unsafe"])), None)
    bad = next((r for r in records if r["status"] == "fallback" or not r["format_ok"]
                or r["raw_unsafe"] or r["personalization"] == 0 or not r["readability_ok"]), None)
    return {"good": good, "validator_fixed": fixed, "bad": bad}


def _go_no_go(summary: Dict[str, Any]) -> Dict[str, Any]:
    weaknesses = []
    for key, op, thr, desc in _GONOGO:
        val = summary.get(key)
        if val is None:
            continue
        failed = (val < thr) if op == "<" else (val > thr)
        if failed:
            weaknesses.append(f"{desc}  ({key}={val}, target {op} {thr})")
    rec = ("GO — fine-tune Qwen3 (QLoRA). The weaknesses below define the training objective."
           if weaknesses else
           "NO-GO suggested — base+RAG+guardrail already meets thresholds; fine-tune only "
           "for marginal gains.")
    return {"recommendation": rec, "weaknesses": weaknesses}


def _ex_block(title: str, r: Optional[Dict]) -> str:
    if not r:
        return f"### {title}\n_None in this run._\n"
    return (
        f"### {title} — `{r['profile_id']}` ({r['risk_class']}, status={r['status']})\n"
        f"- scores: personalization={r['personalization']}, "
        f"factor_relevance={r['factor_relevance']}, readability_grade={r['readability_grade']}, "
        f"format_ok={r['format_ok']}, raw_unsafe={r['raw_unsafe']}, final_unsafe={r['final_unsafe']}\n\n"
        f"**Raw model output:**\n```\n{(r['raw_model_text'] or '').strip()[:900]}\n```\n\n"
        f"**Final (post-validator) output:**\n```\n{(r['final_safe_text'] or '').strip()[:900]}\n```\n"
    )


def _write_summary(path: Path, rt, summary, records, examples, decision) -> None:
    dist = Counter(r["risk_class"] for r in records)
    lines = [
        "# Phase 6 — MVP Evaluation Summary", "",
        f"- Date: {datetime.now(timezone.utc).date().isoformat()}",
        f"- Model: `{rt.model.spec}` | Retriever: `{rt.retriever.embedder.spec}` | k={rt.k}",
        f"- Profiles: {len(records)} (low={dist['low']}, moderate={dist['moderate']}, high={dist['high']})",
        "",
        "> NOTE: numbers from the offline `mock` model only validate the harness. The "
        "decision below is meaningful only for a real Qwen3 run.",
        "",
        "## Aggregate metrics", "",
        "| Metric | Value |", "|---|---|",
    ]
    for kx, vx in summary.items():
        lines.append(f"| {kx} | {vx} |")
    lines += ["", "## Fine-tuning decision", "", f"**{decision['recommendation']}**", ""]
    if decision["weaknesses"]:
        lines.append("Weaknesses fine-tuning should target:")
        lines += [f"- {w}" for w in decision["weaknesses"]]
    else:
        lines.append("No threshold failures detected.")
    lines += ["", "## Examples", "",
              _ex_block("Good output", examples["good"]),
              _ex_block("Validator-fixed output", examples["validator_fixed"]),
              _ex_block("Bad / weak output", examples["bad"])]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 6 MVP evaluation")
    p.add_argument("--model", default="auto", help="auto | qwen | mock")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out-dir", default="phase6_outputs")
    args = p.parse_args(argv)
    run(model=args.model, k=args.k, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
