"""LLM Component — Phase 8 acceptance check (LLM_PLAN.md Phase 8, step 5).

Generates from the **COMPACT** prompt with the safety validator **switched off**,
then measures what fine-tuning was supposed to buy:

  * `raw_unsafe_rate`  — the headline. Phase 6 base scored 0.10 *with* the
                         validator patching it. Target here: ~0 with no validator.
  * `format_ok_rate`   — all 5 sections parse (structural validity).
  * `mentions_risk`    — the answer states the screening risk level.
  * `readability`      — Flesch-Kincaid grade (target <= 8).

Run it twice to get the Phase-9 ablation pair:

    # base arm (no adapter)
    python -m llm.finetune.evaluate_adapter --data-dir phase7_data --tag base
    # fine-tuned arm
    python -m llm.finetune.evaluate_adapter --data-dir phase7_data \
        --adapter llm/finetune/adapters/v1/adapter --tag finetuned

Note: `scan_unsafe` needs the patient's BP to judge whether an emergency
redirect was required. Phase-7 rows don't store `patient_inputs`, so we recover
SBP/DBP from the rendered PATIENT PROFILE block in the user turn (and prefer
`meta.patient_inputs` when a future dataset provides it).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..eval.score import mentions_risk, readability_grade
from ..serving.contract import parse_guidance
from ..serving.safety_validator import scan_unsafe
from .data import load_examples

_SBP = re.compile(r"Systolic blood pressure:\s*([\d.]+)", re.I)
_DBP = re.compile(r"Diastolic blood pressure:\s*([\d.]+)", re.I)


def context_from_example(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the `scan_unsafe` context for one Phase-7 row."""
    risk = str(row.get("risk_class", "")).lower()
    meta = row.get("meta") or {}
    inputs: Dict[str, Any] = dict(meta.get("patient_inputs") or {})

    if not inputs:
        user_turn = next((m["content"] for m in row["messages"]
                          if m.get("role") == "user"), "")
        m = _SBP.search(user_turn)
        if m:
            inputs["mean_sbp"] = float(m.group(1))
        m = _DBP.search(user_turn)
        if m:
            inputs["mean_dbp"] = float(m.group(1))

    return {
        "risk_class": risk,
        "risk_display": f"{risk.capitalize()} risk" if risk else "",
        "patient_inputs": inputs,
        "symptoms": "",
    }


def evaluate(data_dir: str | Path = "phase7_data",
             val_file: Optional[str] = None,
             adapter: Optional[str] = None,
             model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
             n: Optional[int] = None,
             out_dir: str | Path = "phase8_eval",
             tag: str = "finetuned") -> Dict[str, Any]:
    path = Path(val_file) if val_file else Path(data_dir) / "val.jsonl"
    rows = load_examples(path)
    if n:
        rows = rows[:n]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..serving.model_backends import QwenChatModel
    print(f"Loading {model_name}" + (f" + adapter {adapter}" if adapter else " (base, no adapter)"))
    model = QwenChatModel(model_name=model_name, adapter_path=adapter)

    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        prompt_messages = row["messages"][:-1]          # compact system + user
        ctx = context_from_example(row)
        text = model.generate(prompt_messages)          # NO validator

        violations = scan_unsafe(text, ctx)
        parsed = parse_guidance(text)
        records.append({
            "profile_id": row.get("profile_id", f"row{i}"),
            "risk_class": row.get("risk_class", ""),
            "raw_unsafe": int(bool(violations)),
            "violation_categories": ";".join(sorted({v.category for v in violations})),
            "format_ok": int(parsed.is_structurally_valid()),
            "mentions_risk": int(mentions_risk(text, row.get("risk_class", ""))),
            "readability": round(readability_grade(text), 2),
            "n_chars": len(text),
            "text": text,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] raw_unsafe so far: "
                  f"{sum(r['raw_unsafe'] for r in records)}")

    summary = _summarize(records, tag, adapter, model_name)
    _write_outputs(records, summary, out_dir, tag)
    _print_summary(summary)
    return {"records": records, "summary": summary}


def _summarize(records: List[Dict[str, Any]], tag: str, adapter: Optional[str],
               model_name: str) -> Dict[str, Any]:
    n = len(records) or 1
    reads = [r["readability"] for r in records]
    cats: Dict[str, int] = {}
    for r in records:
        for c in filter(None, r["violation_categories"].split(";")):
            cats[c] = cats.get(c, 0) + 1
    return {
        "tag": tag,
        "arm": "fine-tuned" if adapter else "base",
        "model": model_name,
        "adapter": adapter or "",
        "n": len(records),
        "raw_unsafe_rate": round(sum(r["raw_unsafe"] for r in records) / n, 4),
        "format_ok_rate": round(sum(r["format_ok"] for r in records) / n, 4),
        "mentions_risk_rate": round(sum(r["mentions_risk"] for r in records) / n, 4),
        "mean_readability": round(sum(reads) / n, 2) if reads else 0.0,
        "readability_ok_rate": round(sum(1 for x in reads if x <= 8) / n, 4),
        "violation_categories": cats,
    }


def _write_outputs(records, summary, out_dir: Path, tag: str) -> None:
    csv_path = out_dir / f"phase8_{tag}_results.csv"
    cols = ["profile_id", "risk_class", "raw_unsafe", "violation_categories",
            "format_ok", "mentions_risk", "readability", "n_chars"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    (out_dir / f"phase8_{tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        f"# Phase 8 — Adapter evaluation ({summary['arm']})", "",
        f"- Date: {datetime.now(timezone.utc).date().isoformat()}",
        f"- Model: `{summary['model']}`",
        f"- Adapter: `{summary['adapter'] or 'none (base arm)'}`",
        f"- Prompt: **compact** system prompt | Validator: **OFF** (raw generations)",
        f"- Examples: {summary['n']}", "",
        "| Metric | Value |", "|---|---|",
        f"| raw_unsafe_rate (target ~0) | **{summary['raw_unsafe_rate']}** |",
        f"| format_ok_rate | {summary['format_ok_rate']} |",
        f"| mentions_risk_rate | {summary['mentions_risk_rate']} |",
        f"| mean_readability (target <=8) | {summary['mean_readability']} |",
        f"| readability_ok_rate | {summary['readability_ok_rate']} |",
        "",
        f"Violation categories: `{summary['violation_categories'] or '{}'}`",
    ]
    bad = [r for r in records if r["raw_unsafe"]][:3]
    if bad:
        md += ["", "## Unsafe examples (first 3)"]
        for r in bad:
            md += ["", f"**{r['profile_id']}** ({r['risk_class']}) — "
                       f"`{r['violation_categories']}`", "```", r["text"][:700], "```"]
    (out_dir / f"phase8_{tag}_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nOutputs -> {out_dir}/phase8_{tag}_*.csv|json|md")


def _print_summary(s: Dict[str, Any]) -> None:
    print(f"\n=== Phase 8 eval ({s['arm']}) - n={s['n']} ===")
    print(f"  raw_unsafe_rate    : {s['raw_unsafe_rate']}   <- headline (validator OFF)")
    print(f"  format_ok_rate     : {s['format_ok_rate']}")
    print(f"  mentions_risk_rate : {s['mentions_risk_rate']}")
    print(f"  mean_readability   : {s['mean_readability']} "
          f"(<=8 in {s['readability_ok_rate']})")
    if s["violation_categories"]:
        print(f"  violations         : {s['violation_categories']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 8 adapter acceptance check")
    p.add_argument("--data-dir", default="phase7_data")
    p.add_argument("--val-file", default=None)
    p.add_argument("--adapter", default=None,
                   help="path to the LoRA adapter; omit to evaluate the BASE arm")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n", type=int, default=None, help="limit examples")
    p.add_argument("--out-dir", default="phase8_eval")
    p.add_argument("--tag", default=None, help="output filename tag")
    a = p.parse_args(argv)

    tag = a.tag or ("finetuned" if a.adapter else "base")
    evaluate(data_dir=a.data_dir, val_file=a.val_file, adapter=a.adapter,
             model_name=a.model, n=a.n, out_dir=a.out_dir, tag=tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
