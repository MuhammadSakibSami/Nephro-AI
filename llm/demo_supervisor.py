"""Supervisor demo — 3 kidney-risk education cases (LLM_PLAN.md, thesis MVP).

Demonstrates the full LLM component end-to-end through the SAME response builder
the `/generate-guidance` endpoint uses (`backend.llm_endpoint.build_guidance_response`):

  1. Low risk
  2. Moderate risk (blood pressure + glucose)
  3. High risk with EMERGENCY blood pressure — validator adds an urgent-care
     redirect (status = sanitized), showing the safety guardrail at work.

Runs locally with no GPU (falls back to local-demo mode); the production system
uses Qwen3-4B (frozen Phase-6 evaluation). Prints to console and writes
`ckd_risk_app/results/thesis_demo_cases.md`.

Run:
    python -m ckd_risk_app.llm.demo_supervisor
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ckd_risk_app.backend.llm_endpoint import build_guidance_response  # noqa: E402
from ckd_risk_app.llm.data.profiles import build_predict_profile        # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "thesis_demo_cases.md"

CASES = [
    ("Low risk", build_predict_profile(
        "DEMO_LOW", "low", [("bmi", 28.0)], {"age": 42, "sex_female": 1})),
    ("Moderate risk (blood pressure + glucose)", build_predict_profile(
        "DEMO_MODERATE", "moderate", [("mean_sbp", 150.0), ("hba1c", 7.6)],
        {"age": 58, "diabetes_yes": 1, "hypertension_doctor_told": 1})),
    ("High risk / EMERGENCY blood pressure (validator-fixed)", build_predict_profile(
        "DEMO_HIGH_EMERGENCY", "high", [("mean_sbp", 188.0), ("hba1c", 8.8), ("diabetes_yes", 1)],
        {"age": 66, "mean_dbp": 116, "hypertension_doctor_told": 1})),
]


def _render_case(title: str, resp: dict) -> str:
    lines = [f"## {title}", "",
             f"- **mode:** `{resp['mode']}`  |  **model:** `{resp['model']}`  |  "
             f"**risk:** `{resp['risk_class']}`  |  **validator status:** `{resp['status']}`",
             f"- **sources:** {', '.join(resp['provenance'])}",
             f"- **disclaimer:** {resp['disclaimer']}", ""]
    if resp["violations"]:
        cats = ", ".join(v["category"] for v in resp["violations"])
        lines.append(f"- **validator actions:** {cats}")
        lines.append("")
    for label, text in resp["guidance"].items():
        lines.append(f"**{label}:** {text}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    blocks = ["# Thesis Supervisor Demo — Kidney-Risk Education (LLM MVP)", "",
              "Three cases through the `/generate-guidance` pipeline "
              "(RAG → prompt → LLM → safety validator). The high-risk case shows the "
              "validator adding an emergency-care redirect.", ""]
    for title, profile in CASES:
        resp = build_guidance_response(profile)
        block = _render_case(title, resp)
        blocks.append(block)
        # Console
        print("=" * 74)
        print(title, "|", f"mode={resp['mode']} status={resp['status']} risk={resp['risk_class']}")
        print("-" * 74)
        for label, text in resp["guidance"].items():
            print(f"{label}: {text}")
        if resp["violations"]:
            print("validator actions:", ", ".join(v["category"] for v in resp["violations"]))
        print()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(blocks), encoding="utf-8")
    print("=" * 74)
    print("Wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
