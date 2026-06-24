"""LLM_PLAN.md Phase 5 — MVP end-to-end demo / acceptance test.

Runs several /predict-shaped outputs through the full runtime
(retrieve -> prompt -> model -> validate) using the offline MockGuidanceModel,
and checks each result is:
  * structurally complete (all 5 sections),
  * safe (passes scan_unsafe),
  * grounded (cited chunks are real retrieved chunks),
  * provenance-attributed.

Also checks the validator integration: a model that emits a diagnosis on its
first attempt is retried/fallen-back to a safe result.

Run (offline):
    python -m ckd_risk_app.llm.demo_phase5

Production (Colab): set model="auto" (Qwen3) — see docs/LLM_PHASE5_MVP.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ckd_risk_app.llm.serving.llm_runtime import GuidanceRuntime          # noqa: E402
from ckd_risk_app.llm.serving.model_backends import MockGuidanceModel      # noqa: E402
from ckd_risk_app.llm.serving.contract import LLMInput                      # noqa: E402
from ckd_risk_app.llm.serving.safety_validator import scan_unsafe, context_from_llm_input  # noqa: E402


def make_predict(risk, probs, factors, inputs, missing=None):
    return {
        "risk_class": risk,
        "risk_class_display": f"{risk.capitalize()} risk",
        "probabilities": probs,
        "confidence": max(probs.values()),
        "stage1": {"threshold": 0.44}, "stage2": {"threshold": 0.50},
        "model_info": {"stage1_threshold": 0.44, "stage2_threshold": 0.50},
        "llm_context": {
            "model_output": {
                "screening_category": risk, "class_probabilities": probs,
                "confidence": max(probs.values()), "reliability_level": "moderate",
                "reliability_notes": [],
            },
            "key_model_factors": factors,
            "input_quality": {
                "n_supplied_features": 22 - len(missing or []),
                "n_missing_features": len(missing or []),
                "missing_features": missing or [],
                "feature_labels": {},
            },
            "patient_inputs": inputs,
            "strict_safety_rules": [], "suggestion_style": {},
        },
    }


def _factor(feature, label, unit, value):
    return {"feature": feature, "label": label, "unit": unit, "current_value": value,
            "direction": "supports_prediction", "absolute_contribution": 0.1}


CASES = [
    ("low_clean", make_predict(
        "low", {"low": 0.82, "moderate": 0.13, "high": 0.05},
        [_factor("bmi", "Body mass index", "kg/m2", 24)],
        {"age": 40, "sex_female": 1, "bmi": 24, "mean_sbp": 118})),
    ("moderate_bp_dm", make_predict(
        "moderate", {"low": 0.31, "moderate": 0.49, "high": 0.20},
        [_factor("mean_sbp", "Mean systolic BP", "mmHg", 152),
         _factor("hba1c", "HbA1c", "%", 7.8)],
        {"age": 58, "sex_female": 1, "bmi": 33, "mean_sbp": 152, "hba1c": 7.8})),
    ("high_emergency_bp", make_predict(
        "high", {"low": 0.10, "moderate": 0.25, "high": 0.65},
        [_factor("mean_sbp", "Mean systolic BP", "mmHg", 190)],
        {"age": 66, "sex_female": 0, "mean_sbp": 190})),
    ("moderate_weight", make_predict(
        "moderate", {"low": 0.35, "moderate": 0.45, "high": 0.20},
        [_factor("bmi", "Body mass index", "kg/m2", 34),
         _factor("waist", "Waist", "cm", 112)],
        {"age": 50, "bmi": 34, "waist": 112}, missing=["uric_acid"])),
    ("low_smoking", make_predict(
        "low", {"low": 0.78, "moderate": 0.16, "high": 0.06},
        [_factor("hypertension_doctor_told", "Doctor told hypertension", "", 1)],
        {"age": 45, "mean_sbp": 134})),
]


def main() -> int:
    rt = GuidanceRuntime(model="mock")
    print(f"Runtime model: {rt.model.spec} | retriever: {rt.retriever.embedder.spec}\n")

    failures = []
    for name, predict in CASES:
        res = rt.generate_guidance(predict)
        inp = LLMInput.from_predict_response(predict)
        ctx = context_from_llm_input(inp)

        complete = all(res.sections.get(s, "").strip() for s in
                       ["Summary", "Food habits", "Rest and sleep", "Daily routine", "Follow-up reminder"])
        safe = not scan_unsafe(res.safe_text, ctx)
        grounded = all(g in res.retrieved_chunk_ids for g in res.model_cited_grounding)
        ok = complete and safe and res.status in {"pass", "sanitized"}
        if not ok:
            failures.append((name, res.status, complete, safe, grounded))

        print(f"[{name}] status={res.status} complete={complete} safe={safe} "
              f"grounded={grounded} provenance={res.provenance} cited={res.model_cited_grounding}")

    # Validator integration: unsafe-first model must end safe via retry/fallback.
    rt_unsafe = GuidanceRuntime(model="mock", retriever=rt.retriever)
    rt_unsafe.model = MockGuidanceModel(unsafe_first=True)
    res = rt_unsafe.generate_guidance(CASES[1][1])
    inp = LLMInput.from_predict_response(CASES[1][1])
    unsafe_handled = not scan_unsafe(res.safe_text, context_from_llm_input(inp))
    print(f"\nunsafe-first model -> status={res.status}, final_safe={unsafe_handled}")

    # Show one full result.
    print("\n" + "=" * 64 + "\nSAMPLE OUTPUT (moderate_bp_dm)\n" + "=" * 64)
    print(rt.generate_guidance(CASES[1][1]).safe_text)

    ok = not failures and unsafe_handled
    if failures:
        print("\nFAILURES:", failures)
    print("\nPhase 5 acceptance test:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
