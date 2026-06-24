"""LLM_PLAN.md Phase 1 — acceptance demo / smoke test.

Verifies the data contract + prompts without needing the model or any heavy
dependency:

  1. Builds a realistic mock `/predict` response (same shape as
     backend.model_service._build_llm_context).
  2. Adapts it through the INPUT contract (LLMInput.from_predict_response).
  3. Renders the full system + user prompt with mock retrieved guideline chunks.
  4. Round-trips a sample model answer through the OUTPUT parser and validates
     the structure.

Run:
    python -m ckd_risk_app.llm.demo_phase1
or:
    python ckd_risk_app/llm/demo_phase1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (python ckd_risk_app/llm/demo_phase1.py).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ckd_risk_app.llm.serving.contract import (  # noqa: E402
    GuidanceOutput,
    LLMInput,
    parse_guidance,
    validate_predict_response,
)
from ckd_risk_app.llm.serving.prompts import (  # noqa: E402
    build_messages,
    render_messages_preview,
)


def mock_predict_response() -> dict:
    """A realistic moderate-risk `/predict` response (matches backend shape)."""
    key_factors = [
        {
            "feature": "mean_sbp", "label": "Mean systolic BP", "unit": "mmHg",
            "current_value": 152.0, "reference_value": 125.0,
            "contribution_to_predicted_class": 0.14, "absolute_contribution": 0.14,
            "direction": "supports_prediction",
        },
        {
            "feature": "hba1c", "label": "HbA1c", "unit": "%",
            "current_value": 7.8, "reference_value": 5.6,
            "contribution_to_predicted_class": 0.11, "absolute_contribution": 0.11,
            "direction": "supports_prediction",
        },
        {
            "feature": "blood_urea_nitrogen", "label": "Blood urea nitrogen", "unit": "mg/dL",
            "current_value": 22.0, "reference_value": 14.0,
            "contribution_to_predicted_class": 0.07, "absolute_contribution": 0.07,
            "direction": "supports_prediction",
        },
        {
            "feature": "bmi", "label": "Body mass index", "unit": "kg/m²",
            "current_value": 33.0, "reference_value": 28.0,
            "contribution_to_predicted_class": 0.05, "absolute_contribution": 0.05,
            "direction": "supports_prediction",
        },
    ]
    llm_context = {
        "purpose": "Generate non-clinical, supportive lifestyle guidance based on the model screening output.",
        "strict_safety_rules": [
            "Do not diagnose CKD or any disease.",
            "Do not prescribe medication, dosage, supplements, or treatment.",
        ],
        "model_output": {
            "screening_category": "moderate",
            "class_probabilities": {"low": 0.31, "moderate": 0.49, "high": 0.20},
            "confidence": 0.49,
            "reliability_level": "moderate",
            "reliability_notes": ["Prediction is near a model decision boundary."],
        },
        "key_model_factors": key_factors,
        "input_quality": {
            "n_model_features": 22,
            "n_supplied_features": 20,
            "n_missing_features": 2,
            "missing_features": ["uric_acid", "income_poverty_ratio"],
            "feature_labels": {
                "uric_acid": "Uric acid",
                "income_poverty_ratio": "Income-to-poverty ratio",
            },
        },
        "patient_inputs": {
            "age": 58, "sex_female": 1, "bmi": 33.0, "mean_sbp": 152.0,
            "hba1c": 7.8, "blood_urea_nitrogen": 22.0, "diabetes_yes": 1,
            "hypertension_doctor_told": 1, "taking_bp_medicine": 1,
        },
        "suggestion_style": {
            "tone": "calm, supportive, simple, non-alarming",
            "sections": ["Food habits", "Rest and sleep", "Daily routine", "Follow-up reminder"],
            "avoid": ["diagnosis", "treatment claims", "medication instructions"],
        },
    }
    return {
        "risk_class": "moderate",
        "risk_class_display": "Moderate risk",
        "probabilities": {"low": 0.31, "moderate": 0.49, "high": 0.20},
        "confidence": 0.49,
        "stage1": {"p_atrisk": 0.69, "threshold": 0.44, "sent_to_stage2": True},
        "stage2": {"p_high_given_atrisk": 0.29, "threshold": 0.50, "applied_to_final_decision": True},
        "input_quality": llm_context["input_quality"],
        "reliability": {"level": "moderate", "notes": llm_context["model_output"]["reliability_notes"]},
        "explanation": {"method": "local_perturbation", "top_factors": key_factors},
        "llm_context": llm_context,
        "model_info": {"stage1_threshold": 0.44, "stage2_threshold": 0.50},
    }


def mock_retrieved_chunks() -> list:
    return [
        {
            "chunk_id": "KDIGO2024_bp_012", "source": "KDIGO 2024 CKD", "restricted": False,
            "text": "Lifestyle measures to support blood pressure include regular physical "
                    "activity within individual tolerance and maintaining a healthy weight.",
        },
        {
            "chunk_id": "ADA2026_ckd_007", "source": "ADA Standards of Care 2026", "restricted": False,
            "text": "For people with diabetes, keeping blood glucose within an individualized "
                    "target range supports long-term kidney health.",
        },
        {
            "chunk_id": "KDOQI2020_nutr_044", "source": "NKF KDOQI 2020 Nutrition", "restricted": True,
            "text": "Dietary protein and electrolyte needs vary by CKD stage and should be "
                    "individualized by a registered/renal dietitian.",
        },
        {
            "chunk_id": "CDC_ckd_basics_003", "source": "CDC patient education", "restricted": False,
            "text": "Getting enough sleep and staying physically active are general habits that "
                    "support overall and kidney health.",
        },
    ]


def sample_model_answer() -> str:
    """A well-formed answer used to exercise the OUTPUT parser."""
    return (
        "Summary:\n"
        "Your screening puts you at a moderate kidney-risk level. This is not a diagnosis; "
        "it means a few of your numbers are worth keeping an eye on.\n\n"
        "Food habits:\n"
        "Aim for balanced meals with plenty of vegetables and whole foods, and go easy on "
        "very salty and heavily processed foods. For a plan that fits you, talk with a "
        "clinician or a kidney dietitian.\n\n"
        "Rest and sleep:\n"
        "Try for a steady sleep routine of about 7 to 9 hours, going to bed and waking up "
        "around the same time each day.\n\n"
        "Daily routine:\n"
        "Stay active most days with walking or light movement you enjoy, drink water through "
        "the day, and take breaks to lower stress.\n\n"
        "Follow-up reminder:\n"
        "Because this is a moderate result, please check in with a clinician and ask about "
        "simple kidney tests (a blood test and a urine test) to confirm where you stand.\n\n"
        'Grounding used internally: ["KDIGO2024_bp_012", "ADA2026_ckd_007", "CDC_ckd_basics_003"]'
    )


def main() -> int:
    resp = mock_predict_response()

    # 1. Input-contract validation.
    problems = validate_predict_response(resp)
    print("INPUT CONTRACT CHECK:", "OK" if not problems else f"PROBLEMS: {problems}")

    # 2. Adapt through the input contract.
    inp = LLMInput.from_predict_response(resp)
    print(f"Adapted LLMInput: risk={inp.risk_class}, at_risk={inp.is_at_risk}, "
          f"factors={len(inp.key_factors)}, top={inp.top_factor_phrases()}")

    # 3. Build + preview the full prompt.
    messages = build_messages(inp, mock_retrieved_chunks())
    print("\n" + "=" * 70 + "\nFULL PROMPT PREVIEW\n" + "=" * 70)
    print(render_messages_preview(messages))

    # 4. Round-trip the OUTPUT parser.
    print("\n" + "=" * 70 + "\nOUTPUT PARSER ROUND-TRIP\n" + "=" * 70)
    parsed: GuidanceOutput = parse_guidance(sample_model_answer())
    print("Structurally valid:", parsed.is_structurally_valid())
    print("Missing sections:", parsed.missing_sections() or "none")
    print("Grounding ids parsed:", parsed.grounding)
    print("Summary section:", parsed.summary[:80], "...")

    # Acceptance assertions.
    assert not problems, "input contract should be clean"
    assert inp.risk_class == "moderate"
    assert parsed.is_structurally_valid(), "parsed output must have all sections"
    assert parsed.grounding == ["KDIGO2024_bp_012", "ADA2026_ckd_007", "CDC_ckd_basics_003"]
    print("\nPhase 1 acceptance test: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
