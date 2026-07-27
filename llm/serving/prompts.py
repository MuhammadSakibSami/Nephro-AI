"""LLM Component — Canonical Prompts (LLM_PLAN.md Phase 1).

Single source of truth for:
  * the SYSTEM prompt (role, scope, safety rules, output format)
  * the canonical SAFETY_RULES list (reused by the Phase 4 safety validator)
  * the prompt/message builders used by data generation (Phase 7),
    fine-tuning (Phase 8), and serving (Phase 5/10).

Design constraints baked in here:
  * Role = guideline-grounded kidney-risk EDUCATION (not counseling /
    diagnosis / treatment).
  * Output = fixed, evidence-cited, NON-chain-of-thought structure.
  * KDOQI nutrition restriction: never give individualized protein / potassium
    / phosphorus / sodium targets; defer to clinician / renal dietitian.
  * Readability target <= grade 8.

Pure-Python (stdlib only) so it imports without torch / transformers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .contract import (
    EMERGENCY_DBP_THRESHOLD,
    EMERGENCY_SBP_THRESHOLD,
    GROUNDING_LABEL,
    LLMInput,
    OUTPUT_SECTION_LABELS,
    READABILITY_TARGET_GRADE,
)

# ---------------------------------------------------------------------------
# Scope statement — keep wording aligned with LLM_PLAN.md.
# ---------------------------------------------------------------------------
SCOPE_STATEMENT = (
    "This assistant provides personalized, guideline-grounded kidney-risk "
    "education after machine-learning-based screening. It does not diagnose "
    "CKD, prescribe treatment, or replace clinician consultation."
)

# ---------------------------------------------------------------------------
# Canonical safety rules. The Phase 4 validator enforces these post-hoc; the
# model is also trained/instructed to follow them. Keep this list as the one
# authoritative copy (the backend's llm_context.strict_safety_rules mirrors it).
# ---------------------------------------------------------------------------
SAFETY_RULES: List[str] = [
    "Do not state or imply a diagnosis. Never say the person has CKD, kidney "
    "disease, or kidney failure. This is a screening-based risk level, not a diagnosis.",
    "Do not prescribe, recommend, start, stop, change, or dose any medication "
    "or supplement.",
    "Do not give individualized nutrition restrictions or numeric targets for "
    "protein, potassium, phosphorus, or sodium. If diet is relevant, suggest "
    "discussing a personalized plan with a clinician or registered/renal dietitian.",
    "Do not replace clinician judgment, eGFR testing, or urine albumin (UACR) testing.",
    "Give wellness education that is tailored to the patient's specific "
    "conditions and risk factors shown in the patient profile. You MAY and SHOULD "
    "reference their actual conditions (e.g., 'because you have high blood "
    "pressure' or 'since your blood sugar is elevated') when giving food, "
    "activity, or monitoring advice. Do NOT diagnose, prescribe, or treat — "
    "but do personalize the wellness guidance to what this specific patient "
    "reported.",
    "For a moderate or high screening result, always include a reminder to "
    "confirm the result with a clinician and appropriate lab tests.",
    "If the inputs suggest an urgent problem (for example chest pain, severe "
    "shortness of breath, severe swelling, or very high blood pressure), advise "
    "seeking prompt medical care.",
    "Base the education on the provided guideline excerpts. If the excerpts do "
    "not cover something, keep the advice general, safe, and non-specific.",
    "Use calm, supportive, non-alarming, plain language at about an 8th-grade "
    "reading level.",
]

# ---------------------------------------------------------------------------
# Output format block (literal instruction shown to the model).
# ---------------------------------------------------------------------------
_SECTION_HINTS = {
    "Summary": "one or two plain sentences naming the screening risk level and what it means in everyday terms (not a diagnosis); mention the one or two factors that most influenced it",
    "Food habits": "balanced eating habits tied to THIS person's specific factors (e.g. salt for blood pressure, balanced carbohydrates for blood sugar, weight-friendly choices); no numeric nutrient targets or restrictions",
    "Rest and sleep": "simple, practical sleep and rest habits, and briefly how good rest supports THIS person's situation (for example steadier blood pressure or blood sugar when those are factors)",
    "Daily routine": "daily movement, hydration, and routine habits chosen for THIS person's factors (e.g. activity level for weight or blood pressure, hydration habits)",
    "Follow-up reminder": "when and why to check in with a clinician and which routine tests to ask about, framed around this person's risk level and factors; required for moderate/high",
}

def _output_format_block() -> str:
    lines = [
        "Respond with EXACTLY these labeled sections, in this order, and nothing else.",
        "Do not add headings, bullet characters, markdown, or any reasoning/explanation of your own process.",
        "",
    ]
    for label in OUTPUT_SECTION_LABELS:
        lines.append(f"{label}:")
        lines.append(f"<{_SECTION_HINTS.get(label, '')}>")
        lines.append("")
    lines.append(f"{GROUNDING_LABEL}: <JSON array of the guideline excerpt ids you actually used, e.g. [\"KDIGO2024_diet_003\"]>")
    lines.append("")
    lines.append(
        f"The '{GROUNDING_LABEL}' line is internal metadata for record-keeping; "
        "list only excerpt ids that genuinely informed the education."
    )
    return "\n".join(lines)


def build_system_prompt() -> str:
    """The canonical system prompt. Stable across data-gen / training / serving."""
    rules = "\n".join(f"- {r}" for r in SAFETY_RULES)
    return (
        "You are a careful kidney-health education assistant.\n\n"
        f"SCOPE\n{SCOPE_STATEMENT}\n\n"
        "ROLE\n"
        "You receive a machine-learning kidney-risk SCREENING result for one person "
        "(a risk level of low, moderate, or high, the factors that influenced it, and "
        "some of their routine health inputs), together with short excerpts from "
        "clinical guidelines (KDIGO, ADA, NKF) and patient-education sources. "
        "Your job is to turn this into safe, personalized, easy-to-read kidney-risk "
        "EDUCATION that helps the person support their kidney health and know when to "
        "see a professional.\n\n"
        f"SAFETY RULES (follow all, always)\n{rules}\n\n"
        f"READING LEVEL\nWrite at about a grade-{READABILITY_TARGET_GRADE} reading "
        "level. Short sentences. Everyday words.\n\n"
        f"OUTPUT FORMAT\n{_output_format_block()}"
    )


def build_compact_system_prompt() -> str:
    """A short system prompt for the FINE-TUNED model (Phase 7/8).

    The full safety/format detail is internalized during fine-tuning, so the
    deployed prompt can be short → cheaper inference. Training pairs this compact
    prompt with gold answers that were generated under the full prompt.
    """
    labels = " / ".join(OUTPUT_SECTION_LABELS)
    return (
        "You are a kidney-health education assistant. Given a kidney-risk SCREENING "
        "result and short guideline excerpts, write safe, personalized, easy-to-read "
        f"(grade ~{READABILITY_TARGET_GRADE}) kidney-risk EDUCATION.\n"
        "Rules: do not diagnose; do not name, start, stop, change, or dose any "
        "medication or supplement; do not give numeric nutrient targets (defer diet "
        "specifics to a clinician or renal dietitian). For a moderate/high result, "
        "include a clinician + lab-test follow-up reminder. If inputs suggest an "
        "urgent problem (e.g. very high blood pressure), advise prompt medical care.\n"
        f"Reply with EXACTLY these labeled sections, in order — {labels} — then a final "
        f"line '{GROUNDING_LABEL}: [list of the guideline excerpt ids you used]'. "
        "No other text, no reasoning."
    )


# ---------------------------------------------------------------------------
# User-turn rendering: ML output + retrieved guideline excerpts.
# ---------------------------------------------------------------------------
def _render_probabilities(probs: Dict[str, float]) -> str:
    parts = []
    for cls_name in ["low", "moderate", "high"]:
        if cls_name in probs:
            parts.append(f"{cls_name}={probs[cls_name]:.2f}")
    return ", ".join(parts)


def _render_factors(inp: LLMInput, k: int = 6) -> str:
    if not inp.key_factors:
        return "  (no individual factor breakdown available)"
    lines = []
    for f in inp.key_factors[:k]:
        arrow = "raises" if f.direction == "supports_prediction" else "lowers"
        lines.append(f"  - {f.human_phrase} ({arrow} this risk level)")
    return "\n".join(lines)


# Human-readable labels/units for the raw patient input fields.
_PATIENT_FIELD_META: Dict[str, Tuple[str, str]] = {
    "age":                       ("Age",                                    "years"),
    "sex_female":                ("Sex",                                    ""),
    "mean_sbp":                  ("Systolic blood pressure",                "mmHg"),
    "mean_dbp":                  ("Diastolic blood pressure",               "mmHg"),
    "bmi":                       ("BMI",                                    "kg/m²"),
    "waist":                     ("Waist circumference",                    "cm"),
    "hba1c":                     ("HbA1c (blood sugar control)",            "%"),
    "glucose":                   ("Fasting blood glucose",                  "mg/dL"),
    "blood_urea_nitrogen":       ("Blood urea nitrogen (BUN)",              "mg/dL"),
    "uric_acid":                 ("Uric acid",                              "mg/dL"),
    "serum_albumin":             ("Serum albumin (nutrition marker)",       "g/dL"),
    "hemoglobin":                ("Hemoglobin",                             "g/dL"),
    "sodium":                    ("Serum sodium",                           "mEq/L"),
    "potassium":                 ("Serum potassium",                        "mEq/L"),
    "mean_pulse":                ("Resting pulse rate",                     "bpm"),
    "diabetes_yes":              ("Has diabetes (diagnosed)",               ""),
    "borderline_diabetes":       ("Has prediabetes / borderline diabetes",  ""),
    "hypertension_doctor_told":  ("Has high blood pressure (diagnosed)",    ""),
    "taking_bp_medicine":        ("Currently on blood pressure medicine",   ""),
    "high_cholesterol_told":     ("Has high cholesterol (diagnosed)",       ""),
    "taking_cholesterol_medicine": ("Currently on cholesterol medicine",    ""),
    "income_poverty_ratio":      ("Income-to-poverty ratio",                ""),
}

_BINARY_FIELDS = {
    "sex_female", "diabetes_yes", "borderline_diabetes",
    "hypertension_doctor_told", "taking_bp_medicine",
    "high_cholesterol_told", "taking_cholesterol_medicine",
}


def _render_patient_profile(inp: "LLMInput") -> str:
    """Render the patient's raw input values as human-readable bullet lines."""
    pi = getattr(inp, "patient_inputs", None) or {}
    if not pi:
        return "  (patient profile not available)"
    lines = []
    for key, raw_val in pi.items():
        if raw_val is None:
            continue
        try:
            fval = float(raw_val)
            if fval != fval:  # NaN
                continue
        except (TypeError, ValueError):
            continue

        label, unit = _PATIENT_FIELD_META.get(key, (key.replace("_", " ").title(), ""))
        if key == "sex_female":
            display = "Female" if fval else "Male"
        elif key in _BINARY_FIELDS:
            display = "Yes" if fval else "No"
        elif isinstance(raw_val, float) or (isinstance(raw_val, int) and key not in _BINARY_FIELDS):
            display = f"{fval:.1f}"
        else:
            display = str(raw_val)
        if unit:
            display = f"{display} {unit}"
        lines.append(f"  - {label}: {display}")
    return "\n".join(lines) if lines else "  (no patient profile data)"


def _build_personalization_cues(inp: "LLMInput") -> str:
    """Derive explicit per-section personalization hints from the patient profile.

    Returns a short bulleted list that is injected into the user prompt so the
    model knows exactly WHAT to reference in each section for THIS patient.
    """
    pi = getattr(inp, "patient_inputs", None) or {}

    def flag(key: str) -> bool:
        v = pi.get(key)
        try:
            return bool(float(v)) if v is not None else False
        except (TypeError, ValueError):
            return False

    def num(key: str) -> Optional[float]:
        v = pi.get(key)
        try:
            f = float(v)
            return None if f != f else f  # reject NaN
        except (TypeError, ValueError):
            return None

    has_diabetes      = flag("diabetes_yes")
    has_prediabetes   = flag("borderline_diabetes")
    has_htn           = flag("hypertension_doctor_told")
    on_bp_meds        = flag("taking_bp_medicine")
    has_chol          = flag("high_cholesterol_told")
    sbp               = num("mean_sbp")
    hba1c             = num("hba1c")
    bmi_val           = num("bmi")
    age_val           = num("age")

    elevated_bp  = has_htn or (sbp is not None and sbp > 130)
    elevated_bmi = bmi_val is not None and bmi_val >= 25
    any_dm       = has_diabetes or has_prediabetes or (hba1c is not None and hba1c > 5.7)

    cues: List[str] = []

    # Summary
    factor_phrases = inp.top_factor_phrases(2)
    summary_cue = f"Name the {inp.risk_class} risk level in plain terms."
    if factor_phrases:
        summary_cue += f" Briefly mention: {'; '.join(factor_phrases)}."
    cues.append(f"- Summary: {summary_cue}")

    # Food habits
    food_parts: List[str] = []
    if any_dm:
        food_parts.append("focus on low-sugar, low-refined-carb choices because of elevated blood sugar")
    if elevated_bp:
        food_parts.append("suggest reducing salt intake because of high blood pressure")
    if has_chol:
        food_parts.append("recommend heart-healthy foods (less saturated fat) because of high cholesterol")
    if elevated_bmi:
        food_parts.append("encourage balanced portions and whole foods to support a healthy weight")
    cues.append("- Food habits: " + ("; ".join(food_parts) if food_parts
                else "Tailor to the patient's specific risk factors listed in the profile."))

    # Rest and sleep — kept brief and general
    cues.append("- Rest and sleep: Give concise, general sleep-hygiene advice (7-9 hours, consistent schedule). Keep this section brief.")

    # Daily routine
    routine_parts: List[str] = []
    if has_htn or on_bp_meds:
        routine_parts.append("remind them to monitor blood pressure regularly and log the readings")
    if has_diabetes:
        routine_parts.append("encourage daily blood-sugar checks")
    if elevated_bmi:
        routine_parts.append("suggest starting with gentle daily walks to support weight management")
    if age_val is not None and age_val >= 60:
        routine_parts.append("suggest low-impact activities appropriate for their age")
    cues.append("- Daily routine: " + ("; ".join(routine_parts) if routine_parts
                else "Tailor movement and monitoring habits to the patient's conditions."))

    # Follow-up reminder
    followup_parts: List[str] = []
    if inp.risk_class in ("moderate", "high"):
        followup_parts.append("recommend a urine test (for protein) and blood test for kidney function soon")
    if has_diabetes or any_dm:
        followup_parts.append("mention asking the doctor about an HbA1c check")
    if has_htn or elevated_bp:
        followup_parts.append("suggest a blood pressure review with their doctor")
    if sbp is not None and sbp >= 160:
        followup_parts.append("note that their blood pressure reading is very high and they should see a doctor promptly")
    cues.append("- Follow-up reminder: " + ("; ".join(followup_parts) if followup_parts
                else "Remind them to follow up with a clinician given the screening result."))

    return "\n".join(cues)


def render_retrieved_chunks(chunks: Optional[List[Dict[str, Any]]]) -> str:
    """Render retrieved guideline excerpts for the prompt.

    Each chunk dict is expected to have at least: 'chunk_id', 'text'.
    Optional 'source' / 'guideline' / 'restricted' fields are surfaced so the
    model can respect the KDOQI restriction.
    """
    if not chunks:
        return "(no guideline excerpts were retrieved; keep the education general and safe)"
    blocks = []
    for c in chunks:
        cid = c.get("chunk_id", "UNKNOWN_ID")
        src = c.get("source") or c.get("guideline") or "guideline"
        restricted = c.get("restricted", False)
        tag = " [RESTRICTED: mention only as 'discuss with a clinician/renal dietitian']" if restricted else ""
        text = str(c.get("text", "")).strip()
        blocks.append(f"[{cid}] (source: {src}){tag}\n{text}")
    return "\n\n".join(blocks)


def _emergency_alert_block(inp: LLMInput) -> str:
    """An unmissable, patient-specific hypertensive-emergency alert.

    Cites the PATIENT'S OWN triggering value(s) rather than only the abstract
    threshold rule — a generic "blood pressure is in the emergency range"
    line was observed to be ignored by the model when systolic looked normal
    and only diastolic was elevated (the model anchors on systolic as "the"
    blood pressure number). Naming the actual reading the patient reported
    makes the urgent condition concrete instead of something to compare
    against a rule.
    """
    if not inp.is_emergency_bp:
        return ""

    def _num(key: str) -> Optional[float]:
        try:
            v = float(inp.patient_inputs.get(key))
            return None if v != v else v
        except (TypeError, ValueError):
            return None

    sbp, dbp = _num("mean_sbp"), _num("mean_dbp")
    sbp_flagged = sbp is not None and sbp >= EMERGENCY_SBP_THRESHOLD
    dbp_flagged = dbp is not None and dbp >= EMERGENCY_DBP_THRESHOLD
    readings = []
    if sbp_flagged:
        readings.append(f"a systolic (top number) reading of {sbp:.0f} mmHg")
    if dbp_flagged:
        readings.append(f"a diastolic (bottom number) reading of {dbp:.0f} mmHg")
    reading_text = " and ".join(readings) if readings else "a blood pressure reading"

    # The "may look ordinary" caveat only matters for the diastolic-only case —
    # a systolic >=180 reading is not subtle and doesn't need that framing.
    caveat = (" even though the systolic number alone may look ordinary at a glance"
              if dbp_flagged and not sbp_flagged else "")

    return (
        "URGENT VITALS ALERT\n"
        f"This patient reported {reading_text}, which is in the hypertensive-emergency "
        f"range (systolic >= {EMERGENCY_SBP_THRESHOLD:.0f} or diastolic >= "
        f"{EMERGENCY_DBP_THRESHOLD:.0f} mmHg){caveat}. Regardless of the overall "
        "screening risk level, you MUST advise seeking prompt/urgent medical care "
        "(e.g. in the Summary or Follow-up reminder section).\n\n"
    )


def build_user_prompt(
    inp: LLMInput,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render the user turn from the frozen LLM input + retrieved excerpts."""
    at_risk_note = (
        "This is a moderate/high screening result: a follow-up reminder to confirm "
        "with a clinician and lab tests is REQUIRED."
        if inp.is_at_risk
        else "This is a low screening result: keep the tone reassuring and preventive."
    )
    emergency_block = _emergency_alert_block(inp)
    missing = (
        f"{inp.n_missing_features} input(s) were missing and estimated by the model"
        + (f" ({', '.join(inp.missing_feature_labels)})." if inp.missing_feature_labels else ".")
        if inp.n_missing_features
        else "All expected inputs were provided."
    )

    personalization_cues = _build_personalization_cues(inp)
    return (
        f"{emergency_block}"
        "SCREENING RESULT\n"
        f"- Risk level: {inp.risk_class_display} ({inp.risk_class})\n"
        f"- Class probabilities: {_render_probabilities(inp.probabilities)}\n"
        f"- Model confidence: {inp.confidence:.2f}\n"
        f"- Reliability: {inp.reliability_level}\n"
        f"- Input completeness: {missing}\n"
        f"- Note: {at_risk_note}\n\n"
        "PATIENT PROFILE (actual health values for this person — use these to personalize every section)\n"
        f"{_render_patient_profile(inp)}\n\n"
        "TOP FACTORS THAT MOST INFLUENCED THIS RESULT\n"
        f"{_render_factors(inp)}\n\n"
        "GUIDELINE EXCERPTS (use these to ground the education; cite their ids in the grounding line)\n"
        f"{render_retrieved_chunks(retrieved_chunks)}\n\n"
        "Now write the kidney-risk education for this person using the required output format.\n"
        "PERSONALIZATION REQUIREMENTS — tailor EACH section to this specific patient:\n"
        f"{personalization_cues}\n\n"
        "Important: Do NOT write generic advice that could apply to any patient. "
        "Reference the patient's actual conditions and values from the PATIENT PROFILE "
        "above in each section where relevant. Follow every safety rule."
    )


def build_messages(
    inp: LLMInput,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Chat-format messages for Qwen3 (system + user).

    The tokenizer's chat template is applied downstream (fine-tune / serving);
    here we only assemble role/content pairs so the same builder feeds every phase.
    """
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(inp, retrieved_chunks)},
    ]


def build_compact_messages(
    inp: LLMInput,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Chat messages with the COMPACT system prompt (fine-tuned model / training)."""
    return [
        {"role": "system", "content": build_compact_system_prompt()},
        {"role": "user", "content": build_user_prompt(inp, retrieved_chunks)},
    ]


def render_messages_preview(messages: List[Dict[str, str]]) -> str:
    """Human-readable flattening of chat messages (for previews / debugging)."""
    out = []
    for m in messages:
        out.append(f"===== {m['role'].upper()} =====\n{m['content']}")
    return "\n\n".join(out)
