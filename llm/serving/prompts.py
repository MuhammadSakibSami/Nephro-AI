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

from typing import Any, Dict, List, Optional

from .contract import (
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
    "Only give general wellness education: balanced food habits, hydration "
    "habits, sleep and rest, physical activity, daily routine, and when to seek "
    "professional review.",
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
    missing = (
        f"{inp.n_missing_features} input(s) were missing and estimated by the model"
        + (f" ({', '.join(inp.missing_feature_labels)})." if inp.missing_feature_labels else ".")
        if inp.n_missing_features
        else "All expected inputs were provided."
    )

    return (
        "SCREENING RESULT\n"
        f"- Risk level: {inp.risk_class_display} ({inp.risk_class})\n"
        f"- Class probabilities: {_render_probabilities(inp.probabilities)}\n"
        f"- Model confidence: {inp.confidence:.2f}\n"
        f"- Reliability: {inp.reliability_level}\n"
        f"- Input completeness: {missing}\n"
        f"- Note: {at_risk_note}\n\n"
        "TOP FACTORS THAT INFLUENCED THIS RESULT\n"
        f"{_render_factors(inp)}\n\n"
        "GUIDELINE EXCERPTS (use these to ground the education; cite their ids in the grounding line)\n"
        f"{render_retrieved_chunks(retrieved_chunks)}\n\n"
        "PERSONALIZATION (important)\n"
        "- Write for THIS person, not a generic reader. In EVERY section, connect the "
        "advice to their risk level and the specific top factors listed above (name the "
        "relevant area in plain words, e.g. blood pressure, blood sugar, weight, or hydration).\n"
        "- Make at least one concrete, factor-linked point per section. Avoid one-size-"
        "fits-all sentences that would read the same for any person.\n"
        "- Keep every safety rule: no diagnosis, no medications or supplements, and no "
        "numeric nutrient targets (defer diet specifics to a clinician or renal dietitian).\n\n"
        "Now write the kidney-risk education for this person using the required output format."
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
