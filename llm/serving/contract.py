"""LLM Component — Data Contract (LLM_PLAN.md Phase 1).

Single source of truth for the SHAPES that flow through the LLM component.
Every later phase (RAG, data generation, fine-tuning, serving, evaluation)
imports these definitions so nothing drifts.

Two contracts are defined here:

1. INPUT contract  — what the LLM consumes.
   This is the `llm_context` block already produced by the existing backend
   (`model_service._build_llm_context`) plus the surrounding `/predict`
   response. We FREEZE that shape here and provide a small adapter
   (`LLMInput.from_predict_response`) that extracts exactly what the prompt
   builder needs, defensively.

2. OUTPUT contract — what the LLM must produce.
   A fixed, evidence-cited, NON-chain-of-thought structure:

       Summary:
       ...
       Food habits:
       ...
       Rest and sleep:
       ...
       Daily routine:
       ...
       Follow-up reminder:
       ...
       Grounding used internally: ["KDIGO2024_diet_003", "ADA2026_ckd_007"]

   `Grounding used internally` is INTERNAL metadata (logging + groundedness
   evaluation). It is NOT shown verbatim to the user; the frontend shows only
   a general provenance line.

This module is pure Python (stdlib only) so it can be imported and tested
without the model, torch, or any heavy dependency.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Risk classes (must match feature_config.LABEL_ORDER)
# ---------------------------------------------------------------------------
RISK_CLASSES = ["low", "moderate", "high"]
AT_RISK_CLASSES = ["moderate", "high"]  # classes that require a lab/clinician reminder

# ---------------------------------------------------------------------------
# Hypertensive-emergency vitals thresholds. Single source of truth: both the
# prompt builder (so the model is explicitly told) and the safety validator's
# `is_emergency` (so it knows whether an urgent-care line was required) must
# use the SAME threshold, or the model has no way to know what "very high
# blood pressure" numerically means to the validator that checks its output.
# ---------------------------------------------------------------------------
EMERGENCY_SBP_THRESHOLD = 180.0
EMERGENCY_DBP_THRESHOLD = 120.0


def has_emergency_vitals(patient_inputs: Dict[str, Any]) -> bool:
    """True if the patient's BP inputs are in the hypertensive-emergency range."""
    def _num(key: str) -> Optional[float]:
        try:
            v = float(patient_inputs.get(key))
            return None if v != v else v  # reject NaN
        except (TypeError, ValueError):
            return None

    sbp = _num("mean_sbp")
    if sbp is not None and sbp >= EMERGENCY_SBP_THRESHOLD:
        return True
    dbp = _num("mean_dbp")
    if dbp is not None and dbp >= EMERGENCY_DBP_THRESHOLD:
        return True
    return False


# ===========================================================================
# OUTPUT CONTRACT
# ===========================================================================

# Ordered (machine_key, display_label) for the user-facing sections.
OUTPUT_SECTIONS: List[tuple[str, str]] = [
    ("summary", "Summary"),
    ("food_habits", "Food habits"),
    ("rest_and_sleep", "Rest and sleep"),
    ("daily_routine", "Daily routine"),
    ("follow_up_reminder", "Follow-up reminder"),
]

# Display labels in order (used by prompt builder + parser + validator).
OUTPUT_SECTION_LABELS: List[str] = [label for _key, label in OUTPUT_SECTIONS]
OUTPUT_SECTION_KEYS: List[str] = [key for key, _label in OUTPUT_SECTIONS]

# Internal-only grounding line label.
GROUNDING_LABEL = "Grounding used internally"

# Readability goal for generated text (Flesch-Kincaid grade level).
READABILITY_TARGET_GRADE = 8


@dataclass
class GuidanceOutput:
    """Parsed, structured LLM output (the OUTPUT contract)."""

    summary: str = ""
    food_habits: str = ""
    rest_and_sleep: str = ""
    daily_routine: str = ""
    follow_up_reminder: str = ""
    grounding: List[str] = field(default_factory=list)  # internal chunk ids
    raw_text: str = ""

    def user_facing_sections(self) -> Dict[str, str]:
        """The sections that may be shown to the user (grounding excluded)."""
        return {
            "Summary": self.summary,
            "Food habits": self.food_habits,
            "Rest and sleep": self.rest_and_sleep,
            "Daily routine": self.daily_routine,
            "Follow-up reminder": self.follow_up_reminder,
        }

    def missing_sections(self) -> List[str]:
        """Display labels of required sections that came back empty."""
        out = []
        for key, label in OUTPUT_SECTIONS:
            if not str(getattr(self, key, "")).strip():
                out.append(label)
        return out

    def is_structurally_valid(self) -> bool:
        return len(self.missing_sections()) == 0

    def render(self) -> str:
        """Re-emit the guidance in the canonical labeled format (incl. grounding).

        Used by the safety validator to return sanitized output in a clean,
        well-formed shape.
        """
        parts: List[str] = []
        for key, label in OUTPUT_SECTIONS:
            parts.append(f"{label}:")
            parts.append(str(getattr(self, key, "") or "").strip())
            parts.append("")
        parts.append(f"{GROUNDING_LABEL}: {json.dumps(self.grounding, ensure_ascii=False)}")
        return "\n".join(parts).strip()

    def with_sections(self, sections: Dict[str, str]) -> "GuidanceOutput":
        """Return a copy with user-facing sections replaced (by display label)."""
        label_to_key = {label: key for key, label in OUTPUT_SECTIONS}
        updated = GuidanceOutput(
            summary=self.summary, food_habits=self.food_habits,
            rest_and_sleep=self.rest_and_sleep, daily_routine=self.daily_routine,
            follow_up_reminder=self.follow_up_reminder,
            grounding=list(self.grounding), raw_text=self.raw_text,
        )
        for label, text in sections.items():
            key = label_to_key.get(label)
            if key:
                setattr(updated, key, text)
        return updated


# Regex that matches any known section/grounding header at the start of a line.
_ALL_HEADERS = OUTPUT_SECTION_LABELS + [GROUNDING_LABEL]
_HEADER_PATTERN = re.compile(
    r"(?im)^\s*(" + "|".join(re.escape(h) for h in _ALL_HEADERS) + r")\s*:\s*"
)


def parse_guidance(text: str) -> GuidanceOutput:
    """Parse raw model text into a GuidanceOutput.

    Tolerant by design: headers are matched case-insensitively at line starts,
    each section captures everything up to the next known header. Missing
    sections come back as empty strings (caught by `missing_sections`).
    """
    out = GuidanceOutput(raw_text=text or "")
    if not text:
        return out

    # Find all header positions, then slice the body between consecutive headers.
    matches = list(_HEADER_PATTERN.finditer(text))
    label_to_key = {label: key for key, label in OUTPUT_SECTIONS}

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if header == GROUNDING_LABEL:
            out.grounding = _parse_grounding_list(body)
        else:
            key = label_to_key.get(_canonical_label(header))
            if key:
                setattr(out, key, body)

    return out


def _canonical_label(header: str) -> str:
    """Map a case-insensitive matched header back to its canonical label."""
    for label in _ALL_HEADERS:
        if header.lower() == label.lower():
            return label
    return header


def _parse_grounding_list(body: str) -> List[str]:
    """Parse the grounding line body into a list of chunk-id strings."""
    body = body.strip()
    if not body:
        return []
    # Preferred form: a JSON array.
    try:
        first = body.find("[")
        last = body.rfind("]")
        if first != -1 and last != -1 and last > first:
            arr = json.loads(body[first : last + 1])
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # Fallback: comma / whitespace separated, strip quotes and brackets.
    cleaned = body.strip("[]")
    parts = re.split(r"[,\n]", cleaned)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip().strip('"').strip("'")]


# ===========================================================================
# INPUT CONTRACT
# ===========================================================================

# The exact top-level keys we rely on from the `/predict` response.
PREDICT_RESPONSE_KEYS = [
    "risk_class",
    "risk_class_display",
    "probabilities",
    "confidence",
    "stage1",
    "stage2",
    "input_quality",
    "reliability",
    "explanation",
    "llm_context",
    "model_info",
]

# The exact keys we rely on from the `llm_context` sub-block.
LLM_CONTEXT_KEYS = [
    "purpose",
    "strict_safety_rules",
    "model_output",
    "key_model_factors",
    "input_quality",
    "patient_inputs",
    "suggestion_style",
]


@dataclass
class RiskFactor:
    """One contributing model factor (from `key_model_factors`)."""

    feature: str
    label: str
    unit: str
    current_value: Optional[float]
    direction: str  # "supports_prediction" | "pulls_away_from_prediction"
    absolute_contribution: float

    @property
    def human_phrase(self) -> str:
        """A short human-readable phrase, e.g. 'Mean systolic BP 152 mmHg'."""
        val = ""
        if self.current_value is not None:
            num = f"{self.current_value:g}"
            val = f" {num}{(' ' + self.unit) if self.unit else ''}"
        return f"{self.label}{val}".strip()

    @classmethod
    def from_factor_dict(cls, d: Dict[str, Any]) -> "RiskFactor":
        return cls(
            feature=str(d.get("feature", "")),
            label=str(d.get("label", d.get("feature", ""))),
            unit=str(d.get("unit", "") or ""),
            current_value=_safe_float(d.get("current_value")),
            direction=str(d.get("direction", "")),
            absolute_contribution=float(d.get("absolute_contribution", 0.0) or 0.0),
        )


@dataclass
class LLMInput:
    """Frozen, validated view of everything the prompt builder needs.

    Built from a full `/predict` response via `from_predict_response`. All
    fields are plain Python types so this is trivially serializable for the
    training-data pipeline (Phase 7) and the runtime (Phase 5/10).
    """

    risk_class: str
    risk_class_display: str
    probabilities: Dict[str, float]
    confidence: float
    reliability_level: str
    reliability_notes: List[str]
    n_supplied_features: int
    n_missing_features: int
    missing_feature_labels: List[str]
    key_factors: List[RiskFactor]
    patient_inputs: Dict[str, Any]
    stage1_threshold: float
    stage2_threshold: float

    # --- convenience ------------------------------------------------------
    @property
    def is_at_risk(self) -> bool:
        return self.risk_class in AT_RISK_CLASSES

    @property
    def is_emergency_bp(self) -> bool:
        return has_emergency_vitals(self.patient_inputs)

    def top_factor_phrases(self, k: int = 3) -> List[str]:
        supporting = [f for f in self.key_factors if f.direction == "supports_prediction"]
        chosen = (supporting or self.key_factors)[:k]
        return [f.human_phrase for f in chosen]

    # --- construction -----------------------------------------------------
    @classmethod
    def from_predict_response(cls, resp: Dict[str, Any]) -> "LLMInput":
        """Adapter: extract the LLM input from a `/predict` response dict.

        Accepts either the full response or just its `llm_context` block.
        Defensive against missing keys so it never raises on partial data.
        """
        ctx = resp.get("llm_context", resp) or {}
        model_output = ctx.get("model_output", {}) or {}
        input_quality = ctx.get("input_quality", resp.get("input_quality", {})) or {}

        # Probabilities / risk class can come from a few places; prefer ctx.
        probs = (
            model_output.get("class_probabilities")
            or resp.get("probabilities")
            or {}
        )
        probs = {str(k).lower(): _safe_float(v) or 0.0 for k, v in probs.items()}
        for cls_name in RISK_CLASSES:
            probs.setdefault(cls_name, 0.0)

        risk_class = str(
            model_output.get("screening_category")
            or resp.get("risk_class")
            or "low"
        ).lower()

        factors_raw = ctx.get("key_model_factors", []) or []
        key_factors = [RiskFactor.from_factor_dict(d) for d in factors_raw]

        feature_labels = input_quality.get("feature_labels", {}) or {}
        missing = input_quality.get("missing_features", []) or []
        missing_labels = [feature_labels.get(f, f) for f in missing]

        model_info = resp.get("model_info", {}) or {}
        stage1 = resp.get("stage1", {}) or {}
        stage2 = resp.get("stage2", {}) or {}

        return cls(
            risk_class=risk_class,
            risk_class_display=str(
                resp.get("risk_class_display") or risk_class.capitalize()
            ),
            probabilities=probs,
            confidence=_safe_float(
                model_output.get("confidence", resp.get("confidence"))
            ) or 0.0,
            reliability_level=str(
                model_output.get("reliability_level")
                or (resp.get("reliability", {}) or {}).get("level", "unknown")
            ),
            reliability_notes=list(
                model_output.get("reliability_notes")
                or (resp.get("reliability", {}) or {}).get("notes", [])
                or []
            ),
            n_supplied_features=int(input_quality.get("n_supplied_features", 0) or 0),
            n_missing_features=int(input_quality.get("n_missing_features", 0) or 0),
            missing_feature_labels=missing_labels,
            key_factors=key_factors,
            patient_inputs=dict(ctx.get("patient_inputs", {}) or {}),
            stage1_threshold=_safe_float(
                stage1.get("threshold", model_info.get("stage1_threshold"))
            ) or 0.44,
            stage2_threshold=_safe_float(
                stage2.get("threshold", model_info.get("stage2_threshold"))
            ) or 0.50,
        )


# ===========================================================================
# Validation of an INPUT contract (used by tests / data-gen guards)
# ===========================================================================

def validate_predict_response(resp: Dict[str, Any]) -> List[str]:
    """Return a list of contract problems; empty list means the shape is OK."""
    problems: List[str] = []
    if not isinstance(resp, dict):
        return ["response is not a dict"]

    ctx = resp.get("llm_context")
    if not isinstance(ctx, dict):
        problems.append("missing 'llm_context' block")
        return problems

    for key in ["model_output", "key_model_factors", "strict_safety_rules", "suggestion_style"]:
        if key not in ctx:
            problems.append(f"llm_context missing '{key}'")

    mo = ctx.get("model_output", {}) or {}
    if str(mo.get("screening_category", "")).lower() not in RISK_CLASSES:
        problems.append("model_output.screening_category not in low/moderate/high")

    return problems


# ---------------------------------------------------------------------------
def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None
