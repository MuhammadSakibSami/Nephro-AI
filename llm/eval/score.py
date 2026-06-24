"""LLM Component — MVP Output Scoring (LLM_PLAN.md Phase 6).

Pure-Python scoring of a generated guidance answer. No heavy deps; uses
`textstat` for readability if available, else a Flesch-Kincaid estimate.

Metrics per output:
  format_ok           - raw model text parses into all 5 required sections
  raw_unsafe          - model output unsafe BEFORE the validator
  final_unsafe        - returned safe_text unsafe AFTER the validator (target False)
  factor_relevance    - fraction of the top-2 model factors addressed in the text
  grounding_has_any   - cited >= 1 valid (retrieved) chunk
  grounding_valid_ratio - cited ids that are real retrieved ids / cited ids
  readability_grade   - Flesch-Kincaid grade of the final text
  readability_ok      - grade <= target (8)
  personalization     - 0..2 (mentions risk level; addresses a top factor)
  mentions_risk       - risk-level word present
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..serving.contract import READABILITY_TARGET_GRADE, parse_guidance
from ..serving.safety_validator import scan_unsafe

# Feature -> lay theme keywords for relevance/personalization.
FACTOR_THEME_KEYWORDS: Dict[str, List[str]] = {
    "mean_sbp": ["blood pressure", "bp"], "mean_dbp": ["blood pressure", "bp"],
    "hypertension_doctor_told": ["blood pressure", "hypertension"],
    "taking_bp_medicine": ["blood pressure"],
    "hba1c": ["blood sugar", "diabetes", "glucose", "a1c"],
    "glucose": ["blood sugar", "glucose", "diabetes"],
    "diabetes_yes": ["diabetes", "blood sugar"],
    "borderline_diabetes": ["prediabetes", "blood sugar", "diabetes"],
    "bmi": ["weight", "body", "obesity"], "waist": ["weight", "waist", "body"],
    "blood_urea_nitrogen": ["kidney", "protein", "diet", "hydration", "fluid"],
    "uric_acid": ["uric acid", "diet", "purine"],
    "serum_albumin": ["nutrition", "protein", "diet"],
    "hemoglobin": ["anemia", "iron", "nutrition"],
    "sodium": ["salt", "sodium"], "potassium": ["potassium", "diet"],
    "high_cholesterol_told": ["cholesterol", "fat", "diet"],
    "taking_cholesterol_medicine": ["cholesterol"],
    "mean_pulse": ["heart", "active", "activity"],
    "age": [], "sex_female": [], "income_poverty_ratio": [],
}

_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.I)
_SENT = re.compile(r"[.!?]+")
_WORD = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------
def _fk_grade_estimate(text: str) -> float:
    words = _WORD.findall(text)
    n_words = len(words)
    n_sents = max(len([s for s in _SENT.split(text) if s.strip()]), 1)
    if n_words == 0:
        return 0.0
    syllables = sum(max(len(_VOWEL_GROUP.findall(w)), 1) for w in words)
    return round(0.39 * (n_words / n_sents) + 11.8 * (syllables / n_words) - 15.59, 2)


def readability_grade(text: str) -> float:
    try:
        import textstat
        return round(float(textstat.flesch_kincaid_grade(text)), 2)
    except Exception:
        return _fk_grade_estimate(text)


# ---------------------------------------------------------------------------
# Relevance / personalization
# ---------------------------------------------------------------------------
def _top_features(profile: Dict[str, Any], n: int = 2) -> List[str]:
    factors = (profile.get("llm_context", {}) or {}).get("key_model_factors", []) or []
    return [f.get("feature") for f in factors[:n]]


def factor_relevance(text: str, profile: Dict[str, Any], n: int = 2) -> Optional[float]:
    low = text.lower()
    scorable, hits = 0, 0
    for feat in _top_features(profile, n):
        kws = FACTOR_THEME_KEYWORDS.get(feat, [])
        if not kws:
            continue
        scorable += 1
        if any(k in low for k in kws):
            hits += 1
    return None if scorable == 0 else round(hits / scorable, 3)


def mentions_risk(text: str, risk_class: str) -> bool:
    return bool(re.search(rf"\b{re.escape(risk_class)}\b", text, re.I))


def personalization(text: str, profile: Dict[str, Any]) -> int:
    score = 0
    if mentions_risk(text, profile.get("risk_class", "")):
        score += 1
    low = text.lower()
    for feat in _top_features(profile, 3):
        if any(k in low for k in FACTOR_THEME_KEYWORDS.get(feat, [])):
            score += 1
            break
    return score


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------
def grounding_scores(cited: List[str], retrieved: List[str]) -> Dict[str, Any]:
    cited = cited or []
    rset = set(retrieved or [])
    valid = [c for c in cited if c in rset]
    return {
        "grounding_has_any": len(valid) > 0,
        "grounding_valid_ratio": round(len(valid) / len(cited), 3) if cited else 0.0,
        "n_cited": len(cited),
    }


# ---------------------------------------------------------------------------
# Full record scorer
# ---------------------------------------------------------------------------
def score_record(profile: Dict[str, Any], result, context: Dict[str, Any]) -> Dict[str, Any]:
    """`result` is a GuidanceResult; `context` from context_from_llm_input."""
    raw = result.raw_model_text or ""
    final = result.safe_text or ""

    fmt_ok = parse_guidance(raw).is_structurally_valid()
    raw_unsafe = len(scan_unsafe(raw, context)) > 0
    final_unsafe = len(scan_unsafe(final, context)) > 0
    grade = readability_grade(final)

    out = {
        "format_ok": fmt_ok,
        "raw_unsafe": raw_unsafe,
        "final_unsafe": final_unsafe,
        "status": result.status,
        "n_attempts": result.n_attempts,
        "factor_relevance": factor_relevance(final, profile),
        "readability_grade": grade,
        "readability_ok": grade <= READABILITY_TARGET_GRADE,
        "personalization": personalization(final, profile),
        "mentions_risk": mentions_risk(final, profile.get("risk_class", "")),
    }
    out.update(grounding_scores(result.model_cited_grounding, result.retrieved_chunk_ids))
    return out


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {}

    def rate(key):
        return round(sum(1 for s in scores if s[key]) / n, 3)

    def mean(key, exclude_none=False):
        vals = [s[key] for s in scores if not (exclude_none and s[key] is None)]
        return round(sum(vals) / len(vals), 3) if vals else None

    statuses = [s["status"] for s in scores]
    return {
        "n": n,
        "format_adherence_rate": rate("format_ok"),
        "raw_unsafe_rate": rate("raw_unsafe"),
        "final_unsafe_rate": rate("final_unsafe"),
        "pass_rate": round(statuses.count("pass") / n, 3),
        "sanitized_rate": round(statuses.count("sanitized") / n, 3),
        "fallback_rate": round(statuses.count("fallback") / n, 3),
        "retry_rate": round(sum(1 for s in scores if s["n_attempts"] > 1) / n, 3),
        "mean_factor_relevance": mean("factor_relevance", exclude_none=True),
        "grounding_any_rate": rate("grounding_has_any"),
        "mean_grounding_valid_ratio": mean("grounding_valid_ratio"),
        "mean_readability_grade": mean("readability_grade"),
        "readability_ok_rate": rate("readability_ok"),
        "mean_personalization": mean("personalization"),
        "mentions_risk_rate": rate("mentions_risk"),
    }
