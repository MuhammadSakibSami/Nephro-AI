"""Q1 response-quality, grounding and efficiency metrics (Part 4).

All metrics are deterministic and computed from stored run records, so the
analysis notebook can be re-run without touching a GPU.

Operationalisations are stated explicitly because several of these names are
used loosely in the literature:

* `grounding_any`      - the answer cited >=1 retrieved chunk id. This is
                         **source presence, not factual correctness.**
* `valid_source_id`    - every cited id actually exists in the retrieved set
                         (i.e. the model did not invent a citation).
* relevance judgments  - a retrieved chunk counts as relevant when its KB
                         `topic` is in the profile's expected `evidence_topics`.
                         This is a topical proxy, not human relevance labelling.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    from ...serving.contract import parse_guidance, OUTPUT_SECTION_LABELS
    from ..score import readability_grade, factor_relevance, mentions_risk, personalization
except ImportError:  # pragma: no cover
    from llm.serving.contract import parse_guidance, OUTPUT_SECTION_LABELS
    from llm.eval.score import (readability_grade, factor_relevance,
                                mentions_risk, personalization)

REFERRAL_ORDER = ["routine", "primary_care", "urgent", "emergency"]

_EMERGENCY_RE = re.compile(
    r"\b(emergency room|emergency care|call (?:911|999|112|emergency)|"
    r"seek immediate|immediate medical|go to (?:the )?(?:er|a&e|hospital)|"
    r"right away|without delay|urgently)\b", re.I)
_URGENT_RE = re.compile(
    r"\b(as soon as possible|promptly|within (?:a few )?days|soon|"
    r"do not wait|arrange (?:an )?appointment soon)\b", re.I)
_PRIMARY_RE = re.compile(
    r"\b(clinician|doctor|physician|healthcare provider|health professional|nurse|"
    r"gp\b|follow[- ]up|blood test|urine test|lab test|check[- ]?up|get tested)\b", re.I)

_SENT = re.compile(r"(?<=[.!?])\s+")

# Personalisation anchors: mentions of the patient's own values/conditions.
_ANCHOR = re.compile(
    r"\b(your|you're|you have|because you|since you)\b", re.I)
_NUMBER = re.compile(r"\d")

_GENERIC = re.compile(
    r"\b(eat (?:a )?balanced|stay hydrated|exercise regularly|get enough sleep|"
    r"maintain a healthy lifestyle|live healthily|be active)\b", re.I)


# ---------------------------------------------------------------------------
# Format / quality
# ---------------------------------------------------------------------------
def format_metrics(text: str) -> Dict[str, Any]:
    parsed = parse_guidance(text)
    present = parsed.user_facing_sections() or {}
    n_present = sum(1 for lbl in OUTPUT_SECTION_LABELS
                    if (present.get(lbl) or "").strip())
    return {
        "format_ok": int(bool(parsed.is_structurally_valid())),
        "sections_present": n_present,
        "section_completeness": round(n_present / len(OUTPUT_SECTION_LABELS), 4),
        "missing_sections": ";".join(parsed.missing_sections() or []),
    }


def user_facing_text(text: str) -> str:
    """The five patient-facing sections only; excludes the internal grounding line."""
    body = " ".join((parse_guidance(text).user_facing_sections() or {}).values()).strip()
    return body or text


def repetition_rate(text: str) -> float:
    """Fraction of sentences that duplicate an earlier sentence (normalised)."""
    sents = [s.strip().lower() for s in _SENT.split(text) if len(s.strip()) > 25]
    if not sents:
        return 0.0
    seen: Set[str] = set()
    dupes = 0
    for s in sents:
        key = re.sub(r"[^a-z0-9 ]", "", s)
        if key in seen:
            dupes += 1
        seen.add(key)
    return round(dupes / len(sents), 4)


def non_specific_rate(text: str) -> float:
    """Fraction of advice sentences with no patient-specific anchor."""
    sents = [s.strip() for s in _SENT.split(user_facing_text(text)) if len(s.strip()) > 25]
    if not sents:
        return 0.0
    generic = 0
    for s in sents:
        anchored = bool(_ANCHOR.search(s) or _NUMBER.search(s))
        if not anchored or _GENERIC.search(s):
            generic += 1
    return round(generic / len(sents), 4)


def detect_referral_level(text: str) -> str:
    """Highest referral urgency the answer actually communicates."""
    if _EMERGENCY_RE.search(text):
        return "emergency"
    if _URGENT_RE.search(text) and _PRIMARY_RE.search(text):
        return "urgent"
    if _PRIMARY_RE.search(text):
        return "primary_care"
    return "routine"


def followup_appropriate(text: str, expected: str) -> int:
    """1 when the delivered urgency is at least the expected level."""
    got = detect_referral_level(text)
    try:
        return int(REFERRAL_ORDER.index(got) >= REFERRAL_ORDER.index(expected))
    except ValueError:
        return 0


def quality_metrics(text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    ev = profile.get("evaluator", {})
    out: Dict[str, Any] = {}
    out.update(format_metrics(text))
    # Content metrics score what the patient reads; internal source IDs
    # (e.g. "SYK2015_diabetes_001") would otherwise inflate FK grade and
    # keyword-match factor themes.
    body = user_facing_text(text)
    fk = readability_grade(body)
    out["readability_fk"] = round(float(fk), 3)
    out["readability_ok"] = int(fk <= 8.0)
    out["mentions_risk"] = int(bool(mentions_risk(body, profile["risk_class"])))
    fr = factor_relevance(body, profile)
    out["factor_relevance"] = None if fr is None else round(float(fr), 4)
    out["personalization"] = int(personalization(body, profile))
    out["repetition_rate"] = repetition_rate(body)
    out["non_specific_rate"] = non_specific_rate(text)
    out["referral_level_detected"] = detect_referral_level(body)
    out["referral_expected"] = ev.get("expected_referral_level", "routine")
    out["followup_appropriate"] = followup_appropriate(
        body, ev.get("expected_referral_level", "routine"))
    return out


# ---------------------------------------------------------------------------
# Retrieval / grounding
# ---------------------------------------------------------------------------
def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def grounding_metrics(text: str, retrieved_ids: Sequence[str],
                      retrieved_topics: Sequence[str],
                      expected_topics: Sequence[str]) -> Dict[str, Any]:
    """Citation + retrieval quality.

    `retrieved_topics` must align positionally with `retrieved_ids` (rank order).
    Relevance = chunk topic is one the profile is expected to need.
    """
    parsed = parse_guidance(text)
    cited = list(parsed.grounding or [])
    retrieved = list(retrieved_ids)
    rset = set(retrieved)
    expected = set(expected_topics)

    valid = [c for c in cited if c in rset]
    rel = [1.0 if t in expected else 0.0 for t in retrieved_topics]
    n_rel = sum(rel)

    # MRR over the retrieved ranking
    mrr = 0.0
    for i, r in enumerate(rel):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break

    ideal = sorted(rel, reverse=True)
    ndcg = (_dcg(rel) / _dcg(ideal)) if any(ideal) else 0.0

    covered = {t for t in retrieved_topics if t in expected}
    return {
        "n_cited": len(cited),
        "n_retrieved": len(retrieved),
        "grounding_any": int(len(valid) > 0),
        "valid_source_id_rate": round(len(valid) / len(cited), 4) if cited else None,
        "invented_citation": int(any(c not in rset for c in cited)),
        # Unique expected topics covered; counting relevant chunks instead lets
        # two chunks on one topic push recall above 1.
        "recall_at_4": round(len(covered) / len(expected), 4) if expected else None,
        "precision_at_4": round(n_rel / len(retrieved), 4) if retrieved else None,
        "mrr": round(mrr, 4),
        "ndcg_at_4": round(ndcg, 4),
        "evidence_topic_coverage": round(len(covered) / len(expected), 4) if expected else None,
    }


_UNSUPPORTED_NUM = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|gram|grams|mmol|meq|mmhg|ml|litre|liter|%)\b", re.I)


def unsupported_numeric_claim(text: str, retrieved_texts: Sequence[str]) -> int:
    """A numeric clinical quantity in the answer that appears in no retrieved chunk."""
    corpus = " ".join(retrieved_texts).lower()
    for m in _UNSUPPORTED_NUM.finditer(user_facing_text(text)):
        token = m.group(0).lower().strip()
        num = re.match(r"[\d.]+", token)
        if num and num.group(0) not in corpus:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], q: float) -> float:
    v = sorted(float(x) for x in values if x is not None)
    if not v:
        return 0.0
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return v[int(k)]
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def efficiency_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def col(key: str) -> List[float]:
        return [r[key] for r in records if r.get(key) is not None]

    pt, ct, tt = col("prompt_tokens"), col("completion_tokens"), col("total_tokens")
    lat = col("latency_s")
    out = {
        "n": len(records),
        "prompt_tokens_mean": round(sum(pt) / len(pt), 2) if pt else 0.0,
        "prompt_tokens_median": round(percentile(pt, 0.5), 2),
        "prompt_tokens_p95": round(percentile(pt, 0.95), 2),
        "completion_tokens_mean": round(sum(ct) / len(ct), 2) if ct else 0.0,
        "completion_tokens_median": round(percentile(ct, 0.5), 2),
        "total_tokens_mean": round(sum(tt) / len(tt), 2) if tt else 0.0,
        "total_tokens_median": round(percentile(tt, 0.5), 2),
        "latency_median_s": round(percentile(lat, 0.5), 4),
        "latency_p95_s": round(percentile(lat, 0.95), 4),
        "outputs_per_minute": round(60.0 / (sum(lat) / len(lat)), 3) if lat and sum(lat) else 0.0,
        "peak_gpu_mem_mb_max": round(max(col("peak_gpu_mem_mb")), 1) if col("peak_gpu_mem_mb") else 0.0,
        "validator_time_ms_mean": round(
            1000 * sum(col("validator_time_s")) / len(col("validator_time_s")), 3)
            if col("validator_time_s") else 0.0,
    }
    return out
