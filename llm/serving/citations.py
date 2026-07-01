"""LLM Component — Citation layer (IEEE-style, deterministic).

Turns the retrieved-chunk provenance the RAG runtime already tracks into
per-section inline ``[n]`` markers plus an IEEE-style reference list.

Attribution is DETERMINISTIC — each output section is mapped to the KB topics it
draws on, and cited to the SOURCES ACTUALLY RETRIEVED for those topics (refined
by what the model reported in its grounding line). So a ``[n]`` never points to a
source that did not ground that section: no model-invented citations, which is
what a reviewing clinician would immediately flag.

Runs AFTER the safety validator, so markers/references never interfere with the
guardrail's detectors. Numbering follows first-citation order across sections
(standard IEEE); the reference list dedupes by source *document*.

Pure-Python (stdlib + project modules); imports without torch/transformers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..rag.ingest import load_sources

# Canonical output order (matches contract.OUTPUT_SECTION_LABELS / the frontend).
SECTION_ORDER: List[str] = [
    "Summary", "Food habits", "Rest and sleep", "Daily routine", "Follow-up reminder",
]

# Each section -> the KB `topic` values it legitimately draws on, in relevance
# order. Used to attribute retrieved chunks (which carry a `topic`) to sections.
SECTION_TOPICS: Dict[str, List[str]] = {
    "Summary": ["general", "prevention", "followup"],
    "Food habits": ["nutrition", "diabetes", "bp"],
    "Rest and sleep": ["sleep", "general", "prevention"],
    "Daily routine": ["activity", "weight", "hydration", "bp"],
    "Follow-up reminder": ["followup", "general"],
}

# Topics used to pick a fallback citation when a section has no direct match
# (e.g. Rest and sleep when no sleep chunk was retrieved).
_FALLBACK_TOPICS = ["general", "prevention", "followup"]

# Cap markers per section so the text stays readable.
MAX_REFS_PER_SECTION = 3


@lru_cache(maxsize=1)
def _source_registry() -> Dict[str, dict]:
    """source_id -> bibliographic fields, read once from sources.csv."""
    reg: Dict[str, dict] = {}
    for s in load_sources():
        reg[s.source_id] = {
            "source_id": s.source_id,
            "title": s.title,
            "organization": s.organization,
            "year": s.year,
            "url": s.url,
        }
    return reg


def format_ieee(rec: dict) -> str:
    """IEEE-style reference string (without the URL; URL is returned separately).

    Guideline/patient-education sources are organizational-author works, so the
    organization stands in for the author list:
        ORG, "Title," Year.
    """
    org = (rec.get("organization") or "").strip()
    title = (rec.get("title") or "").strip()
    year = (rec.get("year") or "").strip()
    parts: List[str] = []
    if org:
        parts.append(f"{org},")
    if title:
        parts.append(f'"{title},"')
    if year:
        parts.append(f"{year}.")
    return " ".join(parts).strip()


def _section_source_ids(section_topics: List[str], retrieved: List[dict],
                        model_cited_ids: set) -> List[str]:
    """Distinct source_ids grounding a section, best-first.

    Ranked by: (a) whether the model actually cited the chunk, (b) topic priority
    within the section, (c) retrieval score.
    """
    ranked: List[Tuple[int, int, float, str]] = []
    for c in retrieved:
        topic = c.get("topic")
        if topic in section_topics:
            cited = 0 if c.get("chunk_id") in model_cited_ids else 1
            prio = section_topics.index(topic)
            score = float(c.get("score") or 0.0)
            sid = c.get("source_id")
            if sid:
                ranked.append((cited, prio, -score, sid))
    ranked.sort()
    out: List[str] = []
    for _, _, _, sid in ranked:
        if sid not in out:
            out.append(sid)
        if len(out) >= MAX_REFS_PER_SECTION:
            break
    return out


def _fallback_source_ids(retrieved: List[dict]) -> List[str]:
    """Best general-topic retrieved source, else the single best-scored source."""
    def best(pred) -> Optional[str]:
        best_sid, best_score = None, float("-inf")
        for c in retrieved:
            if pred(c):
                sc = float(c.get("score") or 0.0)
                if sc > best_score:
                    best_sid, best_score = c.get("source_id"), sc
        return best_sid

    sid = best(lambda c: c.get("topic") in _FALLBACK_TOPICS) or best(lambda c: True)
    return [sid] if sid else []


def build_citations(
    sections: Dict[str, str],
    retrieved_chunks: Optional[List[dict]],
    model_cited_grounding: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], List[dict]]:
    """Annotate sections with ``[n]`` markers and build the IEEE reference list.

    Returns (annotated_sections, references) where references is
    [{n, source_id, citation, organization, title, year, url}, ...] ordered by n.
    Never raises; degrades to (sections unchanged, []) if there is nothing to cite.
    """
    retrieved = retrieved_chunks or []
    if not sections or not retrieved:
        return dict(sections or {}), []

    registry = _source_registry()
    model_cited_ids = set(model_cited_grounding or [])

    # 1) Determine the source_ids grounding each section (with fallback).
    per_section: Dict[str, List[str]] = {}
    for label in SECTION_ORDER:
        if label not in sections:
            continue
        sids = _section_source_ids(SECTION_TOPICS.get(label, []), retrieved, model_cited_ids)
        if not sids:
            sids = _fallback_source_ids(retrieved)
        per_section[label] = [s for s in sids if s in registry]

    # 2) Assign global reference numbers in first-citation order.
    number_of: Dict[str, int] = {}
    for label in SECTION_ORDER:
        for sid in per_section.get(label, []):
            if sid not in number_of:
                number_of[sid] = len(number_of) + 1

    # 3) Append inline markers to each section's text.
    annotated = dict(sections)
    for label, sids in per_section.items():
        nums = sorted({number_of[sid] for sid in sids})
        text = (annotated.get(label) or "").rstrip()
        if nums and text:
            annotated[label] = text + " " + ", ".join(f"[{n}]" for n in nums)

    # 4) Build the ordered reference list.
    references: List[dict] = []
    for sid, n in sorted(number_of.items(), key=lambda kv: kv[1]):
        rec = registry[sid]
        references.append({
            "n": n,
            "source_id": sid,
            "citation": format_ieee(rec),
            "organization": rec.get("organization", ""),
            "title": rec.get("title", ""),
            "year": rec.get("year", ""),
            "url": rec.get("url", ""),
        })
    return annotated, references
