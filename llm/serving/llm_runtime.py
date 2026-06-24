"""LLM Component — Inference Runtime / MVP (LLM_PLAN.md Phase 5).

Chains the four prior phases into one call:

    /predict output
      -> LLMInput            (Phase 1 contract)
      -> Retriever           (Phase 3: query + top-k guideline chunks)
      -> build_messages      (Phase 1 prompt)
      -> model.generate      (Phase 5: base Qwen3, or offline mock)
      -> safety_validator    (Phase 4: enforce -> guaranteed-safe output)
      -> GuidanceResult      (safe text + provenance + audit trail)

This is the MVP: it uses the BASE (un-fine-tuned) Qwen3-4B so Phase 6 can measure
whether fine-tuning is needed. Phase 8 swaps in the fine-tuned model with no other
change.

Usage (Colab / GPU):
    from ckd_risk_app.llm.serving.llm_runtime import GuidanceRuntime
    rt = GuidanceRuntime(model="auto")          # Qwen3 if available
    result = rt.generate_guidance(predict_response)
    print(result.safe_text)

Offline (no torch): pass model="mock" to exercise the full pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..rag.retriever import Retriever
from .contract import GuidanceOutput, LLMInput, parse_guidance
from .model_backends import get_model
from .prompts import build_messages
from .safety_validator import (
    Violation, context_from_llm_input, enforce,
)

_RETRY_NOTE = ("\n\n(Reminder: reply using the EXACT output format with all five "
               "labeled sections. Do not diagnose, name or change any medication, "
               "or give numeric nutrient targets.)")


@dataclass
class GuidanceResult:
    risk_class: str
    status: str                       # pass | sanitized | fallback
    safe_text: str
    sections: Dict[str, str]
    provenance: List[str]             # source organizations (user-facing)
    retrieved_chunk_ids: List[str]    # what was given to the model
    model_cited_grounding: List[str]  # what the model claimed it used (validated)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    model_spec: str = ""
    raw_model_text: str = ""
    n_attempts: int = 1
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)  # admin: id/source/restricted
    grounding_valid_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_class": self.risk_class,
            "status": self.status,
            "guidance": self.sections,
            "safe_text": self.safe_text,
            "provenance": self.provenance,
            "grounding": self.model_cited_grounding,
            "grounding_valid_ratio": self.grounding_valid_ratio,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_chunks": self.retrieved_chunks,
            "violations": self.violations,
            "query": self.query,
            "n_attempts": self.n_attempts,
            "raw_guidance": self.raw_model_text,
            "model": self.model_spec,
        }


class GuidanceRuntime:
    def __init__(self, model: str = "auto", k: int = 4, max_retries: int = 1,
                 log_path: Optional[Union[str, Path]] = None,
                 retriever: Optional[Retriever] = None, model_kwargs: Optional[Dict] = None):
        self.retriever = retriever or Retriever.load()
        self.model = get_model(model, **(model_kwargs or {}))
        self.k = k
        self.max_retries = max_retries
        self.log_path = Path(log_path) if log_path else None

    def generate_guidance(self, predict_response: Union[Dict[str, Any], LLMInput],
                          k: Optional[int] = None) -> GuidanceResult:
        inp = (predict_response if isinstance(predict_response, LLMInput)
               else LLMInput.from_predict_response(predict_response))

        chunks = self.retriever.retrieve_for_llm_input(inp, k=(k or self.k))
        retrieved_ids = [c["chunk_id"] for c in chunks]
        context = context_from_llm_input(inp)
        query = self.retriever.build_query_from_llm_input(inp)

        captured: Dict[str, Any] = {"raw": "", "attempts": 0}

        def _generate(attempt: int) -> str:
            captured["attempts"] += 1
            messages = build_messages(inp, chunks)
            if attempt > 0:
                messages[-1]["content"] += _RETRY_NOTE
            raw = self.model.generate(messages)
            captured["raw"] = raw
            return raw

        result = enforce(_generate, context, max_retries=self.max_retries,
                         log_path=self.log_path)

        parsed: GuidanceOutput = result.parsed or parse_guidance(result.safe_text)
        valid_grounding = [g for g in parsed.grounding if g in set(retrieved_ids)]
        provenance = _provenance(chunks, retrieved_ids)
        grounding_ratio = (round(len(valid_grounding) / len(parsed.grounding), 3)
                           if parsed.grounding else 0.0)
        retrieved_light = [{
            "chunk_id": c.get("chunk_id"), "source_id": c.get("source_id"),
            "organization": c.get("organization"), "topic": c.get("topic"),
            "restricted": bool(c.get("restricted")), "score": c.get("score"),
        } for c in chunks]

        return GuidanceResult(
            risk_class=inp.risk_class,
            status=result.status,
            safe_text=result.safe_text,
            sections=parsed.user_facing_sections(),
            provenance=provenance,
            retrieved_chunk_ids=retrieved_ids,
            model_cited_grounding=valid_grounding,
            violations=[v.__dict__ if isinstance(v, Violation) else v for v in result.violations],
            query=query,
            model_spec=self.model.spec,
            raw_model_text=captured["raw"],
            n_attempts=captured["attempts"],
            retrieved_chunks=retrieved_light,
            grounding_valid_ratio=grounding_ratio,
        )


def _provenance(chunks: List[Dict[str, Any]], retrieved_ids: List[str]) -> List[str]:
    """User-facing source organizations (e.g. 'KDIGO', 'CDC') for a provenance line."""
    orgs = []
    for c in chunks:
        org = (c.get("organization") or c.get("source_id") or "").split("(")[0].strip()
        if org and org not in orgs:
            orgs.append(org)
    return orgs
