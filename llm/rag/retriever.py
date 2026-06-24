"""LLM Component — RAG Retriever (LLM_PLAN.md Phase 3).

Turns an ML risk-screening output into a query and returns the most relevant
guideline chunks, ready for the Phase-1 prompt builder.

Usage:
    from ckd_risk_app.llm.rag.retriever import Retriever
    r = Retriever.load()                      # loads index + matching embedder
    hits = r.retrieve("high blood pressure lifestyle advice", k=4)
    # or, straight from a /predict output:
    from ckd_risk_app.llm.serving.contract import LLMInput
    inp = LLMInput.from_predict_response(predict_response)
    chunks = r.retrieve_for_llm_input(inp, k=4)   # -> list of chunk dicts

Each returned chunk dict is compatible with
`ckd_risk_app.llm.serving.prompts.render_retrieved_chunks` (keys: chunk_id,
text, source, restricted, ...), and carries its `safety_label` so the prompt
builder surfaces restricted (dietitian-referral) chunks correctly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .embedder import get_embedder
from .vector_store import SearchHit, VectorStore

# Map ML feature names -> lay retrieval terms (enriches the query so the right
# lifestyle chunks surface). Mirrors feature_config.FEATURE_METADATA labels.
FACTOR_QUERY_TERMS: Dict[str, str] = {
    "mean_sbp": "high blood pressure",
    "mean_dbp": "high blood pressure",
    "hypertension_doctor_told": "high blood pressure hypertension",
    "taking_bp_medicine": "blood pressure",
    "hba1c": "high blood sugar diabetes",
    "glucose": "high blood sugar",
    "diabetes_yes": "diabetes blood sugar",
    "borderline_diabetes": "prediabetes blood sugar",
    "bmi": "weight obesity",
    "waist": "weight waist",
    "blood_urea_nitrogen": "kidney function diet protein",
    "uric_acid": "uric acid diet",
    "serum_albumin": "nutrition protein diet",
    "hemoglobin": "anemia nutrition",
    "sodium": "sodium salt diet",
    "potassium": "potassium diet",
    "high_cholesterol_told": "cholesterol diet",
    "taking_cholesterol_medicine": "cholesterol",
    "mean_pulse": "physical activity heart",
    "age": "",
    "sex_female": "",
    "income_poverty_ratio": "",
}

_BASE_INTENT = ("kidney health lifestyle education: diet, physical activity, "
                "blood pressure, blood sugar, weight, smoking, follow-up testing")


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store
        # Rebuild the SAME embedder the index was built with (incl. fitted state).
        self.embedder = get_embedder(store.embedder_spec)
        self.embedder.load_state(store.embedder_state)

    @classmethod
    def load(cls) -> "Retriever":
        return cls(VectorStore.load())

    # ----- query construction --------------------------------------------
    @staticmethod
    def build_query(risk_class: str, factor_features: Optional[List[str]] = None,
                    factor_phrases: Optional[List[str]] = None) -> str:
        parts = [f"{risk_class} kidney disease risk"]
        if factor_phrases:
            parts.extend(factor_phrases)
        for feat in (factor_features or []):
            term = FACTOR_QUERY_TERMS.get(feat, "")
            if term:
                parts.append(term)
        parts.append(_BASE_INTENT)
        # De-duplicate while preserving order.
        seen, uniq = set(), []
        for p in parts:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return ". ".join(uniq)

    def build_query_from_llm_input(self, inp) -> str:
        feats = [f.feature for f in inp.key_factors[:4]]
        phrases = inp.top_factor_phrases(3)
        return self.build_query(inp.risk_class, feats, phrases)

    # ----- retrieval ------------------------------------------------------
    def retrieve(self, query: str, k: int = 4,
                 drop_restricted: bool = False,
                 max_restricted: Optional[int] = None) -> List[Dict]:
        """Return up to k chunk dicts most relevant to `query`.

        drop_restricted: if True, omit restricted_clinician_referral chunks.
        max_restricted:  if set, cap how many restricted chunks are returned
                         (safe chunks fill the rest).
        """
        # Over-fetch so safety filtering can still return k results.
        raw: List[SearchHit] = self.store.search(self.embedder.embed_one(query),
                                                  k=max(k * 3, k))
        results: List[Dict] = []
        n_restricted = 0
        for hit in raw:
            c = hit.chunk
            is_restricted = bool(c.get("restricted"))
            if is_restricted and drop_restricted:
                continue
            if is_restricted and max_restricted is not None and n_restricted >= max_restricted:
                continue
            if is_restricted:
                n_restricted += 1
            results.append(self._format(hit))
            if len(results) >= k:
                break
        return results

    def retrieve_for_llm_input(self, inp, k: int = 4, **kwargs) -> List[Dict]:
        return self.retrieve(self.build_query_from_llm_input(inp), k=k, **kwargs)

    @staticmethod
    def _format(hit: SearchHit) -> Dict:
        c = hit.chunk
        return {
            "chunk_id": c.get("chunk_id"),
            "text": c.get("text"),
            "source": c.get("source_id"),
            "source_id": c.get("source_id"),
            "title": c.get("title"),
            "organization": c.get("organization"),
            "topic": c.get("topic"),
            "restricted": bool(c.get("restricted")),
            "safety_label": c.get("safety_label"),
            "score": round(float(hit.score), 4),
            "rank": hit.rank,
        }


# Module-level convenience (lazy singleton).
_DEFAULT: Optional[Retriever] = None


def get_default_retriever() -> Retriever:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever.load()
    return _DEFAULT
