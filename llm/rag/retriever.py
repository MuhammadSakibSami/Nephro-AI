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

# Map ML feature names -> knowledge-base TOPIC (see KB chunk `topic` field:
# bp / diabetes / nutrition / weight / activity / followup / smoking / sleep / general).
# Used to guarantee that the chunks retrieved for a patient actually reflect THEIR
# top risk factors, instead of collapsing to the same generic chunks for everyone
# (which happens with the weak offline TF-IDF embedder + a generic query).
FACTOR_TOPIC: Dict[str, Optional[str]] = {
    "mean_sbp": "bp",
    "mean_dbp": "bp",
    "hypertension_doctor_told": "bp",
    "taking_bp_medicine": "bp",
    "hba1c": "diabetes",
    "glucose": "diabetes",
    "diabetes_yes": "diabetes",
    "borderline_diabetes": "diabetes",
    "bmi": "weight",
    "waist": "weight",
    "blood_urea_nitrogen": "nutrition",
    "uric_acid": "nutrition",
    "serum_albumin": "nutrition",
    "hemoglobin": "nutrition",
    "sodium": "nutrition",
    "potassium": "nutrition",
    "high_cholesterol_told": "nutrition",
    "taking_cholesterol_medicine": "nutrition",
    "mean_pulse": "activity",
    "age": None,
    "sex_female": None,
    "income_poverty_ratio": None,
}

# A short, low-weight anchor. Kept brief on purpose: a long generic intent
# dominates the weak TF-IDF embedding and washes out the factor-specific terms.
_BASE_INTENT = "kidney health lifestyle education"


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

    @staticmethod
    def priority_topics(inp) -> List[str]:
        """Ordered KB topics that matter for THIS patient, from their key factors.

        Factors that *raise* the risk come first (most relevant to the education),
        then the rest, preserving the model's factor-importance order. For an
        at-risk result we always append 'followup' so a lab/clinician chunk is in
        the pool. This drives per-patient retrieval diversity even when the
        offline embedder's raw scores are weak and bunched.
        """
        supporting = [f for f in inp.key_factors if f.direction == "supports_prediction"]
        others = [f for f in inp.key_factors if f.direction != "supports_prediction"]
        topics: List[str] = []
        for f in supporting + others:
            t = FACTOR_TOPIC.get(f.feature)
            if t and t not in topics:
                topics.append(t)
        if getattr(inp, "is_at_risk", False) and "followup" not in topics:
            topics.append("followup")
        return topics

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

    def retrieve_for_llm_input(self, inp, k: int = 4,
                               drop_restricted: bool = False,
                               max_restricted: Optional[int] = None) -> List[Dict]:
        """Factor-aware retrieval: pick chunks that reflect THIS patient's top
        risk-factor topics, so different patients get different (personalized)
        guideline grounding instead of the same generic chunks every time.

        Strategy:
          1. Score a large candidate pool against the patient's query.
          2. Take the single best-scoring chunk from each of the patient's
             priority topics (BP, diabetes, nutrition, ... in importance order).
          3. Fill any remaining slots with the best-scoring leftover chunks.
        Falls back to plain top-k if the patient has no usable factor topics.
        """
        query = self.build_query_from_llm_input(inp)
        topics = self.priority_topics(inp)
        if not topics:
            return self.retrieve(query, k=k, drop_restricted=drop_restricted,
                                 max_restricted=max_restricted)

        # 1) Over-fetch a generous, score-ordered candidate pool (respecting the
        #    same restricted-chunk policy as plain retrieve()).
        pool_k = min(len(self.store.chunks), max(k * 8, 24))
        pool = self.retrieve(query, k=pool_k, drop_restricted=drop_restricted,
                             max_restricted=max_restricted)

        chosen: List[Dict] = []
        used: set = set()

        # 2) One best chunk per priority topic, in factor-importance order.
        for topic in topics:
            if len(chosen) >= k:
                break
            for c in pool:  # pool is already sorted best-first
                if c.get("chunk_id") in used:
                    continue
                if c.get("topic") == topic:
                    chosen.append(c)
                    used.add(c.get("chunk_id"))
                    break

        # 3) Fill remaining slots with the best leftover chunks overall.
        for c in pool:
            if len(chosen) >= k:
                break
            if c.get("chunk_id") not in used:
                chosen.append(c)
                used.add(c.get("chunk_id"))

        # Re-rank the final set best-first for a stable, score-ordered prompt.
        chosen.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return chosen[:k]

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
