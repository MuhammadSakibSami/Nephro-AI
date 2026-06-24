"""LLM Component — Retrieval Relevance Gate (LLM_PLAN.md Phase 3).

Completes the Phase-2/3 audit gate item: "retrieval relevance on 10 test
queries >= 80%". Runs a fixed set of representative queries through the
retriever and checks whether a relevant chunk appears in the top-k.

A query "hits" if any top-k chunk's topic OR source_id is in its expected set.

Run (after building the index):
    python -m ckd_risk_app.llm.rag.eval_retrieval
    python -m ckd_risk_app.llm.rag.eval_retrieval --k 4

Exit code 0 = pass (>=80%), 1 = fail.

NOTE on embedders: with the pure-Python hashing fallback this measures a
*lexical* baseline. The production BGE-M3 index (built in Colab) is expected to
score the same or higher (semantic matching). The gate threshold applies to the
production embedder; the local run validates the pipeline + queries.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Set

from .retriever import Retriever

# Each query: expected topics and/or source_ids that count as a relevant hit.
TEST_QUERIES: List[Dict] = [
    {"q": "high blood pressure lifestyle advice",
     "topics": {"bp"}, "sources": {"KDIGO2021BP", "KDIGO2024"}},
    {"q": "physical activity and exercise for kidney health",
     "topics": {"activity"}, "sources": {"NKFPED"}},
    {"q": "managing diabetes and high blood sugar",
     "topics": {"diabetes"}, "sources": {"ADAKDIGO2022", "ADA2026"}},
    {"q": "losing weight and obesity",
     "topics": {"weight"}, "sources": set()},
    {"q": "quitting smoking and tobacco",
     "topics": {"smoking"}, "sources": set()},
    {"q": "healthy diet and foods for kidney health",
     "topics": {"nutrition"}, "sources": {"KDOQI2020NUTR"}},
    {"q": "when should I get tested or see a doctor follow up",
     "topics": {"followup"}, "sources": {"ADA2026", "CDC_CKD_BASICS"}},
    {"q": "how to prevent chronic kidney disease",
     "topics": {"general", "followup"}, "sources": {"CDC_CKD_PREVENT", "CDC_CKD_BASICS"}},
    {"q": "reducing salt and sodium intake",
     "topics": {"bp", "nutrition"}, "sources": {"KDIGO2021BP"}},
    {"q": "risk factors for kidney disease diabetes high blood pressure",
     "topics": {"general", "bp", "diabetes"}, "sources": {"CDC_CKD_RISK"}},
]


def _is_hit(chunks: List[Dict], topics: Set[str], sources: Set[str]) -> bool:
    for c in chunks:
        if c.get("topic") in topics or c.get("source_id") in sources:
            return True
    return False


def run(k: int = 4, threshold: float = 0.80, verbose: bool = True) -> bool:
    r = Retriever.load()
    print(f"Embedder: {r.embedder.spec} | index size: {len(r.store.chunks)} | k={k}\n")

    hits = 0
    for item in TEST_QUERIES:
        chunks = r.retrieve(item["q"], k=k)
        hit = _is_hit(chunks, item["topics"], item["sources"])
        hits += hit
        if verbose:
            top = ", ".join(f"{c['topic']}/{c['source_id']}" for c in chunks[:k])
            print(f"[{'HIT ' if hit else 'MISS'}] {item['q']}")
            print(f"        top-{k}: {top}")

    rate = hits / len(TEST_QUERIES)
    passed = rate >= threshold
    print(f"\nRelevance: {hits}/{len(TEST_QUERIES)} = {rate:.0%} "
          f"(threshold {threshold:.0%}) -> {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Retrieval relevance gate (hit@k on 10 queries)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.80)
    args = p.parse_args(argv)
    return 0 if run(args.k, args.threshold) else 1


if __name__ == "__main__":
    raise SystemExit(main())
