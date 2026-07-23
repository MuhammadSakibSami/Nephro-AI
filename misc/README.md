# Documentation Index

Per-phase documentation for the CKD risk + LLM thesis project. Each phase gets
its own file (technical + plain-language explanation, how to test/run, and next
steps) for easier searching and tracking.

## Planning documents (project root)

- `PLAN.md` — overall ML project assessment + 4-phase improvement plan
- `LLM_PLAN.md` — detailed LLM component plan (11 phases, MVP-first)
- `THESIS_LLM_MVP_SUMMARY.md` — supervisor-facing summary of the thesis LLM MVP
- `frontend/` — React+Vite+Tailwind demo UI (Patient + Admin/Research views); see `frontend/README.md`

> **Thesis-first mode (June 2026):** Phases 1–6 + a minimal `/generate-guidance`
> endpoint form the thesis MVP. Phase 6 (real Qwen3) is the **frozen** LLM
> evaluation. Phases 7–9 (RAFT data / QLoRA / ablation) are **deferred to the
> paper extension** — the Phase-7 pipeline is built but not yet run at scale.

## LLM component — phase docs

| Phase | Doc | Status |
|---|---|---|
| 1 | [LLM_PHASE1_FOUNDATIONS_AND_CONTRACT.md](LLM_PHASE1_FOUNDATIONS_AND_CONTRACT.md) | ✅ Done |
| 2 | [LLM_PHASE2_KNOWLEDGE_BASE.md](LLM_PHASE2_KNOWLEDGE_BASE.md) | ✅ Done (9 sources, 3-tier safety, 118 chunks, audit passing) |
| 3 | [LLM_PHASE3_RAG_RETRIEVAL.md](LLM_PHASE3_RAG_RETRIEVAL.md) | ✅ Done (BGE-M3+FAISS / TF-IDF fallback, gate 80%) |
| 4 | [LLM_PHASE4_SAFETY_VALIDATOR.md](LLM_PHASE4_SAFETY_VALIDATOR.md) | ✅ Done (raw 78% → final 0% unsafe) |
| 5 | [LLM_PHASE5_MVP.md](LLM_PHASE5_MVP.md) | ✅ Done (runtime verified offline; Colab-ready) |
| 6 | [LLM_PHASE6_MVP_EVALUATION.md](LLM_PHASE6_MVP_EVALUATION.md) | ✅ Done & ❄️ **frozen** as thesis LLM evaluation |
| 7 | [LLM_PHASE7_RAFT_DATA.md](LLM_PHASE7_RAFT_DATA.md) | ⏸ Deferred (paper) — pipeline built, offline-validated |
| 8 | _QLoRA fine-tuning_ | ⏸ Deferred (paper) |
| 9 | _Evaluation & ablation_ | ⏸ Deferred (paper) — base arm captured |
| 10 | _Serving + app integration_ | ✅ Minimal `/generate-guidance` + local-demo done; frontend/GGUF deferred |
| 11 | _Paper write-up_ | ⏸ Deferred (paper) |

## Convention

When a phase is implemented, add a `docs/LLM_PHASE<N>_<NAME>.md` file covering:
1. What it is (plain language)
2. What was built (technical)
3. How to test / run / experiment
4. Design decisions worth remembering
5. Known open items
6. Next steps
