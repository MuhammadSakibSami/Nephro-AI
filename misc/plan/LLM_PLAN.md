# LLM Component — Implementation Plan

**Project:** Non-invasive kidney disease (CKD) risk screening — guideline-grounded kidney-risk education
**Status:** Thesis MVP complete (Phases 1–6 + minimal serving). Fine-tuning track deferred to paper extension.
**Goal:** A RAG-grounded, safety-guarded local LLM that turns the ML model's risk output into safe, personalized, guideline-grounded kidney-risk **education** — and (for the paper) a fine-tuned model + rigorous ablation.

> **⚑ THESIS-FIRST MVP MODE (June 2026).** Priority is the thesis + supervisor demo, so the project is split:
> - **Thesis MVP (DONE):** Phases 1–6 + a minimal `/generate-guidance` endpoint with a local-demo fallback. **Phase 6 (real Qwen3 + BGE-M3 + validator) is FROZEN as the thesis LLM evaluation result.**
> - **Paper extension (IN PROGRESS on a local GPU box):** Phase 7 (RAFT data — **done**, 540/60 train/val), Phase 8 (LoRA fine-tuning — **done**, adapter trained + acceptance-checked; trained with `--precision bf16`, i.e. an unquantized bf16 base, not the QLoRA/4-bit default — the 24 GB training GPU didn't need the extra memory savings), Phase 9 (ablation — **done** on automated axes; human/expert review still open). The original blockers (free-tier API quota + Qwen self-distillation CUDA OOM on Colab) are resolved by local GPU compute + 100% self-distillation. **Not required for the thesis.** The codebase is extensible throughout (pluggable model/teacher backends, `adapter_path` hook, frozen eval profiles).
> - See `THESIS_LLM_MVP_SUMMARY.md` for the supervisor-facing summary.

> **Scope statement (use this wording in app + paper):**
> *The LLM component provides personalized, guideline-grounded kidney-risk education after ML-based screening. It does not diagnose CKD, prescribe treatment, or replace clinician consultation.*
>
> Avoid the terms "CKD lifestyle counseling," "CKD treatment recommendation," "medical counseling." The model does not produce a confirmed CKD diagnosis; it produces safe, educational guidance derived from a KDIGO-aligned risk-screening output.

---

## 0. Architecture Overview

### 0.1 Final architecture (frozen)

```
Patient Inputs
   ↓
Existing Kidney-Risk ML Model  (FastAPI /predict)
   ↓
Risk Output JSON
   - risk_class
   - probabilities
   - reliability / confidence
   - input completeness
   - top model signals (key_model_factors)
   ↓
Risk-to-RAG Query Builder
   ↓
Guideline RAG  (BGE-M3 embeddings + FAISS/Chroma)
   - KDIGO 2024 CKD            (primary)
   - KDIGO 2021 BP in CKD      (primary)
   - KDIGO 2022 Diabetes in CKD(primary)
   - ADA Standards of Care 2026, CKD chapter (primary, diabetes/glucose)
   - KDOQI 2020 Nutrition      (restricted — dietitian-referred wording only)
   - CDC/NKF patient education (patient-friendly language)
   - KDIGO 2012                (background only — target-label justification)
   ↓
Fine-tuned Qwen3-4B-Instruct-2507  (LoRA; served via Ollama / llama.cpp GGUF)
   ↓
Structured Draft Guidance  (evidence-cited, NO chain-of-thought)
   ↓
Safety Validator  (rule-based + LLM-judge → pass / fix / block / safe-fallback)
   ↓
Safe Personalized Kidney-Risk Education  →  /generate-guidance API  →  Web frontend
```

### 0.2 Final stack

| Layer | Choice | Rationale |
|---|---|---|
| Base LLM | **Qwen3-4B-Instruct-2507** | Strong instruction following at 4B; fine-tunable on free/low-cost Colab; permissive license |
| Fine-tuning | **LoRA** adapters; the training script supports either a 4-bit NF4 (QLoRA) or bf16 (unquantized) frozen base via `--precision` — **the actual run used bf16** (24 GB GPU) | 4-bit fits in ≤16 GB VRAM if needed; only adapter weights are trained either way |
| Fine-tuning method | **RAFT** (Retrieval-Augmented Fine-Tuning), **structured output, no CoT** | Model learns to ground in retrieved guidelines + ignore distractors — but emits a fixed evidence-cited format, not visible medical reasoning |
| Embeddings | **BGE-M3** (`BAAI/bge-m3`) | Top-tier retrieval, multilingual, long passages |
| Vector store | **FAISS** (primary) / Chroma (fallback) | FAISS = fast, file-based; Chroma = easier metadata filtering |
| Serving | **Ollama** (primary) / llama.cpp GGUF | Simple local serving; no external API at inference |
| API | **FastAPI** `/generate-guidance` | Extends existing backend |
| Guardrail | Rule-based + LLM-judge validator | Medical-safety is non-negotiable |
| Training env | Google Colab (T4 free / A100 Pro) | Matches existing workflow |

### 0.3 The USPs for the paper

1. **Domain-specialized fine-tuning** of a small open model for guideline-grounded kidney-risk education (vs. a generic API call) — reproducible, self-hosted, no per-call cost, privacy-preserving.
2. **RAFT grounding + strict post-hoc safety validation** — trained to cite guidelines, verified never to diagnose, prescribe, or give individualized nutrition restrictions. This is the safety story reviewers will want.

> **Teacher-model transparency (state explicitly in the paper):** A stronger teacher model may be used **only** for synthetic training-data generation. **No external API is used during final inference or deployment** — deployment runs the local fine-tuned model. Only **synthetic / de-identified** NHANES-style profiles are sent to the teacher; **no real user data**.

### 0.4 Why "CKD" sources are correct (settled)

The target variable is the KDIGO CKD risk grid; the model predicts KDIGO-defined CKD risk levels in a general-population screening setting. The knowledge base uses the **prevention / lifestyle / early-stage** sections of CKD guidelines and **excludes** advanced-CKD content (dialysis, transplant, medication dosing). See Phase 2 for the source hierarchy and inclusion/exclusion list.

---

## 1. Repository Layout (to be created)

```
ckd_risk_app/
  llm/
    knowledge_base/
      raw/                  # source PDFs/HTML of guidelines
      processed/            # cleaned text + chunk JSONL
      index/                # FAISS index + metadata
    rag/
      ingest.py             # PDF → clean text → chunks → JSONL
      embed_index.py        # chunks → BGE-M3 embeddings → FAISS
      retriever.py          # query → top-k chunks (reusable module)
    data/
      profiles.py           # build profiles from TRAIN/VAL distribution only
      generate_dataset.py   # RAFT synthetic data via teacher model
      train.jsonl           # final SFT training set
      val.jsonl
      eval_profiles.jsonl   # held-out profiles for evaluation (never trained on)
    finetune/
      qlora_train.py        # LoRA SFT on Qwen3-4B (4-bit/QLoRA or bf16 base via --precision)
      merge_and_export.py   # merge adapter → GGUF → quantize
      adapters/             # saved LoRA adapters
    serving/
      modelfile             # Ollama Modelfile
      llm_runtime.py        # load model, RAG, build prompt, call model
      safety_validator.py   # post-generation guardrail (strict)
      prompts.py            # system prompt + templates + output schema
    eval/
      eval_safety.py        # adversarial safety test (before/after validator)
      eval_quality.py       # relevance, readability, groundedness
      eval_ablation.py      # base vs finetuned, RAG vs no-RAG
      eval_human.py         # human-review harness + rubric export
      results/              # tables + figures for paper
  backend/
    main.py                 # + new /generate-guidance endpoint
    llm_endpoint.py         # wires llm_runtime into FastAPI
```

---

## 2. Implementation Order (MVP-first)

Build a **base Qwen3 + RAG + guardrail MVP first**, evaluate it, *then* fine-tune. This gives a go/no-go signal on whether fine-tuning is needed and turns "base vs fine-tuned" into a clean, pre-planned ablation.

1. **Phase 1** — Foundations: data contract, prompt, output schema, safety-rule spec
2. **Phase 2** — Knowledge base construction (revised source hierarchy)
3. **Phase 3** — RAG retrieval pipeline (BGE-M3 + FAISS)
4. **Phase 4** — Safety validator (strict guardrail)
5. **Phase 5** — MVP: base Qwen3 + RAG + guardrail (inference runtime)
6. **Phase 6** — MVP evaluation on ~30 profiles (go/no-go for fine-tuning)
7. **Phase 7** — RAFT training-data generation
8. **Phase 8** — LoRA fine-tuning
9. **Phase 9** — Evaluation & ablation (incl. human/expert review)
10. **Phase 10** — Serving (GGUF/Ollama) + app/frontend integration
11. **Phase 11** — Paper write-up

---

## 3. Phase Plan

Each phase: **Objective → Steps → Deliverable → Test/acceptance → Risks.**

---

### PHASE 1 — Foundations & Data Contract  ✅ DONE

**Objective:** Lock the input/output contract and the canonical prompt + safety rules.

**Implemented in:**
- `ckd_risk_app/llm/serving/contract.py` — input contract (`LLMInput.from_predict_response` adapts the existing `/predict` `llm_context`), output contract (`GuidanceOutput`, `parse_guidance`, structural validation, `RISK_CLASSES`, `OUTPUT_SECTIONS`, `READABILITY_TARGET_GRADE`).
- `ckd_risk_app/llm/serving/prompts.py` — canonical `build_system_prompt()`, `SAFETY_RULES`, `SCOPE_STATEMENT`, fixed output-format block, `build_user_prompt()` / `build_messages()` (Qwen3 chat format), KDOQI `[RESTRICTED]` chunk handling.
- `ckd_risk_app/llm/demo_phase1.py` — acceptance smoke test (mock `/predict` → prompt → output round-trip). **Status: PASSED.**

**Teacher-model decision (Phase 1, step 4): ⏳ PENDING USER DECISION** — needed before Phase 7 (data generation), not before MVP. Options: a hosted large model (e.g. Claude/GPT-4-class) or a larger hosted Qwen variant. Whatever is chosen, record it here + the §0.3 transparency statement (teacher used only for synthetic data; no external API at inference; only de-identified profiles sent).

**Steps:**
1. Freeze the **LLM input schema** = the existing `llm_context` block from `model_service._build_llm_context` (`model_output`, `key_model_factors`, `patient_inputs`, `input_quality`, `strict_safety_rules`, `suggestion_style`).
2. Define the **LLM output schema** — fixed, evidence-cited, **no chain-of-thought**:
   ```
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
   ```
   `Grounding used internally` is **internal metadata** (for logging + groundedness eval), **not shown verbatim to the user**. The frontend shows only a general provenance line (e.g. "based on KDIGO/ADA guidelines").
3. Write `llm/serving/prompts.py` as the single source of truth: system prompt (role = kidney-risk **education**, not counseling/diagnosis/treatment), the strict safety rules, the fixed output format, readability target ≤ grade 8.
4. Record the **teacher-model decision** here (hosted large model or larger Qwen variant) plus the transparency statement from §0.3.

**Deliverable:** `prompts.py`, documented input/output JSON schema, teacher decision recorded.

**Test:** Paste a real `/predict` `llm_context` into the template → confirms a coherent, complete prompt that requests the exact output schema.

**Risks:** Schema drift → import shared schema constants everywhere.

---

### PHASE 2 — Knowledge Base Construction  ✅ DONE (9 sources, 3-tier safety, audit passing)

**Objective:** Build a clean, scoped, **safety-audited** corpus with an explicit source hierarchy.

**Implemented in:**
- `ckd_risk_app/llm/knowledge_base/sources.csv` — registry (`priority`, `restricted`, `include_in_retrieval`, `scope`, `page_ranges`, topics, citation, `status`).
- `ckd_risk_app/llm/rag/ingest.py` — pipeline: PDF page-range extract → clean → chunk (~150 words) → lifestyle filter + topic inference → **3-tier safety classification** → `chunks.jsonl`.
- `ckd_risk_app/llm/rag/audit.py` — **independent safety gate** over the final chunks; must pass before freezing.
- Sources (9 retrieved): KDIGO 2024, KDIGO 2021 BP, ADA/KDIGO 2022 consensus, **ADA 2026 Standards Ch.11**, KDOQI 2020 (restricted), **cleaned NKF exercise .md**, **CDC Basics/Prevent/Risk**. Plus KDIGO 2012 (background) + legacy CDC stub.
- Output: **118 chunks** — safe_public=86, restricted=32; **75 clinical chunks dropped** (medication/dialysis/transplant/dosing/BP-target). Audit: **PASSED (0 leakage)**.
- Docs: `docs/LLM_PHASE2_KNOWLEDGE_BASE.md`.

**Three-tier safety model** (per reviewer feedback): `exclude_from_retrieval` (medication, drug dosing, dialysis, transplant, direct BP targets → **dropped entirely**) > `restricted_clinician_referral` (KDOQI + numeric nutrition targets → kept, dietitian-referral only) > `safe_public` (general lifestyle/prevention). Small ~150-word chunks keep exclusion surgical.

**Notes:** ADA 2026 (`dc26s011.pdf`) and the ADA/KDIGO 2022 consensus are both included (different documents). NKF switched to a hand-cleaned markdown (dialysis/transplant removed). CDC sources are public-domain patient education.

**Source hierarchy:**

| Priority | Source | Use |
|---|---|---|
| **Primary** | KDIGO 2024 CKD guideline | CKD evaluation, risk framework, lifestyle/risk-factor management |
| **Primary** | KDIGO 2021 BP in CKD | BP / activity / sodium-related kidney-risk guidance |
| **Primary** | KDIGO 2022 Diabetes in CKD | diabetes + kidney-risk context |
| **Primary** | ADA Standards of Care 2026, CKD & Risk Management chapter | diabetic users / glucose-related risk education |
| **Secondary / restricted** | NKF KDOQI 2020 Nutrition in CKD | nutrition topics — **only with clinician/dietitian-referred wording** |
| **Patient-friendly** | CDC / NKF patient education | simple language, general prevention |
| **Background only** | KDIGO 2012 | target-label justification, **not** a recommendation source |

**Sources / sections to EXCLUDE:** dialysis initiation/management, transplant, phosphate binders, ESA/IV iron, medication dosing, late-stage (G4/G5) management.

**KDOQI nutrition handling (critical):** The LLM must **not** issue direct nutrition restrictions ("eat X g protein," "cut potassium," "restrict phosphorus"). Per KDIGO 2024, dietary changes to sodium/phosphorus/potassium/protein must be individualized by CKD severity, comorbidities, and a renal dietitian. KDOQI-derived chunks must be tagged `restricted=true` and only ever surface as *"discuss your diet with a clinician or renal dietitian"* phrasing.

**Steps:**
1. Collect sources to `knowledge_base/raw/`; record citation + URL + access date per source (reproducibility + `sources.csv`).
2. `ingest.py`: extract text (PyMuPDF / pdfplumber), strip headers/footers/TOC, section-aware splitting.
3. **Manual scoping pass:** keep only included sections; tag each with `source`, `guideline`, `topic` (diet/BP/glucose/activity/general), `priority`, `restricted` (bool), `risk_relevance`.
4. Chunk: recursive/semantic, ~300–500 tokens, ~50-token overlap → JSONL with `chunk_id`, `text`, metadata.

**Deliverable:** `processed/chunks.jsonl` (~200–600 chunks) + `sources.csv`.

**Test:** Spot-check 20 chunks — on-topic, self-contained, no excluded content; all KDOQI chunks carry `restricted=true`.

**Risks:** Copyright/licensing → keep raw docs local; release only cited chunk text per each guideline's reuse terms.

---

### PHASE 3 — RAG Retrieval Pipeline  ✅ DONE (gate passing)

**Objective:** Given an ML risk output, retrieve the most relevant guideline chunks.

**Implemented in:**
- `ckd_risk_app/llm/rag/embedder.py` — `BGEM3Embedder` (production) + `HashingTfidfEmbedder`/`HashingEmbedder` (pure-Python offline fallback); shared interface persists embedder `spec` + fitted state in the index.
- `ckd_risk_app/llm/rag/vector_store.py` — index save/load; **FAISS** `IndexFlatIP` when available, else pure-Python brute-force cosine.
- `ckd_risk_app/llm/rag/embed_index.py` — `chunks.jsonl` → embeddings → `knowledge_base/index/`. Run: `python -m ckd_risk_app.llm.rag.embed_index` (auto BGE-M3; `--embedder hashing-tfidf` offline).
- `ckd_risk_app/llm/rag/retriever.py` — `Retriever.load()`, driver-aware query builder (`build_query_from_llm_input`), safety-aware `retrieve(query, k, drop_restricted, max_restricted)`; output plugs into the Phase-1 prompt builder.
- `ckd_risk_app/llm/rag/eval_retrieval.py` — 10-query hit@k relevance gate.
- Docs: `docs/LLM_PHASE3_RAG_RETRIEVAL.md`.

**Verified:** relevance gate **8/10 = 80% PASS** (local TF-IDF fallback; conservative lower bound — production BGE-M3 expected ≥). End-to-end integration with the Phase-1 contract + prompt builder confirmed; restricted chunks surface labeled; excluded content unreachable.

**Open:** run the production gate with **BGE-M3 + FAISS in Colab** and record that number for the paper. The `index/` artifacts are regenerable from `chunks.jsonl`.

**Steps:**
1. `embed_index.py`: embed chunks with **BGE-M3**, build **FAISS** index (cosine/IP on normalized vectors); persist index + id→metadata map.
2. `retriever.py`: reusable `retrieve(query, k)`. Build the query from the ML output — risk class + top factors mapped to human terms via `feature_config.FEATURE_METADATA` (`mean_sbp`→"blood pressure", `hba1c`→"blood sugar/diabetes", `bmi`→"weight", `uric_acid`→"uric acid", …).
3. Light **metadata filtering**: prefer `topic` matching driving factors; demote/handle `restricted` chunks specially.
4. Tune `k` (start k=4) + a max-context budget so the prompt stays small for a 4B model.

**Deliverable:** `retriever.py` usable by both data-gen and serving.

**Test:** 10 hand-written queries → top-k topically correct (manual relevance + hit@k table).

**Risks:** BGE-M3 is large → batch-embed on T4, cache embeddings.

---

### PHASE 4 — Safety Validator (Strict Guardrail)  ✅ DONE (final unsafe 0%)

**Objective:** Guarantee no unsafe content reaches the user; produce the metric that proves the guardrail's value.

**Implemented in:**
- `ckd_risk_app/llm/serving/safety_validator.py` — `validate()` / `enforce()` / `scan_unsafe()` / `fallback_text()` / `build_judge_prompt()`. Three-severity model: severe (diagnosis, medication, supplement, broken schema → fallback) vs moderate (nutrition restriction → rewrite; missing lab reminder → append; emergency → add urgent-care line). Reuses Phase-2 clinical vocabulary; optional LLM-judge hook.
- `ckd_risk_app/llm/serving/contract.py` — added `GuidanceOutput.render()` + `with_sections()`.
- `ckd_risk_app/llm/demo_phase4.py` — adversarial acceptance test.
- Docs: `docs/LLM_PHASE4_SAFETY_VALIDATOR.md`.

**Verified:** adversarial set **raw unsafe 78% → final unsafe 0%**; no false positives on negation/risk-language; `enforce` ends safe under a flaky generator; fallback templates safe at every risk level. Key fix: nutrition rewrite runs *before* severe detection so numeric nutrient targets (e.g. "potassium 2000 mg/day") aren't misread as medication dosing.

**Open:** wire the LLM-judge layer + enable decision logging in Phase 5/9; validator precision/recall is a Phase 9 metric.

**Guardrail layers:**

| Guardrail | Action |
|---|---|
| JSON/schema validator | Missing/extra section → retry generation |
| Diagnosis blocker | Block direct claims ("you have CKD," "kidney failure") |
| Medication blocker | Block start/stop/change/dose of any medicine |
| Nutrition-restriction blocker | Block direct potassium/phosphorus/protein/sodium restriction advice → rewrite to "discuss with clinician/dietitian" |
| Lab-confirmation reminder | Moderate/high risk → ensure a "repeat test / see clinician" reminder is present |
| Emergency-symptom redirect | Chest pain, severe swelling, breathlessness, extreme BP (from symptom input or extreme vital values) → urgent-care message |
| Safe fallback template | Repeated failure → return a generic safe educational message |

**Implementation:** `safety_validator.validate(text, context) → {status, reasons, safe_text}`.
- **Rule layer:** regex/keyword lists (medications, dosages "mg"/"take…daily", diagnosis phrasing, supplements, nutrition-restriction verbs).
- **Judge layer:** LLM-judge prompt scoring the draft for violations.
- **Decision:** pass → return; minor → strip/rewrite; severe → block + regenerate once → else safe fallback.
- **Logging:** every decision + reason (feeds eval).

**Two evaluation metrics to record:**
1. **Raw unsafe generation rate** (before validator)
2. **Final unsafe output rate** (after validator)

**Deliverable:** `safety_validator.py` + decision logs.

**Test:** Adversarial set (Phase 9) → **0 severe violations** in final output; report validator precision/recall (watch over-blocking).

**Risks:** Over-blocking false positives → tune thresholds; report them.

---

### PHASE 5 — MVP: Base Qwen3 + RAG + Guardrail  ✅ DONE (runtime verified offline)

**Objective:** End-to-end working pipeline using the **un-fine-tuned** Qwen3-4B.

**Implemented in:**
- `ckd_risk_app/llm/serving/model_backends.py` — `QwenChatModel` (Qwen3-4B-Instruct-2507; `adapter_path` hook for Phase 8) + `MockGuidanceModel` (deterministic offline) + `get_model()`.
- `ckd_risk_app/llm/serving/llm_runtime.py` — `GuidanceRuntime.generate_guidance()` chains LLMInput → retriever → prompt → model (`enforce` retry) → validator → `GuidanceResult` (safe_text, sections, provenance, validated grounding, violations).
- `ckd_risk_app/llm/demo_phase5.py` — 5 end-to-end cases + unsafe-first validator-integration check.
- Docs: `docs/LLM_PHASE5_MVP.md` (incl. Colab Qwen3 instructions).

**Verified (offline mock):** all cases complete + safe + grounded; emergency case sanitized; unsafe-first model recovered to safe; returned text always passes `scan_unsafe`. Production swap is one flag (`model="auto"` → Qwen3 in Colab).

**Open:** run the real Qwen3 MVP in Colab and capture outputs + retry/fallback rate for Phase 6.

**Steps:**
1. `llm_runtime.py`: `generate_guidance(predict_output)` → build query → `retriever.retrieve` → assemble prompt (system + ML output + chunks) → call base Qwen3 → parse structured output → run `safety_validator` → attach internal grounding.
2. Decoding tuned for safety: low temperature (0.2–0.4), capped max tokens, schema-parse + one auto-retry.

**Deliverable:** Working MVP callable (no API/frontend yet).

**Test:** 5 real `/predict` outputs end-to-end → correct shape, grounded, safe.

**Risks:** Output-format drift on base model → strict parse + retry; this is partly *why* we may fine-tune.

---

### PHASE 6 — MVP Evaluation (Go/No-Go)  ✅ DONE & ❄️ FROZEN as the thesis LLM evaluation result

**Objective:** Decide whether fine-tuning is justified, with evidence.

**RESULT (real Qwen3-4B-Instruct-2507 + BGE-M3 + guardrail, 30 held-out profiles), in `ckd_risk_app/results/`:**
- format_adherence **1.00**, final_unsafe **0.00**, fallback **0.00**, retry **0.00**.
- factor_relevance **0.966**, personalization **1.93/2**, grounding_any **1.00** (valid_ratio 1.00), readability **6.9** (100% ≤ grade 8).
- raw_unsafe **0.10 (3/30)** — but all 3 are validator-handled, NOT dangerous model failures: H02/H07 = emergency redirect not self-added (the model is never told BP is crisis-level); H09 = a soft "avoid high protein" phrase rewritten to dietitian referral. Zero diagnoses, zero medications, zero dosing.
- **Interpretation:** base + RAG + guardrail is already near-publishable. The auto "GO" trigger overstates a real model weakness.

**Decision (chosen): GO — fine-tune, but REFRAMED.** Not a quality overhaul (base is near-ceiling). Fine-tuning targets: (1) **safety self-containment** — internalize the rules the validator now patches so raw_unsafe → ~0 without the validator; (2) **consistency** — remove rare misses + over-specific numbers; (3) **prompt compression / cheaper inference**. This preserves the thesis USP with an honest, measurable objective. The 30 Phase-6 outputs are frozen as the **base arm** of the Phase 9 ablation.

**Implemented in:**
- `ckd_risk_app/llm/data/profiles.py` — 30 held-out `/predict`-shaped profiles (10 low / 10 moderate / 10 high; mixed BP/glucose/weight/labs; 2 emergency-BP). Saved to `data/eval_profiles.jsonl`. Kept separate from Phase-7 training profiles.
- `ckd_risk_app/llm/eval/score.py` — per-output scoring: format, raw/final unsafe, factor relevance, grounding, readability (textstat/estimate), personalization; `aggregate()`.
- `ckd_risk_app/llm/eval/run_phase6.py` — runs MVP over profiles → `phase6_results.csv` + `phase6_summary.md` (metrics, auto-selected good/bad/validator-fixed examples, **data-driven go/no-go** with named weaknesses).
- `notebooks/phase6_mvp_evaluation.ipynb` — Colab: BGE-M3+FAISS index + retrieval gate + real Qwen3 eval + downloads.
- `GuidanceResult.n_attempts` added for retry tracking. Docs: `docs/LLM_PHASE6_MVP_EVALUATION.md`.

**Artifacts:** `ckd_risk_app/results/phase6_results.csv`, `ckd_risk_app/results/phase6_summary.md`.

**Steps:**
1. Run MVP on ~30 held-out profiles.
2. Score: format adherence, safety (raw vs post-validator), factor relevance, readability, groundedness.
3. Decision: if base+RAG already saturates quality/safety → fine-tuning becomes a smaller refinement; if it shows gaps (format drift, weak grounding, safety misses) → fine-tuning is justified and the gaps define the training objective.

**Deliverable:** MVP baseline table (becomes the "base" column of the Phase 9 ablation).

**Test:** Documented go/no-go with numbers.

**Risks:** None major — this de-risks everything downstream.

---

### PHASE 7 — Training-Data Generation (self-distillation + teacher subset)  ⏸ DEFERRED — paper extension (pipeline built)

> **Not required for the thesis MVP.** Pipeline is built + offline-validated but the real Colab run is deferred (free-tier Gemini quota + Qwen self-distillation CUDA OOM). Resume when API/compute allow.

**Teacher decision (resolved):** **Gemini free tier** (`gemini-2.5-flash`) for the ~20% subset; self-distillation for the bulk. No paid API available now; pluggable so a paid teacher is a one-flag swap later. Only synthetic profiles sent to Gemini.

**Implemented in:**
- `ckd_risk_app/llm/serving/prompts.py` — `build_compact_system_prompt()` / `build_compact_messages()` (compact prompt for training/inference).
- `ckd_risk_app/llm/data/train_profiles.py` — `generate_train_profiles(n, seed)` (training-distribution profiles, ~1/3 per class, ~20% high are emergency-BP).
- `ckd_risk_app/llm/data/teacher_backends.py` — `SelfDistillTeacher` + `GeminiTeacher` + `get_teacher()`; both routed through `safety_validator.enforce`.
- `ckd_risk_app/llm/data/generate_dataset.py` — retrieve golden + 1–3 distractors → teacher → **hard gate (scan_unsafe=0, 5 sections, mentions risk)** → Unicode-normalize → `train.jsonl`/`val.jsonl` + `data_card.md`. Examples pair the COMPACT prompt with gold generated under the FULL prompt.
- `notebooks/phase7_generate_dataset.ipynb` — Colab data-gen (BGE-M3 index + Qwen3 self-distill + Gemini subset + downloads). Docs: `docs/LLM_PHASE7_RAFT_DATA.md`.

**Verified (offline mock):** 18/18 kept (pass 1.0); compact prompt + distractors + valid 5-section grounded gold + Unicode-normalized; data card written. Real targets come from Qwen3/Gemini in Colab.

**Objective (reframed):** Build an SFT dataset that teaches Qwen3 to produce the **already-validated** behavior *directly* — internalizing safety + format so the model needs no post-hoc fixing and a shorter prompt. Each example = (ML output + retrieved chunks [golden + distractors]) → (safe, structured, evidence-cited guidance), **no CoT**.

**Why this shape (Phase-6 evidence):** base+RAG+guardrail is already near-ceiling on quality/format/readability/personalization. The only gap is ~10% of outputs needing the validator (emergency redirect, nutrition deferral). So the training target is the **post-validator** answer, and the win is *guardrail-independence + consistency + prompt compression*, not raw quality.

**Data source (chosen): self-distillation + teacher subset.**
- **Bulk (~80%):** run base Qwen3 + RAG + **validator** over training-distribution profiles; use the **`final_safe_text`** (post-validator, guaranteed-safe, emergency-aware) as the gold target. This directly teaches the model to self-produce what currently needs base+validator. No external API for this bulk → low cost, privacy-preserving.
- **Teacher subset (~20%):** a stronger hosted teacher generates gold answers for harder/edge cases (emergency profiles, rare factor combinations, stylistic nuance/empathy) to raise the ceiling and add diversity. Transparency per §0.3 (teacher only for data; synthetic/de-identified profiles only).

**Steps:**
1. `data/train_profiles.py`: build diverse profiles **from the TRAIN/VAL distribution only** (NOT the ML test set, NOT the 30 Phase-6 eval profiles). Balanced across 3 classes + factor combos; include emergency-vital profiles. Target ~600–1200.
2. Per profile: retriever → golden chunks + **1–3 distractors** (RAFT ingredient).
3. Generate gold: bulk via `GuidanceRuntime` (base Qwen3 + validator → `final_safe_text`); subset via teacher.
4. **Hard gate — every gold answer must pass `scan_unsafe`** (incl. emergency redirect where applicable), parse to all 5 sections, be grounded, address the driving factors, and use only general (non-numeric) nutrition phrasing.
5. **Normalize Unicode** (curly quotes/em-dashes → ASCII) to avoid tokenizer quirks.
6. Split `train.jsonl` / `val.jsonl` (~90/10). `eval_profiles.jsonl` (the 30) stays fully held out.
7. Format to **Qwen3 chat template**; mask loss to assistant turn. Train with a **compact** system prompt (subset of the full prompt) so fine-tuning internalizes format/safety and inference gets cheaper.

> **Profile separation (paper):** training profiles from the training distribution only; the 30 held-out profiles reserved for Phase-9 ablation; the ML test set never used for LLM training.

**Deliverable:** `train.jsonl`, `val.jsonl`, data-card (self-distillation + teacher disclosure + filter pass-rate).

**Test:** Manually review 30 pairs — safe, grounded, formatted, factor-relevant, no CoT, no over-specific numbers. 100% pass `scan_unsafe`.

**Risks:** Self-distillation caps quality near base → mitigated by the teacher subset + style constraints. Teacher unsafe output → same `scan_unsafe` gate applies to teacher answers.

---

### PHASE 8 — LoRA Fine-Tuning  ✅ DONE — trained on the GPU box

> Depends on Phase 7 data. Loads via the existing `QwenChatModel(adapter_path=...)` hook, so no runtime change is needed.
>
> **Precision note:** the actual run used `--precision bf16` — an unquantized bf16
> frozen base, plain **LoRA**, not the script's `4bit`/QLoRA default. The 24 GB
> training GPU had enough headroom that 4-bit NF4 quantization wasn't needed, and
> bf16 avoids the extra quantization error NF4 would add on a small, 540-example
> training set. Result: 3 epochs, val loss 0.1967 → 0.1611 → **0.1546**
> (`llm/finetune/adapters/v1/run_card.md`). The steps below describe the tool's
> general QLoRA-capable design; "4-bit NF4" in step 1 is what the `4bit` mode
> does and was not the mode actually used.

**Objective:** Fine-tune Qwen3-4B to emit grounded, safe, structured guidance directly from a **compact prompt** — internalizing the format + safety so raw_unsafe → ~0 without the validator.

**Implemented in:**
- `ckd_risk_app/llm/finetune/data.py` — chat-template encoding + **assistant-only loss masking** by prompt-prefix length (tokenizer-version independent; raises if the template stops being prefix-stable), truncation/zero-supervision handling, length stats, padding collator.
- `ckd_risk_app/llm/finetune/qlora_train.py` — LoRA (r=16, alpha=32, dropout 0.05, 7 target projections) on a frozen base that's either 4-bit NF4/double-quant (QLoRA, the default) or bf16/unquantized via `--precision` — **the real run used bf16**, 3 epochs / lr 2e-4 cosine, gradient checkpointing + paged AdamW 8-bit, best-checkpoint-on-val-loss. **VRAM auto-detect** sets batch/accum/seq-len (all overridable). Emits `adapter/`, `training_log.csv`, `training_curve.png`, `run_card.md`. `--dry-run` validates the data pipeline with no GPU.
- `ckd_risk_app/llm/finetune/evaluate_adapter.py` — acceptance check: generates from the COMPACT prompt with the **validator OFF** and reports `raw_unsafe_rate` (headline), format, risk-mention, readability. Run with/without `--adapter` for the Phase-9 base-vs-fine-tuned pair.
- `misc/notebooks/phase8_qlora_finetune.ipynb` — Colab mirror. Docs: `misc/docs/LLM_PHASE8_QLORA_FINETUNE.md`.

**Verified (offline, no GPU):** pure masking functions; assistant-only span starts at the answer and ends at `<|im_end|>`; prompt fully masked to `-100`; prefix-stability guard raises on a broken template; real Phase-7 smoke rows encode with 0 dropped / 0 truncated at 2048 (p50 1424, p95 1616 tokens; answers p50 163); zero-supervision examples dropped at a tiny window.

> **Deviation (deliberate):** step 3 below names TRL's `SFTTrainer`; the implementation uses `transformers.Trainer` over a pre-tokenized dataset. Identical objective, but the assistant-only masking is explicit and unit-tested rather than dependent on TRL's shifting collator API. State this in the paper.

**Steps:**
1. `qlora_train.py` (Colab): load Qwen3-4B in **4-bit NF4** (bitsandbytes, double-quant), gradient checkpointing.
2. LoRA: `r=16–32`, `alpha=32`, dropout 0.05, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
3. Train with **TRL `SFTTrainer`** on `train.jsonl` (compact system prompt), loss masked to assistant response. Start: 3 epochs, lr 2e-4 cosine, grad-accumulation, `max_seq_len` sized to compact-prompt+chunks+answer.
4. Track train/val loss; save best adapter. Load it via the existing `QwenChatModel(adapter_path=...)` hook — no runtime changes.
5. Qualitative check on `val.jsonl`.

**Deliverable:** LoRA adapter + training curves.

**Test:** Val loss plateaus without overfitting; 10 val generations coherent, formatted, safe **on the compact prompt**.

**Risks:** OOM on T4 → reduce `max_seq_len`, batch=1 + grad-accum, r=16, or A100. Overfitting (base already strong, data partly self-distilled) → modest epochs, dropout, early stop; watch for the model merely memorizing base phrasing.

---

### PHASE 9 — Evaluation & Ablation (paper's LLM results)

> **✅ DONE (automated axes) — paper extension.** 4-arm ablation run on real Qwen3-4B-Instruct-2507 + the Phase-8 LoRA adapter (bf16-trained), on a **fresh** 30-profile set scored by the real trained ML model (`llm/eval/phase9_profiles.py`), not the hand-authored Phase-6 set — Phase 6 had two evaluation-harness bugs found and fixed here (see `docs/LLM_PHASE9_ABLATION_EVAL.md` §0). **Human/expert review is the one remaining open item.**

**RESULT (base / base+RAG / base+RAG+validator / fine-tuned+RAG+validator, 30 real-model-scored profiles):**
- raw_unsafe_rate: **0.133 / 0.067 / 0.100 / 0.033** — fine-tuning is the only arm that moves the raw (pre-validator) number, and moves it furthest; `diagnosis` and `nutrition_restriction` violations are fully eliminated in the fine-tuned arm, leaving only one hard emergency-redirect edge case.
- final_unsafe_rate: **0.0** in both validator-on arms (validator does its job regardless of arm).
- mean_factor_relevance: 0.95 / 0.983 / 0.983 / **1.0**. mean_prompt_tokens: 1456 / 2243 / 2243 / **1638** (**-27%** vs the validator-on base arm — the quantified prompt-compression claim).
- **Fix applied mid-phase:** the model was never told the BP threshold the validator itself checks (SBP≥180/DBP≥120), so `emergency_redirect` was the one violation category present in every arm. Added a shared threshold (`contract.has_emergency_vitals`) and an unmissable, patient-specific prompt alert citing the patient's own reading. Fixed all 3 clear high-systolic cases outright; two borderline isolated-diastolic cases remain inconsistent even with the fix (sampling-variance non-compliance, not a further wording problem) — an honest stated limitation, harmless in deployment since the validator guarantees final_unsafe=0 regardless.
- Full table, methodology, and the emergency-redirect story: `docs/LLM_PHASE9_ABLATION_EVAL.md`.

**Objective:** Quantify quality, safety, grounding, and the value of each design choice — all on held-out `eval_profiles.jsonl`.

**Automated axes:**

| Axis | Method | Metric |
|---|---|---|
| Safety | Adversarial + normal inputs | **Raw unsafe rate (pre-validator)** and **final unsafe rate (post-validator)**; severe-violation target 0; validator precision/recall |
| Guideline consistency | Rubric / expert review | % consistent with KDIGO/ADA |
| Factor relevance | Output addresses top factors | % covering top-2 factors |
| Groundedness | RAGAS-style: supported by retrieved chunks | faithfulness score |
| Readability | Flesch-Kincaid on 30+ outputs | grade level (target ≤ 8) |
| **Ablation 1** | **Base vs fine-tuned** (base arm = frozen Phase-6 results) | **raw_unsafe ↓ (guardrail-independence)**, format/personalization deltas, **prompt-length / inference cost** |
| **Ablation 2** | **RAG vs no-RAG** (and self-distill+teacher vs naive SFT) | groundedness/factuality deltas |

> **Reframed primary headline (per Phase-6 finding):** base is already strong on quality, so the fine-tuning result is framed as *internalizing safety + format (raw_unsafe 0.10 → ~0 with no validator) on a compact prompt at lower inference cost*, not as a raw-quality jump. Report honestly if quality deltas are small.

**Human / expert review (required for credibility):**
- Reviewers: **1 clinician / medical graduate / nephrology-aware reviewer + 1 ML/AI reviewer**.
- Sample: **30–50 held-out outputs**.
- Rubric (1–5 each): **safety, guideline consistency, personalization, clarity, overclaiming, actionability**.
- If no clinician is available, state the honest limitation: *"The LLM evaluation used a guideline-derived rubric and non-specialist rater; prospective clinical evaluation remains future work."* (Still try to get a medical student / resident to score 30–50 outputs — it markedly raises credibility.)

**Steps:** `eval_safety.py`, `eval_quality.py`, `eval_ablation.py`, `eval_human.py` → tables/figures to `eval/results/`.

**Deliverable:** Results for paper §3.10 incl. both ablations + before/after-validator safety + human-review scores.

**Test:** Reproducible from one run; fine-tuned+RAG best on quality/safety/grounding.

**Risks:** Reviewer availability → clear KDIGO/ADA rubric so scoring is consistent; state rater qualifications.

---

### PHASE 10 — Serving & App Integration  ✅ MINIMAL ENDPOINT DONE (frontend/GGUF deferred)

**Objective:** Expose the component as a local API and surface it in the frontend.

**Done (thesis MVP):**
- `ckd_risk_app/backend/llm_endpoint.py` — `build_guidance_response()` wires the runtime (RAG + model + validator); lazy singleton; adds `mode` ("production" | "local_demo"), `disclaimer`, provenance, safety status. Imports only `llm.*` (no ML stack) so it loads for the demo.
- `ckd_risk_app/backend/main.py` — **`POST /generate-guidance`** (features → `/predict` → guidance) + `GET /generate-guidance/health`.
- **Local-demo mode:** when Qwen3/torch is absent, the runtime falls back to the deterministic safe generator, still passes the **same safety validator**, and is clearly labelled `mode=local_demo` (`CKD_LLM_MODEL` env selects the model).
- `ckd_risk_app/llm/demo_supervisor.py` — 3 supervisor cases (low / moderate / high-emergency-validator-fixed) → `ckd_risk_app/results/thesis_demo_cases.md`.

**Deferred (paper):**
1. `merge_and_export.py`: merge LoRA → base → **GGUF** (llama.cpp) → quantize **Q4_K_M** + Ollama `Modelfile`.
2. Frontend: show guidance under the risk result with the disclaimer + general provenance; never surface chunk IDs/reasoning.

---

### PHASE 11 — Paper Write-up (LLM sections)

**Objective:** Document the LLM component (feeds PLAN.md §2.12, §3.10).

**Covers:** architecture, KB construction + source hierarchy + KDOQI restriction, RAFT data generation (no-CoT) + teacher transparency + profile separation, LoRA fine-tuning setup (bf16, not QLoRA's 4-bit — see Phase 8), strict safety validator + before/after metrics, evaluation protocol + results + human review, limitations (synthetic training data, single/non-specialist rater, no prospective deployment), ethics statement, scope statement (education not counseling).

**Deliverable:** Draft Methods + Results subsections + figures.

---

## 4. Dependencies

```
# RAG
faiss-cpu (or faiss-gpu), FlagEmbedding (BGE-M3), pymupdf, pdfplumber, langchain (optional)
# Fine-tuning
transformers, peft, trl, bitsandbytes, accelerate, datasets
# Serving
ollama / llama-cpp-python, fastapi, uvicorn
# Eval
textstat (Flesch-Kincaid), ragas (optional), pandas, matplotlib
```

---

## 5. Rough Effort

| Phase | Effort |
|---|---|
| 1 Foundations | 0.5 wk |
| 2 Knowledge base | 1 wk (partly manual) |
| 3 RAG retriever | 0.5 wk |
| 4 Safety validator | 0.5 wk |
| 5 MVP | 0.5 wk |
| 6 MVP eval (go/no-go) | 0.25 wk |
| 7 RAFT data | 1 wk |
| 8 LoRA fine-tune | 0.5–1 wk |
| 9 Eval & ablation + human review | 1 wk |
| 10 Serving + frontend | 0.5–1 wk |
| 11 Write-up | ongoing |

Critical path: KB+RAG (2–3) → MVP (4–6) → data+fine-tune (7–8) → eval (9). Serving (10) can overlap eval.

---

## 6. Key Risks (top 5)

1. **Unsafe LLM output** → RAFT safe-format training + strict post-hoc validator + safe fallback + before/after metrics. Non-negotiable gate.
2. **Compute (T4 OOM)** → 4-bit + grad-checkpoint + small batch; A100 only for the final run.
3. **Teacher data quality** → automated safety/quality filter + 30-sample manual audit; transparency disclosure.
4. **Small-dataset overfitting** → modest epochs, dropout, early stop, RAFT distractors.
5. **Evaluation rigor** → KDIGO/ADA rubric, held-out eval profiles, two ablations, human review, before/after-validator safety.

---

## 7. Definition of Done

- [ ] `/generate-guidance` returns safe, grounded, structured kidney-risk **education** for any `/predict` output
- [ ] Fine-tuned Qwen3-4B served locally (GGUF/Ollama); **no external API at inference**
- [ ] Safety validator: **0 severe violations** in final output on the adversarial set; before/after rates reported
- [ ] Evaluation shows fine-tuned+RAG > baselines on quality/safety/grounding; human-review scores reported
- [ ] Clean profile separation (train/val for training, held-out for eval, ML test set untouched)
- [ ] Reproducible scripts for KB, RAG, data-gen, training, eval
- [ ] Paper §2.12 + §3.10 drafted with figures, scope + teacher-transparency + KDOQI-restriction statements
```
