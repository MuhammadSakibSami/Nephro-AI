# Thesis LLM Component — MVP Summary

**Scope of this document:** a concise, supervisor-facing summary of the LLM
component for the thesis. The component is **complete and demonstrable** as a
minimal MVP; paper-grade fine-tuning is a documented, deferred extension.

> **Update:** this document is a pre-paper-extension snapshot. Phases 7–9 and
> the Q1 six-arm evaluation have since been completed on real hardware — see
> `misc/docs/PAPER_METHODS_AND_RESULTS.md` and `results/q1_llm_evaluation_summary.md`
> for current status and results. The "deferred" language below reflects the
> state at the time this summary was written, not the current state.

---

## 1. What it does (scope statement)

> The LLM component provides **personalized, guideline-grounded kidney-risk
> education** after ML-based screening. It does **not** diagnose CKD, prescribe
> treatment, or replace clinician consultation.

It takes the two-stage ML model's risk output (low / moderate / high + the factors
that drove it) and turns it into safe, plain-language, guideline-grounded education
the patient can act on — while guaranteeing it never gives medical/medication
advice.

---

## 2. Architecture

```
Patient inputs
   │
   ▼
Two-stage ML cascade  (/predict)  ──►  risk class, probabilities, top factors
   │
   ▼
LLMInput  (frozen data contract)
   │
   ▼
Guideline RAG  (BGE-M3 embeddings + FAISS;  TF-IDF fallback offline)
   │   retrieves the most relevant guideline chunks for this patient's drivers
   ▼
Prompt builder  ──►  LLM  (Qwen3-4B-Instruct-2507  |  local-demo generator)
   │
   ▼
Safety Validator  (rule layers + safe rewrite/fallback)
   │
   ▼
Safe, personalized kidney-risk education   →   /generate-guidance API
```

- **Existing ML model is unchanged.** The LLM consumes its `/predict` output.
- **Single endpoint:** `POST /generate-guidance` (FastAPI) returns the guidance
  sections, source provenance, a `mode` flag, and a disclaimer.
- **Deployment-agnostic model layer:** the same runtime drives the real Qwen3 (GPU)
  or a deterministic **local-demo** generator (no GPU) — both pass through the same
  validator.

---

## 3. Guideline RAG (knowledge base)

- **9 sources**, scoped to prevention/lifestyle/early-stage content only: KDIGO
  2024, KDIGO 2021 BP, ADA/KDIGO 2022 diabetes consensus, ADA 2026 CKD chapter,
  NKF KDOQI 2020 nutrition (restricted), NKF exercise (cleaned), CDC Basics /
  Prevention / Risk Factors.
- **118 chunks** after a **3-tier safety scoping** of the source PDFs:
  - `exclude_from_retrieval` — medication, dosing, dialysis, transplant, numeric BP
    targets → **dropped from the KB entirely**.
  - `restricted_clinician_referral` — numeric nutrition targets → surfaced only as
    "discuss with a clinician/dietitian".
  - `safe_public` — general lifestyle/prevention education.
- An independent **audit gate** verifies 0 unsafe leakage before the KB is used.
- Retrieval is **driver-aware**: the query is built from the patient's top model
  factors (e.g. high BP, high glucose) so the retrieved guidance is personalized.

---

## 4. Safety validator (guardrail)

Every generated answer is checked **after** generation and **before** the user
sees it. Severity model:
- **Severe → block** (regenerate, else safe fallback): any diagnosis claim,
  medication/dosing, or supplement recommendation; broken output format.
- **Moderate → auto-fix**: numeric nutrition target → dietitian referral; missing
  clinician+test reminder (moderate/high) → appended; **emergency vitals → urgent-care
  redirect added**.

Measured value: on an adversarial set, **raw unsafe 78% → final unsafe 0%**. The
guardrail guarantees the returned text is safe regardless of the model.

---

## 5. Evaluation evidence (frozen Phase-6 result)

Real **Qwen3-4B-Instruct-2507 + BGE-M3 RAG + validator** over **30 held-out
profiles** (10 low / 10 moderate / 10 high), in `ckd_risk_app/results/`:

| Metric | Result |
|---|---|
| Format adherence | **1.00** |
| Final unsafe rate | **0.00** |
| Fallback / retry | **0.00 / 0.00** |
| Factor relevance | **0.966** |
| Personalization | **1.93 / 2** |
| Grounding (cited real chunks) | **1.00** |
| Readability (Flesch-Kincaid) | **6.9** (100% ≤ grade 8) |
| Raw unsafe (pre-validator) | 0.10 — all **validator-handled** (emergency redirect / nutrition deferral); zero diagnoses/medications |

**This is the thesis LLM evaluation result.** It shows the system already produces
safe, grounded, readable, personalized education with the base model.

---

## 6. Demonstration

`python -m ckd_risk_app.llm.demo_supervisor` runs 3 cases end-to-end (artifact:
`ckd_risk_app/results/thesis_demo_cases.md`):
1. **Low risk** — reassuring preventive education.
2. **Moderate risk (BP + glucose)** — personalized to the drivers; clinician + lab
   follow-up reminder.
3. **High risk / emergency BP** — the validator adds an **urgent-care redirect**
   (status = *sanitized*), demonstrating the guardrail.

In the app, `POST /generate-guidance` does the same over the live ML prediction.

---

## 7. Why fine-tuning is deferred (and that's fine)

Phase-6 evidence shows **base Qwen3 + RAG + guardrail is already strong** (format
1.0, final-unsafe 0.0, grounding 1.0, readability 6.9). So fine-tuning is **not**
needed for a safe, working, evaluable thesis MVP. It was also temporarily blocked
by free-tier API quota and Qwen self-distillation CUDA OOM.

Fine-tuning is therefore positioned as a **paper extension**, not a thesis
requirement — its value is *consistency + safety self-containment + cheaper
inference*, not fixing a broken base.

---

## 8. Future paper plan (extension, not required for thesis)

The fine-tuning pipeline is **already built and validated offline**, ready to run
when API/compute allow:

- **Phase 7 — RAFT data generation** *(built; deferred)*: self-distillation bulk
  (base + RAG + validator) + ~20% Gemini-free-tier subset; safety-gated; trains a
  compact prompt. Code: `ckd_risk_app/llm/data/`, `notebooks/phase7_generate_dataset.ipynb`.
- **Phase 8 — LoRA fine-tuning** *(deferred at time of writing; since completed —
  see the update note above)*: LoRA on Qwen3-4B on the RAFT data (the training
  script supports either a 4-bit/QLoRA or bf16 frozen base via `--precision`; the
  real run used bf16, since the 24 GB training GPU didn't need 4-bit's memory
  savings); loads via the existing `QwenChatModel(adapter_path=...)` hook — no
  runtime change.
- **Phase 9 — Evaluation & ablation** *(deferred)*: fine-tuned vs the frozen Phase-6
  base arm; headline = **raw_unsafe 0.10 → ~0 without the validator + lower
  inference cost**; plus RAG-vs-no-RAG and a small human/expert review.

The codebase is kept **extensible** for this: pluggable model + teacher backends,
the `adapter_path` hook, frozen eval profiles, and a one-flag switch between
local-demo and production model.

---

## 9. Status at a glance

| Item | Status |
|---|---|
| Data contract, RAG, validator, runtime, MVP | ✅ Done (Phases 1–5) |
| Real-model evaluation (Qwen3) | ✅ Done & frozen (Phase 6) |
| `/generate-guidance` endpoint + local-demo mode | ✅ Done (minimal serving) |
| Supervisor demo (3 cases) | ✅ Done |
| RAFT data / LoRA fine-tuning / ablation | ✅ Since completed (paper extension) — see the update note above |
| GGUF/Ollama packaging, web frontend | ⏸ Deferred |
