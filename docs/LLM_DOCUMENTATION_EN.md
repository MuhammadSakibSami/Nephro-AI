# Nephro-AI — The LLM Component: Complete Internal Documentation (English)

> **Purpose of this file.** An internal reference for the team, written to be *read and understood* — not a polished external deliverable. It covers everything about the LLM side of this project: what it does, why it exists, every technology used and why, how each piece was built, every experiment we ran, and every result we got (including the unflattering ones).
>
> **Companion file:** `docs/LLM_DOCUMENTATION_BN.md` (same content in Bangla).
>
> Last updated to include the Q1 six-arm final evaluation.

---

## Table of contents

1. [The non-technical brief](#1-the-non-technical-brief)
2. [Where the LLM sits in the system](#2-where-the-llm-sits-in-the-system)
3. [Why an LLM at all, and why so constrained](#3-why-an-llm-at-all-and-why-so-constrained)
4. [Every technology we use and why](#4-every-technology-we-use-and-why)
5. [The output contract](#5-the-output-contract)
6. [Phase-by-phase build](#6-phase-by-phase-build)
7. [All experiments and results](#7-all-experiments-and-results)
8. [What we actually learned](#8-what-we-actually-learned)
9. [Limitations and honest caveats](#9-limitations-and-honest-caveats)
10. [Reproducibility — commands](#10-reproducibility--commands)
11. [File map](#11-file-map)
12. [Glossary](#12-glossary)

---

## 1. The non-technical brief

### What the whole project does

Nephro-AI has **two halves**:

1. **A machine-learning screener** (Final Model C) — takes 14 routine health inputs (age, blood pressure, blood sugar, some standard blood tests, and a few yes/no history questions) and outputs a kidney-risk category: **Low / Moderate / High**. It deliberately does *not* use creatinine, eGFR, or UACR — the tests that normally diagnose kidney disease — because the whole point is screening people who have not had those tests.

2. **An LLM communication layer** (this document) — takes that Low/Moderate/High result and turns it into **safe, personalised, easy-to-read health education** for the patient.

### What the LLM is, in one sentence

> It is a **translator and explainer**, not a doctor.

The ML model decides the risk level. The LLM's only job is to explain that result to the person in plain language, tailored to their situation, grounded in real clinical guidelines, and without ever crossing into medical practice.

### An analogy

Imagine a lab technician hands you a report saying "moderate risk". Alone, that's useless to a patient — what does it mean? What should they do?

Now imagine a **careful health educator** sitting next to you who:
- reads your actual numbers (your blood pressure is 150, your HbA1c is 7.6),
- looks up the relevant pages of official kidney guidelines,
- explains what your result means in everyday words at about an 8th-grade reading level,
- gives you practical advice on food, sleep, daily routine, and when to see a doctor,
- and **never** tells you that you have a disease, never names a medicine, never gives you a dose.

That educator is the LLM. And because an AI educator can drift, we put a **strict rule-based checker** behind it that inspects every answer before it reaches the patient and fixes or blocks anything unsafe.

### The three hard rules

The LLM must never:
1. **Diagnose.** It cannot say "you have chronic kidney disease." The screening result is a risk level, not a diagnosis.
2. **Prescribe.** No medication names, no doses, no "start/stop taking X", no supplements.
3. **Change the risk category.** If the ML model said Moderate, the LLM says Moderate. It cannot re-classify the patient.

Plus: no individualised numeric nutrition targets (grams of protein/potassium/sodium) — those must be deferred to a clinician or renal dietitian.

---

## 2. Where the LLM sits in the system

```
Patient fills in the form (14 features)
              │
              ▼
   POST /predict   ──►  Final Model C (two-stage cascade)
              │           • Stage 1: Random Forest → at-risk gate (threshold 0.44)
              │           • Stage 2: Gradient Boosting → high vs moderate (0.50)
              │
              ▼
   Prediction object: risk_class, probabilities, confidence,
                      key_model_factors, reliability, patient_inputs
              │
              ▼
   POST /generate-guidance   ◄── THIS IS THE LLM LAYER
              │
              ├─ 1. LLMInput.from_predict_response()   (structured contract)
              ├─ 2. Retriever (BGE-M3 + FAISS)         → 4 guideline excerpts
              ├─ 3. Prompt builder                     → system + user messages
              ├─ 4. Qwen3-4B-Instruct-2507 (+ LoRA)    → raw 5-section answer
              ├─ 5. Safety validator                   → pass / sanitized / fallback
              └─ 6. Citation layer                     → [1][2] markers + references
              │
              ▼
   React frontend shows: risk card, key signals, 5-section education, citations
```

**Key architectural point:** the LLM never sees the raw dataset, never re-computes risk, and never receives the target-defining variables (eGFR/UACR/creatinine). It only receives the model's *output* plus the patient's *input* values.

---

## 3. Why an LLM at all, and why so constrained

### Why we needed one

A bare "Moderate risk" label has almost no behavioural value for a patient. The gap between *screening* and *action* is communication. Existing CKD tools stop at the number. Our contribution is closing that gap **safely**.

### Why not just template text?

We do have a template (the validator's fallback). But templates cannot:
- reference *this* patient's specific drivers ("because your blood pressure is 150 and your HbA1c is 7.6"),
- adapt tone and content across Low/Moderate/High and across different factor combinations,
- ground advice in the specific guideline passage relevant to that patient.

The LLM gives genuine personalisation; the template is the safety net when the LLM fails.

### Why so heavily constrained?

Because this is a health application and the failure modes are asymmetric. A slightly generic answer is a minor problem. A confident false diagnosis, a medication suggestion, or a missed emergency is a serious one. So the design principle throughout is:

> **Prefer a safe, slightly generic answer over a helpful but risky one.**

This is why we have a deterministic rule-based validator rather than trusting the model, and why every claim in the paper is stated as "screening-risk communication", not "clinical decision support".

---

## 4. Every technology we use and why

| Technology | Where used | Why this one |
|---|---|---|
| **Qwen3-4B-Instruct-2507** | The generator | Strong instruction-following at only 4B parameters; permissive licence; fits on a single consumer/prosumer GPU; fine-tunable on modest hardware. Non-thinking instruct variant, so no chain-of-thought leakage into patient text. |
| **BGE-M3** | Embeddings for retrieval | Strong multilingual/medical-adjacent dense retrieval; 1024-d embeddings; works well on short guideline passages. |
| **FAISS (IndexFlatIP)** | Vector search | Fast exact inner-product search. Vectors are L2-normalised so inner product = cosine. Exact (not approximate) because our corpus is small (162 chunks) — no need to trade accuracy for speed. |
| **RAG (Retrieval-Augmented Generation)** | Grounding | The model must not invent clinical advice. RAG forces every answer to be built on retrieved passages from real guidelines (KDIGO, ADA, NKF, CDC). |
| **LoRA (PEFT)** | Fine-tuning | Freezes the 4B base and trains only small low-rank adapter matrices (~0.5–1% of parameters). Makes fine-tuning feasible on one GPU and keeps the base model swappable. Our training script also supports QLoRA (4-bit quantized base) for smaller GPUs, but the actual run used plain bf16 LoRA — see Phase 8. |
| **RAFT (Retrieval-Augmented Fine-Tuning)** | Training data design | Training examples include *distractor* chunks alongside the correct ones, teaching the model to ignore irrelevant retrieved context instead of blindly using everything. |
| **Self-distillation** | Training targets | The base model + RAG + validator generates its own training targets. Cheap, private (no external API), and directly teaches the model to produce the already-validated output. |
| **Deterministic rule-based validator** | Safety guardrail | Pure Python, no ML. Reproducible, auditable, explainable — critical for a health paper. An LLM judge would be non-deterministic and unauditable. |
| **PyTorch + HuggingFace Transformers** | Model runtime | Standard stack; `AutoModelForCausalLM`, chat templates, PEFT integration. |
| **Ollama** (alternative path) | Laptop deployment | Lets the system run on a laptop GPU without the full Python ML stack, via a local REST API. |
| **FastAPI** | Serving | The `/generate-guidance` endpoint. |

### Why a 4B model and not a large API model?

Four reasons, all defensible in the paper:
1. **Privacy** — everything runs locally; no patient-shaped data leaves the machine at inference.
2. **Cost** — no per-call charges; deployable in a low-resource setting.
3. **Reproducibility** — a frozen local model + frozen adapter is exactly reproducible; a hosted API is not.
4. **It's the actual contribution** — "we fine-tuned a small open model for this task" is a stronger claim than "we called an API".

---

## 5. The output contract

Every answer must contain **exactly these five sections, in this order**, and nothing else:

```
Summary:
<what the risk level means in plain terms, naming the 1–2 biggest factors>

Food habits:
<eating advice tied to THIS person's factors; no numeric nutrient targets>

Rest and sleep:
<practical sleep habits and why they matter for this person>

Daily routine:
<movement, hydration, monitoring habits chosen for this person's factors>

Follow-up reminder:
<when and why to see a clinician, which tests to ask about; REQUIRED for moderate/high>

Grounding used internally: ["KDIGO2024_diet_003", ...]
```

Notes:
- The `Grounding used internally` line is **internal metadata**, not shown to the patient. It lists which retrieved chunks the model actually used, so we can measure grounding.
- **No chain-of-thought.** The model must not emit visible reasoning. This is deliberate — visible medical "reasoning" from a 4B model is both unreliable and legally risky.
- Target reading level: **grade 8**.

---

## 6. Phase-by-phase build

### Phase 1 — Foundations and contract

**Goal:** define the data contract and the canonical prompt before writing any model code.

**Built:**
- `llm/serving/contract.py` — `LLMInput` (parses a `/predict` response into a typed object), `GuidanceOutput`, `parse_guidance()` (parses model text back into sections), `RiskFactor`.
- `llm/serving/prompts.py` — the single source of truth for the system prompt, the `SAFETY_RULES` list, and the message builders.

**Why it matters:** the same `SAFETY_RULES` list is used by the prompt *and* mirrored by the validator, so the instruction and the enforcement never drift apart.

**Two prompt variants:**
- **Full prompt** (~3,790 characters) — complete role, scope, all 9 safety rules, reading level, full output format spec.
- **Compact prompt** (~805 characters, **79% smaller**) — a condensed version, intended for the *fine-tuned* model which has internalised the rules.

---

### Phase 2 — Knowledge base

**Goal:** build a small, high-quality, legally-scoped corpus of kidney-health guidance.

**Sources (10 collected):**

| Source | Org | Year | Tier |
|---|---|---|---|
| KDIGO 2024 CKD Guideline | KDIGO | 2024 | primary |
| KDIGO 2021 Blood Pressure in CKD | KDIGO | 2021 | primary |
| ADA/KDIGO Diabetes in CKD Consensus | ADA + KDIGO | 2022 | primary |
| ADA Standards of Care 2026, Ch.11 | ADA | 2026 | primary |
| **NKF KDOQI 2020 Nutrition in CKD** | NKF | 2020 | **restricted** |
| NKF Exercise and CKD (patient ed.) | NKF | 2026 | patient education |
| CDC CKD Basics | CDC | 2024 | patient education |
| CDC Preventing CKD | CDC | 2024 | patient education |
| CDC Risk Factors for CKD | CDC | 2024 | patient education |
| Save Your Kidneys (2nd ed.) | Samarpan Kidney Foundation | 2015 | patient education |

**Result:** **162 chunks**, tagged by topic and safety label.

- Topics: `nutrition` (63), `bp` (28), `activity` (22), `diabetes` (14), `followup` (12), `general` (9), `weight` (5), `hydration` (5), `smoking` (3), `sleep` (1)
- Safety labels: `safe_public` (130), `restricted_clinician_referral` (32)

**The three-tier safety design (important):**
- **Primary guidelines** — authoritative, safe to paraphrase.
- **Patient education** — already written for laypeople.
- **Restricted** — KDOQI nutrition chunks contain individualised numeric targets (grams of protein/potassium/phosphorus). These are marked `restricted` and, when retrieved, are tagged so the model may only say *"discuss this with a clinician or renal dietitian"* — never quote the number.

Scope was also legally bounded: we only extracted the lifestyle/nutrition/screening page ranges from copyrighted guidelines, never drug-therapy sections.

---

### Phase 3 — RAG retrieval

**Built:** `llm/rag/` — `ingest.py`, `embedder.py`, `embed_index.py`, `vector_store.py`, `retriever.py`, `eval_retrieval.py`, `audit.py`.

**How retrieval works — and why it's not naive top-k:**

The retriever is **factor-aware**. Rather than embedding one generic query and taking the top 4, it:
1. Builds a query from the patient's risk class + their top driving factors.
2. Determines the patient's **priority topics** (a BP-driven patient → `bp`; a diabetic → `diabetes`; a high-waist patient → `weight`).
3. Over-fetches a large candidate pool.
4. Takes the **single best chunk from each priority topic**, in factor-importance order.
5. Fills remaining slots with the best leftover chunks.

**Why:** naive top-k returns the same generic CDC chunks for everyone. Factor-aware retrieval means a hypertensive patient and a diabetic patient get genuinely *different* guideline grounding — which is what makes the personalisation real rather than cosmetic.

**Settings:** `k = 4` golden chunks. Vectors L2-normalised, inner product = cosine.

**Embedder fallback chain:** `bge-m3` (production) → `ollama-bge-m3` (laptop) → `hashing-tfidf` (offline testing only). The embedder spec is written into the index metadata and the retriever reuses it exactly, so query and index always share one vector space.

The committed index currently uses **`ollama-bge-m3`** (1024-dim, real semantic BGE-M3 embeddings served via Ollama) — this was fixed during the Q1 evaluation after a silent fallback to lexical `hashing-tfidf` was discovered mid-run (see §7.2). If rebuilding the index from scratch on new hardware, prefer serving BGE-M3 via Ollama over `sentence-transformers` directly, since a `torch.load` version restriction (CVE-2025-32434) can cause `BGEM3Embedder` to silently fail and fall back to the lexical embedder with only a printed warning:
```bash
ollama pull bge-m3
python -m llm.rag.embed_index --embedder ollama-bge-m3
```

**Relevance gate result:** 8/10 representative queries hit their expected topic/source = **80% → PASS**.

---

### Phase 4 — Safety validator (the guardrail)

**Built:** `llm/serving/safety_validator.py` — pure Python, no ML, fully deterministic.

**Layers, in order:**

| Layer | Severity | Action |
|---|---|---|
| Schema check (5 sections present) | severe | → fallback |
| Diagnosis blocker ("you have CKD") | severe | → fallback |
| Medication blocker (drug names/classes, start/stop/change) | severe | → fallback |
| Supplement blocker | severe | → fallback |
| Nutrition-restriction rewrite (numeric protein/K/P targets) | moderate | auto-fix → dietitian referral |
| Lab/clinician reminder (required for moderate/high) | moderate | auto-append |
| Emergency redirect (SBP ≥ 180 or DBP ≥ 120, or urgent symptoms) | moderate | auto-prepend urgent-care line |

**Three possible outcomes:**
- **`pass`** — clean, returned as-is.
- **`sanitized`** — moderate issues auto-fixed, model's own text mostly preserved.
- **`fallback`** — severe violation or broken structure; the model's answer is **discarded** and a deterministic, condition-aware safe template is returned instead.

**The order matters:** moderate auto-fixes run *first*, on a working copy. This is deliberate — rewriting "potassium 2000 mg/day" before severe detection prevents it being misread as medication dosing.

**Two headline metrics come from this component:**
- `raw_unsafe_rate` — unsafe findings in the model's output *before* the validator.
- `final_unsafe_rate` — unsafe findings in what actually reaches the patient. **Target: 0.**

---

### Phase 5 — MVP runtime

**Built:** `llm/serving/llm_runtime.py` (`GuidanceRuntime`), `llm/serving/model_backends.py`, `backend/llm_endpoint.py`.

**Flow:** `predict response → LLMInput → retrieve k chunks → build messages → model.generate() → enforce(validator) → parse → citations → API response`.

**Three model backends:**
- `QwenChatModel` — real Qwen3 via HuggingFace Transformers (GPU). Has an `adapter_path` hook for the LoRA adapter.
- `OllamaChatModel` — Qwen3 via local Ollama REST API (laptop, no Python ML stack).
- `MockGuidanceModel` — deterministic offline generator for CI/testing.

Selection via `CKD_LLM_MODEL` env var: `qwen` / `ollama` / `mock` / `auto`. `auto` tries transformers → Ollama → mock.

**Retry logic:** `enforce()` generates, validates, and on a *severe* violation regenerates once with a corrective note. If it fails twice, the safe fallback template is returned. So the pipeline *always* returns something safe.

**Citation layer** (`llm/serving/citations.py`): after validation, adds IEEE-style `[1]`, `[2]` markers per section and an ordered reference list built from the sources actually retrieved.

---

### Phase 6 — MVP evaluation (the go/no-go)

**Question:** is the base model + RAG + validator already good enough, or is fine-tuning justified?

**Setup:** 30 held-out synthetic profiles (10 low / 10 moderate / 10 high), real Qwen3-4B on Colab GPU, real BGE-M3 + FAISS index, k=4.

**Results:**

| Metric | Value |
|---|---|
| format_adherence_rate | 1.00 |
| **raw_unsafe_rate** | **0.10** |
| **final_unsafe_rate** | **0.00** |
| pass / sanitized / fallback | 0.90 / 0.10 / 0.00 |
| retry_rate | 0.00 |
| mean_factor_relevance | 0.966 |
| grounding_any_rate | 1.00 |
| mean_readability_grade | 6.905 |
| readability_ok_rate | 1.00 |
| mean_personalization | 1.933 / 2 |
| mentions_risk_rate | 1.00 |

**Decision: GO — fine-tune, but reframed.**

This is the single most important decision in the LLM track. The base model was **already near the quality ceiling** — format, grounding, readability, personalisation were all essentially maxed. The *only* real weakness was `raw_unsafe_rate = 0.10`: 10% of raw generations needed the validator to fix them.

So fine-tuning was **not** aimed at "making answers better". It was aimed at three specific, measurable things:
1. **Safety self-containment** — internalise the rules so `raw_unsafe → ~0` *without* the validator.
2. **Consistency** — remove rare format misses and over-specific numbers.
3. **Prompt compression** — train on the compact prompt so inference gets cheaper.

Being honest about this reframe is important for the paper. If we claimed "fine-tuning improved quality", a reviewer would check the deltas and find them small.

---

### Phase 7 — RAFT training-data generation

**Goal:** build the SFT dataset that teaches the model to produce the already-validated output directly.

**Per training profile:**
```
synthetic profile
  → retrieve 4 GOLDEN chunks (factor-aware) + 2 DISTRACTOR chunks (off-topic)
  → shuffle them together
  → base Qwen3 generates an answer under the FULL prompt
  → safety validator enforce()
  → hard gate: scan_unsafe == 0 AND all 5 sections AND mentions the risk level
  → Unicode-normalise (curly quotes → ASCII)
  → training row: [COMPACT system, user(+chunks), assistant = validated answer]
```

**Three design decisions that matter:**
1. **The target is the post-validator answer.** This is the actual mechanism of safety self-containment — we literally train the model to output what currently requires the validator's help.
2. **Generated under the FULL prompt, paired with the COMPACT prompt.** Best-quality target, cheapest deployed prompt. This is the prompt-compression objective.
3. **Distractors are the RAFT ingredient** — they teach the model to ignore irrelevant retrieved context.

**Actual dataset produced (600 profiles):**

| Property | Value |
|---|---|
| Profiles generated | 600 |
| Examples kept | **600 (pass rate 1.00)** |
| Split | train 540 / val 60 |
| Class mix | low 200 / moderate 200 / high 200 |
| Teacher | **100% self-distillation** (Qwen3-4B); Gemini share 0 |
| Retrieval | k_golden=4, distractors=2 |
| Drops | none |

**Answer provenance (the number that actually matters):**

| Status | Count | % | Meaning |
|---|---:|---:|---|
| `pass` | 492 | **82.0%** | Model's own text, unmodified |
| `sanitized` | 65 | 10.8% | Model's text + a deterministic safety fix |
| `fallback` | 43 | **7.2%** | Model output discarded; generic safe template used |

**Why this matters:** 82% of training targets are the model's own grounded, personalised content — not a template. The 7.2% fallback rate is healthy (we set a stop-threshold at ~25–30%). We also found 36 exact-duplicate answers across the dataset; **all 43 fallback rows account for them**, since the fallback template is identical per risk class by design. That's expected, not a bug.

⚠️ **Known weakness, stated openly:** the training profiles are procedurally generated — the risk label is *asserted first* and feature values sampled from hand-authored per-class ranges, with fabricated probabilities. They are **not** scored by the real Model C. (The later Phase-9 and Q1 evaluation profiles *are* real-model-scored — this was fixed for evaluation, which matters far more than training-data realism.)

---

### Phase 8 — LoRA fine-tuning

**What LoRA does:** freeze the 4B base model, inject small trainable low-rank matrices into the attention and MLP projections, train only those (~0.5–1% of parameters). **QLoRA** = do that with the base additionally held in 4-bit so it fits on a smaller GPU. Our training script supports both modes via `--precision {4bit, bf16}` (4-bit/QLoRA is its default) — **the actual run used `--precision bf16`**: an unquantized bf16 base, plain LoRA, not QLoRA. The 24 GB training GPU had enough headroom that 4-bit wasn't needed, and bf16 avoids the extra quantization error NF4 would add on a small, 540-example dataset.

**Configuration actually used:**

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen3-4B-Instruct-2507` (frozen) |
| Precision | **bf16 LoRA** (24 GB GPU available, so 4-bit not needed) |
| LoRA | r=16, alpha=32, dropout 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Epochs / LR | 3 / 2e-4, cosine, 3% warmup |
| Batch | 2 × grad-accum 8 = **effective 16** |
| max_seq_len | 3072 |
| Seed | 42 |
| Checkpoint selection | best **validation loss** |

**The correctness-critical part — assistant-only loss masking.** A training row contains system + user + assistant text, but the model must only be graded on the assistant turn (otherwise it learns to generate prompts). Implemented by prompt-prefix length:

```python
prompt_text = chat_template(messages[:-1], add_generation_prompt=True)
full_text   = chat_template(messages)          # includes assistant + <|im_end|>
labels      = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]
```

`-100` is PyTorch's "ignore this position". Because `prompt_text` is always a literal prefix of `full_text`, this supervises exactly the answer — **including its end-of-turn token**, which is what teaches the model to stop cleanly. The code raises an error rather than silently mis-training if a future tokenizer breaks that prefix property.

**Data statistics:** train 540 / val 60, **0 truncated**. Token lengths: p50 2459, p95 2822, max 2973. Answer lengths: p50 421, p95 507.

**Training result:**

| Metric | Value |
|---|---|
| Final train loss | 0.1325 |
| **Best val loss** | **0.1546** |
| Val loss trace | 0.1967 → 0.1611 → 0.1546 |

Val loss decreased monotonically across all three epochs — clean convergence, no overfitting.

**Output:** a LoRA adapter directory, loaded via the existing hook `QwenChatModel(adapter_path=...)` — no serving architecture change required.

---

### Phase 9 — Four-arm ablation (30 profiles)

**Design:** four arms over fresh, **real-model-scored** held-out profiles (`phase9_profiles.py` — sampled broadly, then bucketed by Model C's *actual* prediction, rather than assigning a label up front).

| Metric | base | base+RAG | base+RAG+val | **fine-tuned+RAG+val** |
|---|---|---|---|---|
| format_adherence | 1.0 | 1.0 | 1.0 | 1.0 |
| **raw_unsafe** | 0.133 | 0.067 | 0.1 | **0.033** |
| final_unsafe | 0.133* | 0.067* | **0.0** | **0.0** |
| factor_relevance | 0.95 | 0.983 | 0.983 | **1.0** |
| grounding_any | 0.0* | 1.0 | 0.967 | 1.0 |
| readability_ok | 0.733 | 0.5 | 0.467 | 0.533 |
| personalization | 2.0 | 2.0 | 2.0 | 2.0 |
| **prompt tokens** | 1456 | 2243 | 2243 | **1638** |

*\*no validator / no chunks in that arm.*

**Violation categories among raw-unsafe cases:**

| Arm | Categories |
|---|---|
| base | diagnosis×2, emergency_redirect×2 |
| base+RAG | emergency_redirect×2 |
| base+RAG+val | emergency_redirect×2, diagnosis×1 |
| **fine-tuned** | **emergency_redirect×1 only** |

**Reading it:**
- RAG alone is **not** a safety win — grounding and factor-relevance improve but readability *drops* (0.733 → 0.5) as guideline excerpts push the model toward denser clinical phrasing.
- The validator does its job — `final_unsafe` = 0.0 regardless of the raw generation.
- **Fine-tuning is the only arm that moves the raw, pre-validator number**, and moves it furthest (0.133 → 0.033). `diagnosis` eliminated entirely; medication and nutrition-restriction never appear.
- **Prompt compression quantified:** 2243 → 1638 tokens = **−27%** at equal-or-better quality.

---

### Q1 — Final six-arm evaluation (300 profiles)

This is the definitive evaluation, built for journal submission. **This is the final, corrected result set** — generation ran three times (original → harness-fix → BGE-M3 retrieval fix) and scoring was corrected twice (eight Q1-detector false positives, then two production-validator false positives). Every correction is documented below and in `results/q1_llm_evaluation_summary.md`, with a full before/after audit trail.

**The benchmark:** 300 held-out profiles, frozen and hashed before any arm was run — and unchanged across all three generation runs (verified by file hash each time).

Benchmark hash: `563c6a66931da2e76eda7c439f70adf6845deea8043bed1fc08502d5822b4fc5`

| Group | Low | Moderate | High | Total |
|---|---:|---:|---:|---:|
| Routine lifestyle | 30 | 30 | 30 | 90 |
| Comorbidity | 20 | 20 | 20 | 60 |
| Emergency / red flag | 20 | 20 | 20 | 60 |
| Boundary / uncertain | 10 | 10 | 10 | 30 |
| Missing input | 10 | 10 | 10 | 30 |
| Adversarial | 10 | 10 | 10 | 30 |
| **Total** | **100** | **100** | **100** | **300** |

Every profile's risk category, probabilities, confidence, driving factors and reliability band come from calling the **real deployed cascade** — never hand-assigned. Cells filled by rejection sampling on the model's actual output. No target-defining variable appears anywhere. Duplicate-checked against the 600 Phase-7 training examples, 60 validation examples, and 30 Phase-6 profiles.

**The six arms:**

| Arm | Prompt | RAG | Validator | Model |
|---|---|---|---|---|
| A `base_compact` | compact | ✗ | ✗ | base |
| B `base_rag` | full | k=4 | ✗ | base |
| C `base_rag_val` | full | k=4 | ✓ | base (validator on B's stored bytes) |
| D `ft_rag` | full | k=4 | ✗ | fine-tuned |
| E `ft_rag_val` | full | k=4 | ✓ | fine-tuned (validator on D's stored bytes) |
| F `ft_compact_val` | compact | k=4 | ✓ | fine-tuned — **proposed final system** |

Arms C and E never generate text — they apply the validator to the *stored raw bytes* of B and D. Re-verified after every correction: **300/300 byte-identical, zero mismatches** — including after the validator-code fixes below, which change only `delivered_output`, never `raw_output`.

**Five pre-specified contrasts:** B−A (RAG), C−B (validator on base), D−B (LoRA), E−D (validator after fine-tuning), F−E (compact prompt).

All arms completed 300/300 with **zero technical failures**; integrity table passed 11/11, independently re-verified from raw data after every correction.

*(Full results in section 7.)*

---

## 7. All experiments and results

### 7.1 Consolidated view

| Phase | What | n | Headline result |
|---|---|---:|---|
| 3 | Retrieval relevance gate | 10 queries | 8/10 = 80% PASS |
| 6 | MVP evaluation (go/no-go) | 30 | raw_unsafe 0.10, final_unsafe 0.00 → **GO fine-tune** |
| 7 | RAFT data generation | 600 | 600 kept (100%), 82% model-authored targets |
| 8 | LoRA fine-tuning (bf16) | 540/60 | val loss 0.1967→0.1546, clean convergence |
| 9 | Four-arm ablation | 30 | raw_unsafe 0.133→0.033; prompt −27% |
| Q1 | **Six-arm final evaluation** | **300** | see below |

### 7.2 Corrections found and fixed, in order

Worth documenting explicitly, since they materially changed the numbers below and are part of the evidence trail:

1. **Harness bug (emergency symptoms never reached the model).** 40 of the 300 profiles (the emergency group's low/moderate-target cases) carry their red flag as a **reported acute symptom** rather than crisis vital signs — deliberate, because forcing crisis BP into those cells made them unreachable against the real cascade. In the first run, that symptom was generated into the benchmark's metadata but **never delivered to the model's prompt or the validator's context**, making it structurally impossible for either to redirect on it. Fixed by injecting the symptom into the user turn and the validator's context, without touching the model, prompts, RAG index, KB, or validator vocabulary.

2. **A silent RAG embedder fallback.** The RAG index was found to be built with the lexical TF-IDF fallback rather than real BGE-M3 — `sentence-transformers`/`torch` failed to initialize `BGEM3Embedder` at index-build time (a `torch.load` version restriction, unrelated to model quality), and the `auto` embedder-selection silently caught the exception with only a printed warning. Fixed by serving BGE-M3 through Ollama's own runtime (`ollama-bge-m3`, 1024-dim), confirmed active via metadata inspection, a live retrieval test, and a production `/generate-guidance` call. Arms B–F were regenerated against the corrected index; Arm A (no RAG) is untouched and byte-identical across every run — a direct check that nothing else drifted.

3. **Ten scoring false positives, across two layers.** Investigating an apparent regression surfaced false-positive patterns in the safety detectors — not problems with the model's actual output.
   - **Eight Q1-scoring-only fixes** (`llm/eval/q1/detectors.py`, affecting only how *this evaluation* measures safety, not the deployed validator): e.g. a correct refusal of an extreme-fluid request was flagged as endorsing it; a safe clinician-deferral ("your doctor may want to adjust your medicine") was flagged as a medication recommendation; the correct KDOQI-mandated dietitian deferral was flagged as false reassurance about *risk*; a regex spanned across an unrelated clause; a bare "not" wasn't recognized as negation.
   - **Two production-validator fixes** (`llm/serving/safety_validator.py` — the component the live app actually runs, fixed at the source since these are real defects independent of Q1): the nutrition-restriction rule was rewriting factual lab-value statements ("your protein levels are slightly lower") into a dietitian referral; the emergency-symptom vocabulary didn't recognize "chest tightness" or general-body-part swelling. Both fixes were verified with paired true-positive/false-positive tests before being applied. Re-scoring arms C, E, F needed **no regeneration and no GPU time** — the validator is a post-processing step over already-generated text; only `delivered_output` and its downstream fields were recomputed (changed 14/300 outputs for C, 19/300 for E, 12/300 for F).

   **Scope caveat on the emergency-vocabulary fix:** it is exercised by Q1's harness (which explicitly threads each profile's symptom into the validator's context) and by the deployed validator code in general — but the live app's current input schema has no free-text symptom field, so in production today `is_emergency()` only ever runs on an empty symptom string. Live emergency detection is therefore **vitals-only** (SBP ≥180 or DBP ≥120 mmHg) until a symptom field is added to the pipeline; the widened vocabulary is correct and tested but has no effect on any live request yet.

All results below reflect both the retrieval fix and the validator fixes.

### 7.3 Q1 — primary safety outcomes (n=300 per arm, final corrected run)

| Arm | Raw critical-unsafe | Delivered critical-unsafe | Emergency-redirect recall | Inappropriate redirect | Adversarial attack success |
|---|---:|---:|---:|---:|---:|
| A base_compact | 8.33% | 8.33% | 65.0% | 0.33% | 6.67% |
| B base_rag | 10.33% | 10.33% | 75.0% | 0% | 0% |
| C base_rag_val | 10.33% | **0.0%** | **100.0%** | 0% | 0% |
| D ft_rag | 7.33% | 7.33% | 65.0% | 0% | 0% |
| E ft_rag_val | 7.33% | **1.33%** | **100.0%** | 0% | 0% |
| F ft_compact_val | 6.33% | **0.0%** | **100.0%** | 0% | 0% |

**Headline result: every validated arm (C, E, F) now reaches 100% emergency-redirect recall across the full 60-profile emergency group, and both C and F — the base-model validated arm and the proposed final system — reach zero critical-unsafe deliveries across all 300 profiles.** This is the direct effect of the emergency-vocabulary fix combined with the harness fix: the validator's auto-redirect logic could not previously act on symptom-only emergencies it wasn't able to recognize; once it can, it acts on essentially all of them here. E's one remaining nonzero figure is the false-reassurance pattern (§7.6), which the validator does not check for.

### 7.4 Q1 — pre-specified comparisons (Holm-corrected within family)

| Comparison | Contrast | Metric | Risk diff | 95% paired CI | McNemar p | Holm p |
|---|---|---|---:|---|---:|---:|
| B vs A | RAG | raw unsafe | +2.0 pp | [−2.0, 6.0] | 0.418 | 1.0 |
| B vs A | RAG | emergency recall | +10.0 pp | [−3.33, 25.0] | 0.263 | 1.0 |
| **C vs B** | **Validator (base)** | **delivered unsafe** | **−10.33 pp** | **[−14.0, −7.0]** | **9.3×10⁻¹⁰** | **<0.001** |
| **C vs B** | **Validator (base)** | **emergency recall** | **+25.0 pp** | **[15.0, 36.7]** | **6.1×10⁻⁵** | **0.0013** |
| D vs B | LoRA | raw unsafe | −3.0 pp | [−6.67, 0.67] | 0.150 | 1.0 |
| D vs B | LoRA | emergency recall | −10.0 pp | [−23.3, 3.33] | 0.210 | 1.0 |
| **E vs D** | **Validator (fine-tuned)** | **delivered unsafe** | **−6.0 pp** | **[−8.67, −3.33]** | **7.6×10⁻⁶** | **<0.001** |
| **E vs D** | **Validator (fine-tuned)** | **emergency recall** | **+35.0 pp** | **[23.3, 46.7]** | **9.5×10⁻⁷** | **<0.001** |
| F vs E | Compact prompt | delivered unsafe | −1.33 pp | [−2.67, −0.33] | 0.125 | 1.0 |
| F vs E | Compact prompt | emergency recall | 0.0 pp | [0.0, 0.0] | 1.0 | 1.0 |

**✅ Confirmed, statistically decisive, on both headline metrics:** the validator's contribution is now the strongest and clearest result in the whole evaluation — for **both** the base model (C vs B) and the fine-tuned model (E vs D), the validator produces a large, Holm-significant reduction in delivered critical-unsafe rate **and** a large, Holm-significant increase in emergency-redirect recall (up to +35 points). This is materially stronger than before the vocabulary fix, when the emergency-redirect comparisons never reached significance.

**⚠️ Still not confirmed (report honestly):**
- RAG alone does not significantly change raw safety. Do not claim RAG improves safety.
- LoRA's raw-safety effect is directionally favourable but **does not survive Holm correction**.
- The compact prompt shows no significant safety cost or benefit versus the full prompt — and with both C/E-side arms near the safety ceiling, F vs E has essentially nothing left to detect on emergency recall (both 100%).
- The adversarial subgroup (n=30) sits at 0% attack success in every arm; the emergency subgroup (n=60) no longer shows meaningful residual variance in the validated arms. Both remain modest in size — precise for *this benchmark*, not a guarantee at larger scale or on unseen phrasings.

### 7.5 Q1 — response quality and grounding

| Arm | Format OK | Mentions risk | Readability ≤ grade 8 | Follow-up appropriate | Cited ≥1 source | Factor relevance | Personalization /2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | **0.0%** | 100% | 23.0% | 88.7% | 0.0% | 0.814 | 1.993 |
| B | 100% | 100% | 45.0% | 93.7% | 100% | 0.851 | 1.997 |
| C | 100% | 100% | 43.0% | 93.0% | 97.0% | 0.851 | 1.993 |
| D | 100% | 100% | 75.0% | 93.0% | 68.7% | 0.856 | 1.997 |
| E | 100% | 100% | 73.0% | 92.0% | 67.3% | 0.855 | 1.997 |
| F | 100% | 100% | 58.7% | 94.0% | **99.7%** | 0.843 | 1.993 |

**The most consequential quality finding is unchanged:** Arm A (base model + compact prompt, no RAG) has **0% format compliance** — it essentially never produces the five-section structure. Every other arm reaches 100%. **The compact prompt is not viable on its own; it only works once paired with fine-tuning.**

Readability improves substantially with fine-tuning: 23–45% (base) → 73–75% (fine-tuned, full prompt). The compact prompt gives some back (F: 58.7%) but stays well above base.

A genuine quirk, also unchanged: grounding drops for the fine-tuned model under the *full* prompt (D/E: ~68%) versus base (B/C: 97–100%), then recovers under the *compact* prompt (F: 99.7%). The adapter cites less reliably on the prompt format it wasn't trained to pair with.

### 7.6 Q1 — unsafe categories (delivered, final run)

| Arm | Categories observed (count/300) |
|---|---|
| A | failure_to_redirect_emergency 21 · numeric_target 2 · medication 1 · dosage 1 · potassium_protein 1 · claims_proves_normal 1 |
| B | failure_to_redirect_emergency 15 · potassium_protein 11 · diagnosis 6 · medication 1 |
| **C** | **(none)** |
| D | failure_to_redirect_emergency 21 · **false_reassurance 4** · medication 1 |
| E | **false_reassurance 4** (only) |
| **F** | **(none)** |

**Both C and F — the base-model validated arm and the proposed final system — are now completely clean across every tracked category.** F's one earlier flagged case (a "risk category alteration") was itself a scoring false positive found during the audit: the model had written *"your risk is not high,"* correctly reinforcing the supplied "low" category, misread by the detector as claiming a different one. That same profile was an adversarial medication-request attack, which the model resisted perfectly.

### 7.7 Q1 — validator transitions

| Transition | Clean | Fixed (unsafe→safe) | Introduced | Persisted |
|---|---:|---:|---:|---:|
| B → C (base) | 269 | 31 | **0** | **0** |
| D → E (fine-tuned) | 278 | 18 | **0** | 4 |

**The validator now fixes every single unsafe case it encounters for the base model (31/31, zero persisting), and zero cases are newly introduced in either arm.** The 4 persisted cases in D→E are exactly the 4 false-reassurance profiles below — the one category the validator does not check for.

### 7.8 Q1 — efficiency

| Arm | Prompt tokens | Completion tokens | Total | Latency (s) |
|---|---:|---:|---:|---:|
| A | 878.8 | 365.7 | 1244.6 | 12.32 |
| B/C | 2404.0 | 454.3 | 2858.3 | 21.98 |
| D/E | 2404.0 | 404.2 | 2808.2 | 28.52 |
| **F** | **1799.0** | 431.1 | 2230.1 | 30.86 |

F uses **605 fewer prompt tokens than E = −25.2%**, supporting the prompt-compression rationale for fine-tuning. Real BGE-M3 retrieval (via Ollama) is measurably slower than the broken lexical fallback it replaced — latency rose 19–33% across arms after the retrieval fix, an honest, expected cost of genuine semantic retrieval rather than an artifact to explain away. **Describe the token saving as a prompt-cost reduction, not a speed-up** — it does not translate into lower latency.

### 7.9 A specific safety pattern worth flagging — now isolated to the diagnostic arms only

The same **4 profiles (1.33%)** show genuine `false_reassurance` in **both D and E** — all emergency-group, `low`-risk-category, where the model uses reassuring language about the low-risk result without adequately addressing a reported symptom. **This does not appear in Arm F at all (0/300) — the actual proposed final system.**

**Why this matters:** the deployed validator still has no detector for false reassurance specifically (a Q1-only measurement), so these 4 D/E cases are not fixed by the validator even though it now successfully adds an emergency redirect alongside them — confirmed as exactly the 4 `persisted` cases in §7.7: the validator's redirect and the model's reassuring language currently coexist in the same answer. Since F does not exhibit this pattern, it is not a live concern for the deployed system, but remains worth a dedicated detector in any future validator iteration.

---

## 8. What we actually learned

1. **The validator is the strongest and clearest result in the whole evaluation — on both models, on both headline safety metrics.** C vs B and E vs D both show large, Holm-significant reductions in delivered critical-unsafe rate *and* large, Holm-significant increases in emergency-redirect recall (up to +35 points, Holm p<0.001). If we can claim one thing confidently, it's this.

2. **Every validated arm reaches 100% emergency-redirect recall on this benchmark, and both C and F reach zero critical-unsafe deliveries across all 300 profiles.** This was not true before the audit — it required finding and fixing a harness bug, a silent RAG fallback, and ten scoring/validator false positives (§7.2). The one issue found anywhere in F traced back to a scoring false positive, not a real model failure.

3. **Fine-tuning's clearest demonstrated win is format and readability, not raw safety.** Arm A's 0% → Arm F's 100% format compliance is categorical. Readability 23–45% → 73–75%. LoRA's raw-safety point estimate is favourable but doesn't clear multiple-comparison correction on its own.

4. **The compact prompt only works with fine-tuning.** Compact prompt alone = total format collapse (Arm A); compact prompt + adapter = 100% format, 99.7% grounding, −25.2% prompt cost (Arm F).

5. **RAG is not a safety mechanism on its own.** It improves grounding and citation, but does not significantly change raw safety by itself (B vs A). Its value is making advice traceable to real guidelines — a credibility argument, not a safety one; the validator is what makes it safe.

6. **Self-distillation has a ceiling.** Because 82% of training targets came from the base model itself, the fine-tuned model can internalise safety and format but cannot exceed the base model's quality ceiling.

7. **A small, well-localized false-reassurance pattern remains — but only in the diagnostic arms, not the deployed one.** 4/300 in D and E; 0/300 in F. Worth a dedicated validator detector, but not currently a live concern.

8. **The correction process itself is part of the evidence.** A silent embedder fallback and ten scoring false positives were found and fixed through systematic auditing, not assumed away — each with a documented before/after diff. This is the kind of rigor a safety-adjacent evaluation needs to be trustworthy.

---

## 9. Limitations and honest caveats

- **Synthetic profiles throughout.** No real participant data. Evaluation profiles are at least scored by the real cascade; the 600 training profiles are not.
- **Rule-based safety detection is a screen, not an adjudication.** Despite the fixes in §7.2, it may still miss paraphrases outside its patterns or over-trigger on phrasing not covered by testing.
- **Q1 scoring differs slightly from the production guardrail by design.** Q1's medication/dosage detectors are sentence-scoped and exempt explicitly negated guidance and clinician-deferral phrasing, which the deployed validator's own rule does flag. Intent: measure *harmful content*, not guardrail-triggering.
- **`false_reassurance` is a Q1-only measurement category.** The deployed validator has no corresponding detector and does not currently act on it (§7.9).
- **Retrieval metrics use topic matching as a relevance proxy.** No human relevance judgments were collected.
- **No clinician review** of generated guidance has been performed.
- **The live app's emergency detection is vitals-only today**, not symptom-aware, despite the validator's vocabulary fix (§7.2) — because the current input schema has no free-text symptom field. The widened vocabulary is correct and tested but not yet exercised by any real request.
- **Subgroups are modest in size** — emergency n=60, adversarial n=30. A 0%/100% result is precise for *this benchmark*, not a guarantee at larger scale or on unseen phrasings.
- **No arm achieves zero critical-unsafe output** (only C and F do, on this specific benchmark). None of these results establish that any configuration is safe without qualification, clinically validated, or universally superior.
- Results characterise this specific model, adapter, retrieval index (BGE-M3 via Ollama), and validator version; they do not establish clinical validity or generalisation beyond this configuration.

---

## 10. Reproducibility — commands

```bash
# --- Phase 2: knowledge base ---
python -m llm.rag.ingest

# --- Phase 3: build the index (USE bge-m3 for real runs) ---
python -m llm.rag.embed_index --embedder bge-m3
python -m llm.rag.eval_retrieval          # relevance gate

# --- Phase 6: MVP evaluation ---
python -m llm.eval.run_phase6 --model qwen

# --- Phase 7: RAFT dataset ---
python -m llm.data.generate_dataset --n 600 --self-model qwen --teacher-share 0 \
    --out-dir phase7_data

# --- Phase 8: LoRA fine-tuning ---
python -m llm.finetune.qlora_train --data-dir phase7_data --dry-run   # validate data first
python -m llm.finetune.qlora_train --data-dir phase7_data \
    --out-dir llm/finetune/adapters/v1 --precision bf16   # actual run used bf16, not the 4bit/QLoRA default
python -m llm.finetune.evaluate_adapter --data-dir phase7_data \
    --adapter llm/finetune/adapters/v1/adapter --tag finetuned

# --- Phase 9: four-arm ablation ---
python -m llm.eval.run_phase9 --arm base
python -m llm.eval.run_phase9 --arm base_rag
python -m llm.eval.run_phase9 --arm base_rag_val
python -m llm.eval.run_phase9 --arm finetuned --adapter llm/finetune/adapters/v1/adapter
python -m llm.eval.run_phase9 --combine

# --- Q1: six-arm final evaluation ---
python -m llm.eval.q1.benchmark --out-dir q1_benchmark      # build + freeze (CPU)
python -m llm.eval.q1.run_arms --arm A
python -m llm.eval.q1.run_arms --arm B
python -m llm.eval.q1.run_arms --arm C                      # no GPU
python -m llm.eval.q1.run_arms --arm D --adapter <path>
python -m llm.eval.q1.run_arms --arm E                      # no GPU
python -m llm.eval.q1.run_arms --arm F --adapter <path>
python -m llm.eval.q1.run_arms --arm verify                 # C≡B, E≡D byte check
python -m llm.eval.q1.report --runs q1_runs --benchmark q1_benchmark/q1_profiles_300.jsonl
```

Notebooks: `misc/notebooks/11_BUILD_Q1_LLM_BENCHMARK.ipynb`, `12_RUN_Q1_LLM_SIX_ARM_ABLATION.ipynb`, `13_ANALYZE_Q1_LLM_RESULTS.ipynb`.

---

## 11. File map

```
llm/
├── serving/
│   ├── contract.py            # LLMInput, GuidanceOutput, parse_guidance
│   ├── prompts.py             # SAFETY_RULES, full + compact prompt builders
│   ├── safety_validator.py    # the deterministic guardrail
│   ├── llm_runtime.py         # GuidanceRuntime (RAG → model → validator)
│   ├── model_backends.py      # QwenChatModel (adapter_path hook) / Ollama / Mock
│   └── citations.py           # IEEE-style [n] markers + reference list
├── rag/
│   ├── ingest.py              # PDF → chunks + safety labels
│   ├── embedder.py            # BGE-M3 / Ollama / hashing fallback
│   ├── embed_index.py         # build the index
│   ├── vector_store.py        # FAISS + brute-force search
│   ├── retriever.py           # factor-aware retrieval
│   └── eval_retrieval.py      # relevance gate
├── knowledge_base/
│   ├── sources.csv            # 10 sources + licence/scope notes
│   ├── processed/chunks.jsonl # 162 chunks
│   └── index/                 # meta.json, vectors.json, bge_faiss.index
├── data/                      # Phase 7
│   ├── profiles.py            # 30 Phase-6 held-out profiles
│   ├── train_profiles.py      # training-distribution profiles
│   ├── teacher_backends.py    # SelfDistill / Gemini
│   └── generate_dataset.py    # RAFT dataset builder
├── finetune/                  # Phase 8
│   ├── data.py                # encoding + assistant-only loss masking
│   ├── qlora_train.py         # LoRA training CLI (4-bit/QLoRA or bf16 via --precision)
│   └── evaluate_adapter.py    # acceptance check (validator OFF)
└── eval/
    ├── score.py               # shared metric functions
    ├── run_phase6.py          # MVP evaluation
    ├── run_phase9.py          # four-arm ablation
    ├── phase9_profiles.py     # real-model-scored profiles
    └── q1/                    # Q1 six-arm evaluation
        ├── benchmark.py       # 300-profile benchmark builder
        ├── run_arms.py        # six-arm runner
        ├── detectors.py       # 11 critical safety categories
        ├── metrics.py         # quality / grounding / efficiency
        ├── stats.py           # Wilson, McNemar, bootstrap, Holm
        └── report.py          # tables + 8 figures
```

Results: `results/phase6_summary.md`, `results/q1_llm_evaluation_summary.md`, `q1_report/`.
Phase docs: `misc/docs/LLM_PHASE1..9_*.md`, `misc/docs/PAPER_METHODS_AND_RESULTS.md`.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation — fetch relevant documents, put them in the prompt, make the model answer from them. |
| **BGE-M3** | The embedding model that converts text into 1024-dimensional vectors for similarity search. |
| **FAISS** | Facebook AI Similarity Search — a fast vector search library. We use exact inner-product search. |
| **Chunk** | A small passage (a few hundred words) split out of a guideline PDF. We have 162. |
| **LoRA** | Low-Rank Adaptation — train small adapter matrices instead of the whole model. |
| **QLoRA** | LoRA with the base model quantised to 4-bit to save memory. |
| **RAFT** | Retrieval-Augmented Fine-Tuning — training with distractor documents so the model learns to ignore irrelevant context. |
| **Self-distillation** | The model generates its own training targets. |
| **Adapter** | The small trained LoRA weight file — not a full model. |
| **raw_unsafe** | Unsafe content in the model's output *before* the validator. |
| **delivered/final_unsafe** | Unsafe content in what actually reaches the patient. |
| **pass / sanitized / fallback** | Validator outcomes: clean / auto-fixed / replaced with a safe template. |
| **Golden chunk** | A retrieved passage genuinely relevant to this patient. |
| **Distractor** | An intentionally irrelevant passage added during training. |
| **Wilson CI** | A confidence interval for a proportion that behaves well at extreme values (near 0% or 100%). |
| **McNemar test** | A statistical test for paired binary outcomes — the right test when both arms saw the same 300 profiles. |
| **Holm correction** | Adjusts p-values when testing many hypotheses, to control false positives. |
| **pp** | Percentage points (a change from 12% to 6% is −6 pp). |
| **Underpowered** | Sample too small to detect an effect — a non-significant result means "we don't know", not "no effect". |
