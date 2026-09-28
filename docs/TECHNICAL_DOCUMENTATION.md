# Nephro-AI: Technical Documentation
### Routine-Blood-Test-Based, Cross-Sectional Kidney Risk Screening with a Two-Stage ML Cascade and a Fine-Tuned, RAG-Grounded, Safety-Validated LLM Education Layer

**Version:** Final Thesis Documentation 2.0
**Authors:** Mohammad Walid Hasan
**Institution:** Jahangirnagar University (JU) / IIT
**Date:** September 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack](#3-tech-stack)
4. [How It Works — End-to-End](#4-how-it-works--end-to-end)
   - 4.1 [Input Processing](#41-input-processing)
   - 4.2 [ML Prediction Mechanism](#42-ml-prediction-mechanism)
   - 4.3 [Between Prediction and the LLM](#43-between-prediction-and-the-llm)
   - 4.4 [Model Output → LLM Pipeline](#44-model-output--llm-pipeline)
   - 4.5 [How the LLM Works — Prompt Engineering](#45-how-the-llm-works--prompt-engineering)
   - 4.6 [LLM Generation](#46-llm-generation)
   - 4.7 [Personalized or Generalized?](#47-personalized-or-generalized)
   - 4.8 [Safety Validation](#48-safety-validation)
   - 4.9 [Frontend Rendering](#49-frontend-rendering)
5. [Knowledge Base](#5-knowledge-base)
6. [API Reference](#6-api-reference)
7. [Evaluation Results](#7-evaluation-results)
8. [Design Decisions & Tradeoffs](#8-design-decisions--tradeoffs)
9. [Technology Stack — Deep Dive (Where / How / Why / Pros / Cons)](#9-technology-stack--deep-dive-where--how--why--pros--cons)
10. [Unique Selling Points & Novelty (USP)](#10-unique-selling-points--novelty-usp)
11. [Limitations & Scope](#11-limitations--scope)
12. [Citations & References](#12-citations--references)
13. [Frontend Component Reference](#13-frontend-component-reference)
14. [Database Schema](#14-database-schema)
15. [Project Directory Structure](#15-project-directory-structure)
16. [Deployment Modes Comparison](#16-deployment-modes-comparison)
17. [Quick-Reference Numbers](#17-quick-reference-numbers)

---

## 1. Project Overview

### 1.1 What is Nephro-AI?

Nephro-AI is a **routine-blood-test-based, cross-sectional kidney risk screening and patient education system**. It combines:

1. A **two-stage ML cascade** (Stage-1 RandomForest gate → Stage-2 GradientBoosting discriminator) that screens for CKD risk (low / moderate / high) from 14 routine, low-cost inputs — vitals, comorbidity history, and blood-test values that are already part of a standard general checkup — *without requiring the specialized confirmatory kidney tests* (creatinine-based eGFR, UACR).
2. A **fine-tuned, retrieval-augmented, safety-validated LLM education layer** (Qwen3-4B-Instruct-2507 + LoRA fine-tuning + BGE-M3/FAISS RAG + a deterministic rule-based safety validator) that turns the screening result into personalized, plain-language kidney-health guidance grounded in clinical guidelines (KDIGO, ADA, KDOQI, CDC, NKF), guaranteed by the validator to contain no diagnosis, no medication advice, and no individualized numeric nutrient targets.

### 1.2 Problem Statement

Chronic kidney disease affects approximately 800 million people globally. The majority of cases go undetected until advanced stages because:

- The **specialized confirmatory kidney tests** (eGFR, serum creatinine, urine albumin-to-creatinine ratio/UACR) are not part of a routine general checkup — they require a kidney-specific lab order that is frequently skipped unless a clinician already suspects kidney disease, which creates the very detection gap this project targets.
- Asymptomatic early-stage CKD produces no obvious symptoms, so there is little clinical trigger to order that specialized test in the first place.
- Patient health literacy about kidney risk is generally very low.

Existing ML-based CKD detection models almost universally rely on eGFR or serum creatinine as input features — useful for clinical confirmation, but circular for *early screening*, since they require the very specialized test the screening step is meant to avoid ordering unnecessarily. Most ML screening tools also provide only a risk score, without explaining what the risk means, what to do about it, or when to seek care, in language the patient can understand.

### 1.3 Solution

- **Routine, low-cost-blood-test-based, cross-sectional screening:** the ML model is trained entirely on features that are already collected in a standard general checkup or population health survey — vitals, comorbidity history, and common blood tests (glucose, HbA1c, BUN, uric acid, albumin, hemoglobin) — deliberately excluding creatinine, eGFR, and UACR. It is *cross-sectional* in the epidemiological sense: each screening is a single point-in-time risk estimate, trained on cross-sectional NHANES survey data (one measurement per participant, not a repeated-measures/longitudinal design). This makes the tool usable anywhere a routine blood panel is already run, without ordering the additional specialized kidney-function test up front.
- **Explainable, personalized, safety-guaranteed education:** the LLM pipeline translates the ML result + patient profile into five-section patient education, grounded in retrieved clinical-guideline text, personalized to the patient's own reported values, and passed through a deterministic post-generation safety validator before delivery. This configuration has been fine-tuned (Phase 8, bf16 LoRA) and rigorously evaluated on a 300-profile, six-arm ablation ("Q1") that isolates the individual contribution of RAG, fine-tuning, the validator, and prompt compression (§7).

> **Scope disclaimer:** This is a screening and education tool. It does not diagnose CKD, prescribe treatment, or replace clinician judgment, eGFR/UACR testing, or medical evaluation.

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PATIENT INPUTS (14)                      │
│   Age, Income ratio, Waist, Systolic BP, Diastolic BP,          │
│   Glucose, HbA1c, BUN, Uric Acid, Albumin, Hemoglobin,           │
│   Diabetes, Hypertension, BP medicine                           │
└─────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼ POST /predict  or  POST /generate-guidance
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (backend/)                     │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │             TWO-STAGE ML CASCADE (Final Model C)           │  │
│  │                                                             │  │
│  │  Stage 1: Random Forest (At-risk gate, 13 features)         │  │
│  │     → P(at-risk); threshold 0.44; low → exit                │  │
│  │  Stage 2: Gradient Boosting (High-risk split, 6 features)    │  │
│  │     → P(high | at-risk); threshold 0.50                      │  │
│  │     → combined: low / moderate / high                        │  │
│  │                                                             │  │
│  │  Local Explanation (model-agnostic single-feature            │  │
│  │  perturbation against fixed clinical reference values)       │  │
│  │  Reliability heuristic (confidence / margin / missingness)   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│   risk_class · probabilities · top_factors · reliability ·       │
│   input_quality · patient_inputs  →  llm_context                 │
│                       │                                          │
│   (best-effort) ──────┼────────────────► PostgreSQL `screenings` │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM COMPONENT (llm/)                           │
│                                                                    │
│  ┌────────────────────┐    ┌────────────────────────────────┐    │
│  │   RAG RETRIEVER     │    │       PROMPT BUILDER            │    │
│  │ Query built from     │    │ Adapter active → COMPACT prompt│    │
│  │ risk class + top     │    │ Adapter absent → FULL prompt   │    │
│  │ factor phrases        │    │  - system: role/scope/safety   │    │
│  │ Embed: BGE-M3 (Ollama)│    │    rules/output format         │    │
│  │ Search: FAISS         │    │  - user: screening result +    │    │
│  │ Factor-aware:          │    │    PATIENT PROFILE + top       │    │
│  │  1 best chunk per      │    │    factors + retrieved         │    │
│  │  priority topic, then  │    │    chunks + per-section        │    │
│  │  fill to k              │    │    PERSONALIZATION cues        │    │
│  └──────────┬─────────────┘    └───────────────┬─────────────────┘  │
│             └───────────────────┬───────────────┘                    │
│                                  ▼                                    │
│                     ┌────────────────────────────┐                   │
│                     │   LLM GENERATION            │                   │
│                     │  Qwen3-4B-Instruct-2507      │                   │
│                     │  + LoRA adapter (Phase 8,     │                   │
│                     │  r=16/alpha=32, when active)  │                   │
│                     └──────────────┬───────────────┘                  │
│                                    ▼                                   │
│                     ┌──────────────────────────────┐                  │
│                     │   SAFETY VALIDATOR            │                  │
│                     │  Diagnosis / medication /      │                  │
│                     │  supplement blockers (severe)  │                  │
│                     │  Nutrition-restriction rewrite, │                 │
│                     │  lab reminder, emergency         │                 │
│                     │  redirect (moderate, auto-fix)    │                 │
│                     │  status: pass / sanitized /        │                │
│                     │          fallback                   │               │
│                     └──────────────┬───────────────────┘                 │
│                                    ▼                                      │
│                     ┌──────────────────────────────┐                     │
│                     │   IEEE CITATION LAYER           │                    │
│                     │  Deterministic [n] markers +      │                   │
│                     │  reference list, from sources       │                  │
│                     │  actually retrieved (never          │                  │
│                     │  model-invented)                     │                 │
│                     └──────────────┬───────────────────┘                    │
└────────────────────────────────────┼───────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND (frontend/)                    │
│  Patient View: Risk card · Reliability · Main signals ·          │
│                5-section education with citations                │
│  Admin View:  Probabilities · Explanation · RAG audit ·           │
│               Safety audit · Evaluation (ML + Q1 LLM) · Export    │
│  Demo Cases:  Low / Moderate / High-emergency — real inputs       │
│               run through the real live pipeline                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Deployment Modes

| Mode (`CKD_LLM_MODEL`) | LLM backend | Embedder | Requires |
|---|---|---|---|
| `local_demo` / unset | `MockGuidanceModel` (deterministic Python) | HashingTFIDF (pure Python, offline) | Nothing — runs anywhere |
| `ollama` | Qwen3-4B via Ollama REST API | BGE-M3 via Ollama REST API | Ollama running locally |
| `qwen` (**deployed default**) | Qwen3-4B-Instruct-2507 via HuggingFace `transformers`, optionally with the Phase-8 LoRA adapter (`CKD_LLM_ADAPTER`) attached via PEFT | BGE-M3 via Ollama (`ollama-bge-m3` embedder spec), backed by a real FAISS index | Ollama running (for the embedder) + a GPU for the LLM (CPU works but is slow) |

The active mode, model spec (including whether the LoRA adapter is attached, e.g. `qwen:Qwen/Qwen3-4B-Instruct-2507+lora:v1/adapter`), and prompt variant (`compact`/`full`) are all reported live via `GET /generate-guidance/health` and shown in the UI's System Health panel — never hardcoded or assumed. The safety validator runs identically in every mode.

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend API** | FastAPI + Uvicorn | REST API, routing, CORS, lifespan management |
| **ML Model** | scikit-learn + imbalanced-learn | Two-stage cascade training & inference |
| **Model serialization** | joblib | Plain-`dict` `.joblib` package: model, preprocessor, features, thresholds — no custom pickle classes |
| **Data processing** | pandas + numpy | Feature matrix construction, imputation |
| **Dense Embeddings** | BAAI/BGE-M3, served via **Ollama** (`ollama-bge-m3`) | Semantic query/chunk embedding for RAG, 1024-dim |
| **Lexical Embeddings (fallback only)** | Pure-Python HashingTFIDF (stdlib) | Offline demo-mode RAG, no dependencies |
| **Vector Store** | FAISS (production) / custom pure-Python brute-force (offline fallback) | Nearest-neighbor chunk search |
| **LLM (production)** | Qwen3-4B-Instruct-2507 via HuggingFace `transformers` | Guideline-grounded education generation |
| **LLM fine-tuning** | LoRA adapters via `peft`, trained in **bf16 without quantization** (the script also supports a 4-bit QLoRA mode, not used for the reported adapter) | Phase 8: internalizes format/safety/personalization, enables the compact prompt |
| **LLM (offline demo)** | `MockGuidanceModel` (deterministic Python) | Full-pipeline testing with no GPU/network |
| **Safety Validator** | Pure Python (stdlib + project modules only) | Deterministic post-generation rule enforcement |
| **Citation layer** | Pure Python, deterministic | IEEE-style `[n]` markers + reference list from KB `sources.csv` |
| **Frontend** | React 18 + Vite + Tailwind CSS | Patient UI + Admin/Research dashboard |
| **Frontend state** | React hooks (`useState`, `useEffect`) | Local state, no Redux |
| **API client** | Fetch-based client (`frontend/src/api/client.js`) | Frontend → backend communication |
| **Database** | PostgreSQL 18 (configured, live; best-effort at runtime) | Research persistence of every screening + guidance call |
| **Schema migration** | Alembic | Explicit, version-controlled schema (`CKD_DB_AUTO_CREATE=false` once migrations are in use) |
| **Config** | python-dotenv + Vite env | Backend and frontend environment configuration |
| **Platform** | Windows 10 Pro (primary, dev machine) / cross-platform | Runs on any OS |
| **GPU / compute** | NVIDIA GeForce RTX 3090, 24 GB VRAM; PyTorch 2.5.1+cu121; transformers 5.14.1 | Fine-tuning, Q1 evaluation and live serving all ran on this card; inference in bfloat16 (no quantization). See [HARDWARE_AND_ENVIRONMENT.md](HARDWARE_AND_ENVIRONMENT.md) |
| **Training data** | NHANES 2021–23 (CDC/NCHS) | Population-level health survey (US) |

### 3.1 Key Python Dependencies

```
fastapi, uvicorn, python-dotenv, joblib, scikit-learn, imbalanced-learn,
pandas, numpy, requests            # core serving + ML
sqlalchemy, psycopg2-binary, alembic  # database
torch 2.5.1+cu121, transformers 5.14.1, peft 0.19.1, accelerate 1.14.0, faiss 1.14.3  # production LLM + fine-tuning + RAG
# bitsandbytes is only needed for the optional 4-bit mode of qlora_train.py; it is not installed or used.
```

### 3.2 Frontend Dependencies

```
react 18.x, react-dom 18.x, vite 5.x, tailwindcss 3.x
```

---

## 4. How It Works — End-to-End

This section follows the request from the very first field a patient types, through to the guidance text rendered on screen, answering explicitly: *how the input is processed, what the model produces and how, what happens before the LLM ever sees the case, how the LLM is prompted and run, whether its output is truly personalized, and how every technology choice fits into that chain.*

### 4.1 Input Processing

**Where:** `frontend/src/components/patient/PatientForm.jsx` → `POST /predict` or `POST /generate-guidance` → `backend/schemas.py` → `backend/model_service.py`.

1. **Client-side collection.** The patient form (`featureSchema.js`) collects exactly the **14 features** the deployed model consumes — no more, no fewer. Each field carries a declared numeric range or binary flag; the form computes a live "completeness" percentage as the patient fills it in.
2. **Schema validation.** The Pydantic model `PatientFeatures` (`backend/schemas.py`) enforces `extra="forbid"` (no stray fields silently accepted) and a declared `ge`/`le` numeric range or 0/1 constraint per field. A malformed request is rejected with a 422 before it ever reaches the model.
3. **Consistency rule.** `model_service.py::_clean_payload()` applies one structural rule that mirrors how the training data was prepared: if a patient reports no diagnosed hypertension and leaves "currently on BP medicine" blank, that field defaults to 0 (not-taking) rather than staying missing — this matches how the same structural-missing case was handled at training time, so inference doesn't introduce a distribution shift the model never saw.
4. **Missing-value handling.** Any field left blank becomes `NaN` in the feature row (`make_dataframe()`), then flows into each stage's own fitted `ColumnTransformer` preprocessor (loaded from inside the joblib package), which imputes it (median for numeric, most-frequent for binary — whatever was fit at training time). The count and identity of every imputed field is preserved and returned in `input_quality`, so nothing is silently guessed without a trace.

### 4.2 ML Prediction Mechanism

**Where:** `backend/model_service.py::_score_df()`.

The deployed model is **Final Model C**, a two-stage cascade, loaded once at startup from `model_package/FINAL_MODEL_C_FULL_DEVELOPMENT.joblib` (a plain Python `dict`, not a custom class — safer to deserialize and easier to inspect).

```
Stage 1 — Random Forest "at-risk gate" (13 features)
    p1 = P(at-risk)                      [positive-class probability]
    gate = p1 >= stage1_threshold (0.44)

Stage 2 — Gradient Boosting "high-risk discriminator" (6 "direct" features:
          blood_urea_nitrogen, uric_acid, hemoglobin, serum_albumin, mean_sbp, age)
    p2 = P(high | at-risk)               [computed for every row, gated below]

Decision:
    risk_class = "low"      if not gate
                 "high"     if gate and p2 >= stage2_threshold (0.50)
                 "moderate" otherwise

Class probabilities (row-normalized):
    p_low      = 1 - p1
    p_moderate = p1 * (1 - p2)
    p_high     = p1 * p2
    confidence = max(p_low, p_moderate, p_high)
```

**Why a cascade instead of one multi-class model:** NHANES risk classes are heavily imbalanced (far more "low" than "high"). A single 3-class classifier tends to optimize aggregate accuracy at the expense of the clinically worst error — calling a genuinely high-risk patient "low." Splitting the decision lets Stage 1 be tuned purely for **catching every at-risk case** (high sensitivity gate) while Stage 2 is tuned to **correctly separate high from moderate** among those already flagged (higher specificity where it matters). Result on the held-out test set: only **2 high→low misses**.

**Two things computed alongside the prediction, before anything goes to the LLM:**

- **Local explanation** (`_local_explanation()`): for each supplied feature with a defined clinical reference value, the service re-scores the *entire cascade* with that one feature swapped to its reference value, and records how much the predicted class's probability shifted. Features are ranked by `|shift|`; the top 8 are kept (the frontend shows the top 5 to patients, all 8 to admins). This is a genuine, model-agnostic local-perturbation explanation — not SHAP, not causal, explicitly labeled as such in the API response (`explanation_method: "model-agnostic local perturbation against reference input values"`).
- **Reliability heuristic** (`_reliability()`): a deterministic rule combining model confidence, the probability margin between the top two classes, proximity to the Stage-1 decision boundary, and the count of missing/imputed features, into a `low`/`moderate`/`high` reliability label plus explanatory notes. See `docs/FRONTEND_ELEMENTS_JUSTIFICATION.md` §2 for the exact thresholds.

### 4.3 Between Prediction and the LLM

**Where:** `backend/model_service.py::_build_llm_context()` and `predict()`.

Before anything reaches the LLM component, the backend assembles an `llm_context` object that becomes the **frozen, self-contained input contract** for the rest of the pipeline (`llm/serving/contract.py::LLMInput.from_predict_response()`):

```json
{
  "purpose": "Generate non-clinical, supportive lifestyle guidance...",
  "strict_safety_rules": [ /* 9 rules, mirrored from llm/serving/prompts.py */ ],
  "model_output": {
    "screening_category": "high",
    "class_probabilities": {"low": 0.05, "moderate": 0.15, "high": 0.80},
    "confidence": 0.80,
    "reliability_level": "high",
    "reliability_notes": [ "..." ]
  },
  "key_model_factors": [ /* top 6 of the 8 explanation factors */ ],
  "input_quality": { "n_supplied_features": 14, "n_missing_features": 0, ... },
  "patient_inputs": { "age": 65, "mean_sbp": 160, "diabetes_yes": 1, ... },
  "suggestion_style": { "tone": "...", "sections": [...], "avoid": [...] }
}
```

Two properties of this handoff matter for the rest of the pipeline:
1. **The full raw patient profile is carried forward, not just the ML explanation.** The explanation factors say *which inputs drove the classifier's decision*; `patient_inputs` carries *everything the patient actually reported*, some of which is lifestyle-relevant even if it wasn't a top statistical driver (e.g., a diagnosed comorbidity that the model weighted lightly but which still deserves a mention in food/routine advice). Both are needed for genuine personalization (§4.7).
2. **This object is what gets persisted** — `backend/database/repository.py::save_screening()` writes the prediction (and, for `/generate-guidance` calls, the resulting guidance) to the PostgreSQL `screenings` table, best-effort (a database outage never blocks or fails a screening request — see §14).

### 4.4 Model Output → LLM Pipeline

**Where:** `llm/serving/llm_runtime.py::GuidanceRuntime.generate_guidance()`.

Once `LLMInput` is built, four things happen in order, every request:

1. **Retrieval query construction** (`Retriever.build_query_from_llm_input`): the top 4 explanation factors and their human-readable phrases, plus the risk class, become a short natural-language query (e.g. *"high kidney disease risk. Mean systolic BP 160 mmHg. HbA1c 8.5%. high blood pressure. high blood sugar diabetes. kidney health lifestyle education"*).
2. **Embedding + search:** the query is embedded with **BGE-M3** (served through Ollama's own runtime, confirmed genuinely active — not the lexical fallback — via direct index metadata inspection during the full-project audit) and matched against the knowledge-base index with **FAISS**.
3. **Factor-aware chunk selection** (`Retriever.retrieve_for_llm_input`): rather than returning the raw top-k by similarity (which collapses to the same generic chunks for every patient at a given risk level), the retriever maps each of the patient's top factors to a KB topic (blood pressure → `bp`, HbA1c/diabetes flag → `diabetes`, etc., in factor-importance order), takes the single best-scoring chunk per topic, then fills any remaining slots with the next-best leftover chunks. For an at-risk patient, a `followup` (lab/clinician) chunk is always guaranteed a slot. This is the mechanism that makes two patients at the *same* risk class receive *different* guideline grounding, matched to their own driving factors.
4. **Prompt assembly and generation** — described in full in §4.5–§4.6.

### 4.5 How the LLM Works — Prompt Engineering

**Where:** `llm/serving/prompts.py`.

Two complete system-prompt variants exist, and exactly one is used per request, chosen automatically by whether a fine-tuned adapter is attached (`llm_runtime.py:101-106`):

**Full prompt** (`build_system_prompt()`, used when no adapter is loaded — the base, un-fine-tuned model): a long, explicit system prompt containing —
- A scope statement (education, not diagnosis/treatment/prescription).
- **9 numbered safety rules** verbatim-shared with the safety validator's own design (no diagnosis; no medication/supplement start-stop-change-dose advice; no individualized numeric nutrient targets — defer to a dietitian; never substitute for eGFR/UACR; *personalize* wellness content to the patient's actual conditions; mandatory lab/clinician reminder for moderate/high; urgent-care redirect for red-flag symptoms/vitals; ground answers in the supplied guideline excerpts; write at roughly an 8th-grade reading level).
- A literal **output-format specification**: five labeled sections in a fixed order (`Summary`, `Food habits`, `Rest and sleep`, `Daily routine`, `Follow-up reminder`), each with a one-line content hint, followed by a machine-readable `grounding:` line listing the guideline-excerpt ids actually used.

**Compact prompt** (`build_compact_system_prompt()`, used automatically once `CKD_LLM_ADAPTER` is set): a much shorter system prompt covering the same rules and format in a fraction of the tokens. This is only viable because the **fine-tuned model has internalized the full-prompt behavior during training** (§4.6) — every Phase-7 training example pairs the compact prompt with a gold answer that was itself generated and safety-checked under the *full* prompt, i.e. the fine-tune is a **prompt-compression distillation objective**, not just a style transfer. Measured effect on the deployed system: a **25% reduction in prompt tokens** versus running the fine-tuned model with the full prompt, with no confirmed safety cost (Q1 F-vs-E comparison, §7).

**The user turn** (`build_user_prompt()`, identical structure regardless of which system prompt is active) contains, in order:
1. An **urgent vitals alert block** (only emitted when systolic ≥180 or diastolic ≥120 mmHg) that names the patient's *actual* triggering reading rather than only citing the abstract rule — added after observing the model anchor on systolic as "the" blood-pressure number and under-react to a diastolic-only emergency.
2. The **screening result**: risk level, class probabilities, confidence, reliability level, and an explicit completeness note.
3. The **PATIENT PROFILE** — every one of the patient's raw reported values, human-labeled with units (e.g. "Systolic blood pressure: 160.0 mmHg", "Has diabetes (diagnosed): Yes"). This is not the top-factor list; it's the full picture, deliberately, per §4.3.
4. **TOP FACTORS THAT MOST INFLUENCED THIS RESULT** — the ML explanation's top factors, each phrased as raising or lowering the predicted risk.
5. **GUIDELINE EXCERPTS** — the retrieved chunks, each tagged with its chunk id and source, and a `[RESTRICTED: mention only as 'discuss with a clinician/renal dietitian']` tag on any chunk containing individualized numeric nutrition content (see §5.2) — a structural nudge the model is instructed to respect regardless of what the retrieved text itself says.
6. A **PERSONALIZATION REQUIREMENTS** block — explicit, condition-derived, per-section instructions computed in Python from the patient's actual flags and values (`_build_personalization_cues()`), e.g. *"Food habits: focus on low-sugar, low-refined-carb choices because of elevated blood sugar; suggest reducing salt intake because of high blood pressure."* This exists because early testing showed that telling the model "personalize it" in the abstract was too vague — it produced near-identical text for every patient at the same risk level. Stating exactly *what fact to reference in which section* closes that gap. This logic uses simple, transparent threshold rules on the raw inputs (not the model's calibrated probabilities) — sufficient for a prompt-hint purpose, and it is not presented anywhere as a second predictive model.

### 4.6 LLM Generation

**Model:** Qwen3-4B-Instruct-2507 (non-thinking instruct variant), loaded via HuggingFace `transformers` (`llm/serving/model_backends.py::QwenChatModel`) with `torch_dtype="auto"`, which resolves to the checkpoint's native **bfloat16** — measured: all base weights bf16, no quantization, fully on the GPU, 7.62 GiB VRAM with the adapter attached. Live-serving generation parameters: `max_new_tokens=700`, `temperature=0.3`, `top_p=0.9` — a fairly low temperature, chosen to keep the guidance consistent and reduce the chance of drifting into unsafe or off-topic phrasing, while still allowing natural variation across patients. The Q1 evaluation instead used **greedy (deterministic) decoding**, seed 42, so its results are exactly reproducible.

**Hardware and versions:** NVIDIA GeForce RTX 3090 (24 GB), PyTorch 2.5.1+cu121, transformers 5.14.1, PEFT 0.19.1 — used for fine-tuning, the Q1 evaluation and live serving alike. Full verified record: [HARDWARE_AND_ENVIRONMENT.md](HARDWARE_AND_ENVIRONMENT.md).

**Fine-tuning (Phase 8):** the base model was fine-tuned with a **LoRA adapter** (rank `r=16`, `alpha=32`, dropout `0.05`, targeting all seven attention/MLP projection matrices — `q/k/v/o_proj`, `gate/up/down_proj`) in **bf16 without quantization** (`--precision bf16`, optimizer `adamw_torch`; the phase and script are named "QLoRA" after the script's default 4-bit mode, which was not used for the reported adapter), for **3 epochs** at learning rate `2e-4` (cosine schedule, 3% warmup), effective batch size 16 (2 × 8 gradient-accumulation), on **540 training examples** (60 held out for validation), with the loss masked to only the assistant's answer tokens. Final train loss 0.133, best validation loss 0.155. The adapter is ~132 MB and loads on top of the frozen base weights via `peft.PeftModel.from_pretrained()`.

**Training data provenance (Phase 7):** 600 synthetic, class-balanced (200/200/200 low/moderate/high) patient profiles, each with a gold answer generated under the *full* prompt via **self-distillation** — i.e., the same base Qwen3 + RAG + safety-validator pipeline generating its own training targets, with every single one required to pass `scan_unsafe` (0 unsafe) and contain all 5 required sections before being kept (100% pass rate on this run; the architecture also supports an external teacher (Gemini) for diversity, but the actual v1 adapter run used 100% self-distillation, `teacher_mix: {"self": 600}`). No real patient data was used; the 30 Phase-6 held-out evaluation profiles and the ML test set were explicitly excluded from the training pool to avoid leakage.

**Retry logic:** if a generation fails the validator's structural check (a required section missing), the same prompt is re-sent once with an appended format reminder before falling back to the safe generic template. Measured retry rate across the Q1 benchmark: effectively 0.

**Offline/demo mode:** `MockGuidanceModel` — a deterministic Python function that reads the same patient-profile flags out of the rendered prompt and produces condition-aware (but not naturally-language-varied) text, requiring no GPU, no network, and no model weights. It exists purely so the full retrieve→prompt→validate pipeline is testable and demonstrable on any machine; it is not used for any reported evaluation number.

### 4.7 Personalized or Generalized?

**Answer: genuinely personalized, by construction and by measurement, not merely by claim.** Three independent mechanisms make this true, and one metric measures it directly:

1. **Different retrieval per patient** (§4.4) — two patients at the same risk class but different driving factors receive different guideline chunks, because retrieval is keyed to the patient's own top factors, not just their risk class.
2. **Full patient profile in the prompt** (§4.5) — the model sees the patient's actual reported values, not just an abstract "moderate risk" label.
3. **Explicit per-section personalization cues** (§4.5) — computed in Python from the patient's specific flags, telling the model exactly what to reference in each of the five sections.
4. **Measured, not assumed:** the Q1 evaluation's `personalization_mean` metric (a 2-point rubric score for whether generated text actually references the patient's own factors rather than generic language) scores **1.99 / 2** for the deployed configuration across 300 profiles (§7) — i.e., near-ceiling, verified on data, not only architected for.

What is **not** personalized, and shouldn't be presented as such: the **safety rules, output format, and reading-level target are identical for every patient** — personalization applies to the *content* of the advice, never to whether a rule applies. A patient in the emergency subgroup and a patient with only mildly elevated glucose are held to the exact same validator, the same diagnosis/medication blockers, and the same five-section structure.

### 4.8 Safety Validation

**Where:** `llm/serving/safety_validator.py` — runs on every LLM output, in every mode, before it can reach a user.

| Layer | Trigger | Severity | Action |
|---|---|---|---|
| Diagnosis blocker | "you have CKD / kidney disease / kidney failure" phrasing | Severe | Regenerate → safe fallback template |
| Medication blocker | Named drugs/classes, dosing language, start/stop/change instructions | Severe | Regenerate → safe fallback template |
| Supplement blocker | "take fish oil / vitamin D / probiotics" etc. | Severe | Regenerate → safe fallback template |
| Nutrition-restriction rewrite | Numeric protein/potassium/phosphorus/sodium target | Moderate | Auto-rewritten to a dietitian-referral sentence |
| Lab-confirmation reminder | Moderate/high result missing a clinician + lab-test mention | Moderate | Auto-appended |
| Emergency-symptom/vitals redirect | SBP ≥180 or DBP ≥120, or urgent symptom phrasing | Moderate | Urgent-care line auto-added |
| Structural/format check | Missing required section | — | Triggers the one retry described in §4.6 |

**Outcome states:** `pass` (returned as-is), `sanitized` (moderate issues auto-fixed, fixed text returned), `fallback` (severe or structurally broken output replaced entirely by a safe, condition-aware generic template). Every decision is logged with the exact rule/category that fired, and surfaced verbatim in the admin Safety panel — nothing about the validator's behavior is hidden from inspection.

**Known, documented scope limitation:** the emergency-symptom vocabulary check (§ above) is real and tested, but the live app's current input schema has no free-text symptom field — so in production today, emergency detection fires only via the numeric-vitals path. The symptom-phrase vocabulary is fully exercised by the Q1 evaluation harness (which injects symptom text explicitly) but not by any current production request. This is recorded honestly in `results/q1_llm_evaluation_summary.md` §10 and is not glossed over.

After the validator runs, a **deterministic IEEE citation layer** (`llm/serving/citations.py`) maps each output section to the guideline sources actually retrieved for it, inserts `[n]` markers, and builds a numbered reference list — a `[n]` can never point to a source the model wasn't actually given, which rules out invented citations by construction.

### 4.9 Frontend Rendering

The React frontend has two audiences:

**Patient View** — a calm, non-technical result: risk card, reliability badge with a completeness bar, a "Main signals" chip row (top 5 explanation factors, plain-language), the five-section education text with inline citation markers and a reference list, and a safety-note card restating the education-only scope.

**Admin / Research View** — the full technical audit trail: raw probabilities and both stage outputs, input-quality breakdown (missing/out-of-range), the full 8-factor explanation table, the RAG panel (query string, retrieved chunks with similarity scores, grounding-validity ratio, raw pre-validator text, final post-validator JSON), the Safety panel (each layer's pass/fail state), a System Health panel (live backend/model/DB/adapter status), and an Evaluation tab showing the frozen ML test metrics and the final Q1 LLM ablation results (§7) — every number on this tab is traced in `docs/FRONTEND_ELEMENTS_JUSTIFICATION.md`.

**Demo Cases** — one-click Low / Moderate / High-emergency example inputs. These are real payloads run through the exact same live `/generate-guidance` call as a manually filled form; there is no pre-baked canned output anywhere in the codebase for these cases.

---

## 5. Knowledge Base

### 5.1 Sources

| Source ID | Document | Organization | Year | Topics |
|---|---|---|---|---|
| KDIGO2024 | KDIGO 2024 Clinical Practice Guideline for CKD Evaluation and Management | KDIGO | 2024 | general, bp, activity, weight, followup |
| KDIGO2021BP | KDIGO 2021 Clinical Practice Guideline for BP Management in CKD | KDIGO | 2021 | bp, activity, sodium |
| ADAKDIGO2022 | Diabetes Management in CKD — ADA/KDIGO Consensus Report | ADA & KDIGO | 2022 | glucose, diabetes |
| ADA2026 | ADA Standards of Care in Diabetes 2026 — Ch.11 CKD and Risk Management | ADA | 2026 | general, bp, diabetes, activity, followup |
| KDOQI2020NUTR | NKF KDOQI 2020 Clinical Practice Guideline for Nutrition in CKD | NKF (KDOQI) | 2020 | nutrition (**restricted** source — see §5.2) |
| NKFPED | NKF — Exercise and Chronic Kidney Disease (patient education) | NKF | 2026 | activity, weight |
| CDC_CKD_BASICS | CDC Chronic Kidney Disease Basics | CDC | 2024 | general, prevention, bp, diabetes, activity, weight, followup |
| CDC_CKD_PREVENT | CDC Preventing Chronic Kidney Disease | CDC | 2024 | general, prevention, bp, diabetes, followup |
| CDC_CKD_RISK | CDC Risk Factors for Chronic Kidney Disease | CDC | 2024 | general, risk, bp, diabetes, weight |
| SYK2015 | Save Your Kidneys — Complete Guide for Kidney Patients (2nd Ed.) | Samarpan Kidney Foundation | 2015 | prevention, bp, diabetes, weight, activity, nutrition, hydration, followup |

**Total chunks:** 162 (grew from the original 118 as the source set was expanded with ADA2026 and SYK2015 during the project). **Chunk length:** ~150 words each. **Safety labels:** `safe_public` or `restricted_clinician_referral`.

Two additional entries exist in the source registry but are **not** collected/active: `CDCKIDNEY` (a legacy placeholder, superseded by the three current CDC sources) and `KDIGO2012` (kept only as background justification for the KDIGO risk-grid target definition, not used for retrieval).

### 5.2 Safety Scoping of the Knowledge Base

Chunks containing **numeric individualized targets** for protein, potassium, phosphorus, or sodium (overwhelmingly from the KDOQI nutrition source) are flagged `restricted=True` at ingestion time. When such a chunk is retrieved, it is still given to the model — but tagged inline as `[RESTRICTED: mention only as 'discuss with a clinician/renal dietitian']`, and the safety validator independently rewrites any numeric nutrient target that slips into the output regardless of what the retrieved text said. This two-layer defense (prompt-level tag + post-generation rewrite) implements the KDOQI constraint that individualized nutrient targets must never be given to a patient without dietitian involvement.

### 5.3 Topic Tags

`bp` · `diabetes` · `nutrition` · `weight` · `activity` · `smoking` · `sleep` · `followup` · `general` · `hydration` · `prevention` · `risk`

These tags drive the factor-aware retrieval described in §4.4.

---

## 6. API Reference

All endpoints are served by FastAPI at `http://localhost:8000`. Interactive Swagger docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root — links to docs, health, predict, guidance |
| `GET` | `/health` | Backend + model status, runtime versions, database status |
| `GET` | `/db/health` | Database connectivity + stored record count |
| `GET` | `/features` | Feature schema, metadata, numeric/binary lists |
| `GET` | `/model-info` | Model thresholds, selected feature lists |
| `POST` | `/predict` | Two-stage cascade for one patient — risk class, probabilities, explanation, reliability, `llm_context` |
| `POST` | `/reload-model` | Hot-reload the model joblib (for model updates) |
| `GET` | `/generate-guidance/health` | LLM mode, model spec (adapter status baked in), active prompt variant |
| `POST` | `/generate-guidance` | Full pipeline: predict → RAG → LLM → validator → citations — returns guidance sections, references, provenance, full audit trail |

### 6.1 GET /generate-guidance/health (example, adapter active)

```json
{
  "configured_model": "qwen",
  "runtime_loaded": true,
  "model_spec": "qwen:Qwen/Qwen3-4B-Instruct-2507+lora:v1/adapter",
  "mode": "production",
  "configured_adapter": "llm/finetune/adapters/v1/adapter",
  "adapter_active": true,
  "prompt": "compact"
}
```

### 6.2 POST /generate-guidance (response, abbreviated)

```json
{
  "risk_class": "high",
  "mode": "production",
  "status": "sanitized",
  "disclaimer": "This assistant provides personalized, guideline-grounded kidney-risk education...",
  "guidance": {
    "Summary": "Your screening shows a high risk level [1]...",
    "Food habits": "Since your blood pressure is elevated, try to reduce salt [2]...",
    "Rest and sleep": "Aim for 7 to 9 hours of sleep each night...",
    "Daily routine": "Check your blood pressure regularly and log the readings...",
    "Follow-up reminder": "Ask for a urine test and blood test for kidney function [3]..."
  },
  "references": [
    {"n": 1, "citation": "KDIGO, \"KDIGO 2024 Clinical Practice Guideline...,\" 2024.", "url": "https://kdigo.org/..."}
  ],
  "provenance": ["KDIGO", "CDC"],
  "retrieved_chunk_ids": ["KDIGO2024_bp_001", "KDIGO2024_diabetes_003", "..."],
  "model_cited_grounding": ["KDIGO2024_bp_001", "KDIGO2024_diabetes_003"],
  "grounding_valid_ratio": 1.0,
  "query": "high kidney disease risk. Mean systolic BP 160 mmHg...",
  "violations": [{"category": "emergency_redirect", "severity": "moderate", "layer": "rule", "detail": "..."}],
  "n_attempts": 1,
  "model_spec": "qwen:Qwen/Qwen3-4B-Instruct-2507+lora:v1/adapter",
  "prediction": { "...full /predict output..." }
}
```

---

## 7. Evaluation Results

### 7.1 ML Two-Stage Cascade

**Dataset:** NHANES 2021–23, held-out test split (n=833), evaluated on the exact **Full-development deployment** configuration that produced the currently-deployed `FINAL_MODEL_C_FULL_DEVELOPMENT.joblib` (training N = 4,719). These numbers were independently re-verified against the training notebook during a full-project audit and match to 4–6 significant figures.

| Metric | Value | Meaning |
|---|---:|---|
| Accuracy | 0.7095 | Correct predictions / total |
| Balanced accuracy | 0.6602 | Mean of per-class recall — corrects for class imbalance |
| Macro F1 | 0.5561 | Unweighted mean of per-class F1 |
| Low recall | 0.7391 | Of true "low" cases, fraction correctly identified |
| Moderate recall | 0.5048 | Of true "moderate" cases, fraction correctly identified |
| High recall | 0.7368 | Of true "high" cases, fraction correctly identified |
| **High → Low misses** | **2** | Genuinely high-risk patients told "low" — the single most safety-critical error type |

### 7.2 LLM — Final Q1 Evaluation (Six-Arm Ablation)

The Phase-6 30-profile evaluation referenced in earlier project documentation has been **superseded** by a much larger, more rigorous evaluation ("Q1"): a **300-profile frozen benchmark**, built by rejection sampling *against the real deployed cascade* (so every profile's stated risk label is one the production model would actually assign), split into 6 subgroups × 3 risk classes × 50 profiles, including a 60-profile emergency subgroup and a 30-profile adversarial (explicit medication-request attack) subgroup.

**Six arms isolate each component's contribution:**

| Arm | Prompt | RAG | Validator | Model |
|---|---|---|---|---|
| A | compact | no | no | base |
| B | full | yes | no | base |
| C | full | yes | yes | base |
| D | full | yes | no | fine-tuned |
| E | full | yes | yes | fine-tuned |
| **F (deployed)** | **compact** | **yes** | **yes** | **fine-tuned** |

**Primary safety outcomes (n=300 per arm):**

| Arm | Raw critical-unsafe | Delivered critical-unsafe | Emergency-redirect recall | Adversarial attack success |
|---|---:|---:|---:|---:|
| A | 8.33% | 8.33% | 65.0% | 6.67% |
| B | 10.33% | 10.33% | 75.0% | 0% |
| C | 10.33% | **0.0%** | **100.0%** | 0% |
| D | 7.33% | 7.33% | 65.0% | 0% |
| E | 7.33% | 1.33% | 100.0% | 0% |
| **F (deployed)** | 6.33% | **0.0%** | **100.0%** | 0% |

**Statistically confirmed (McNemar's test, Holm-corrected within each pre-declared metric family, Wilson confidence intervals for proportions):** the safety validator produces a large, Holm-significant reduction in delivered critical-unsafe rate and a large, Holm-significant increase in emergency-redirect recall, for **both** the base model (C vs B, p<0.001) and the fine-tuned model (E vs D, p<0.001). This is the strongest, most decisive result in the evaluation: **both the base-model validated arm (C) and the deployed configuration (F) reach zero critical-unsafe deliveries and 100% emergency-redirect recall across all 300 profiles.**

**Quality and grounding (deployed configuration, Arm F):**

| Metric | Value |
|---|---:|
| Format adherence | 100% |
| Grounding (cited ≥1 real source) | 99.7% |
| Personalization (2-point rubric) | 1.99 / 2 |
| Factor relevance | 0.843 |
| Readability ≤ grade 8 | 58.7% |
| Prompt-token reduction vs. full-prompt fine-tuned arm (E) | 25% |

**Not statistically confirmed, reported honestly:** RAG alone does not significantly change raw safety (B vs A). LoRA's raw-safety effect is directionally favorable but does not clear Holm correction on its own (D vs B). The compact prompt shows no statistically confirmed safety cost versus the full prompt (F vs E) — both arms are near the safety ceiling on this benchmark, so there is little room left to detect a difference.

**Caveats, stated proactively:** the 300 profiles are synthetic, scored by the deployed cascade rather than drawn from real patients; safety detection (both Q1's scoring and the production validator) is rule-based and reproducible but not a clinical adjudication; no clinician review of generated guidance has been performed. Full methodology, every statistic, and the complete correction history (ten scoring false-positive fixes and two production-validator fixes found and corrected during this evaluation, documented transparently) are in `results/q1_llm_evaluation_summary.md`.

---

## 8. Design Decisions & Tradeoffs

### 8.1 Two-Stage Cascade vs. Single Multi-Class Classifier

**Decision:** Random Forest gate → Gradient Boosting discriminator, rather than one three-class classifier. **Rationale:** severe class imbalance in NHANES; a single model tends to under-catch high-risk cases. **Tradeoff:** two models to train/validate/serialize, two decision thresholds to tune, a slightly more complex inference path — accepted because it directly reduced the clinically worst error (high→low misses to 2).

### 8.2 Routine, Low-Cost Blood-Test Feature Set (No eGFR/Creatinine)

**Decision:** exclude creatinine, eGFR, UACR, and urine protein entirely; use only routine, low-cost, commonly-ordered blood values (glucose, HbA1c, BUN, uric acid, albumin, hemoglobin) plus vitals and comorbidity history. **Rationale:** eGFR/creatinine/UACR are gold-standard *confirmatory* markers, but they require a kidney-specific lab order that is not part of a standard general checkup — using them as inputs would mean the screening step needs the very specialized test it's meant to triage toward. Building on values already covered by a routine blood panel makes the tool usable anywhere that panel is already run, with no additional specialized draw. **Tradeoff:** lower ceiling accuracy (71.0%) than an eGFR-based classifier — acceptable because the system is explicitly a triage/education trigger, not a diagnostic replacement.

### 8.3 Retrieval-Augmented Generation over Fine-Tuning Alone

**Decision:** ground the LLM's factual content in retrieved guideline chunks rather than relying solely on what fine-tuning bakes in. **Rationale:** guidelines change (KDIGO updates periodically) — updating RAG means re-indexing, not retraining; RAG is traceable and auditable (every citation maps to a real retrieved source); a model fine-tuned to "know" guidelines by heart risks confidently hallucinating outdated or invented specifics. **Tradeoff:** retrieval adds latency and a possible source of irrelevant chunks — mitigated by the factor-aware retrieval strategy and the grounding-validity check.

### 8.4 Fine-Tuning as a Complement, Not a Replacement, for RAG and the Validator

**Decision:** add LoRA fine-tuning (Phase 8) on top of the already-RAG-grounded, validator-guarded base pipeline, rather than treating fine-tuning as an alternative safety mechanism. **Rationale:** the Q1 ablation shows fine-tuning's own raw-safety effect is directionally favorable but not independently significant (D vs B) — its real, measured value is enabling the **compact prompt** (25% token reduction) with a validator-confirmed lack of safety regression (F vs E), plus real quality gains (grounding rose from ~68% under the full prompt to 99.7% under the compact prompt it was actually trained for). **Tradeoff:** fine-tuning requires a training pipeline, GPU time, and a held-out data-generation phase (Phase 7) — justified by the token-cost saving and quality improvement it delivers on top of the safety guarantees that RAG + the validator already provide independently.

### 8.5 Rule-Based Safety Validator over LLM-as-Judge

**Decision:** deterministic regex/heuristic validator rather than a second LLM judging the first. **Rationale:** a rule that blocks "you have CKD" is unconditionally reliable; deterministic rules are auditable (every decision traces to the exact rule that fired) and add no extra model-call latency; a medical-safety gate benefits from a hard guarantee more than from a probabilistic judgment. **Tradeoff:** rules cannot catch a genuinely novel unsafe phrasing outside their patterns — the Q1 evaluation's own detector-bug-fixing process (§7) is itself evidence of this tradeoff in action: several rounds of real bugs (missed negation, over-broad regex spans) were found and fixed by testing against real model output, exactly the kind of blind spot a purely rule-based approach can have. An optional LLM-judge layer is architected in the codebase but not activated in the deployed configuration.

### 8.6 Real BGE-M3 via Ollama, Not the Sentence-Transformers Path

**Decision:** serve the BGE-M3 embedder through Ollama's own runtime rather than `sentence-transformers`/`torch` directly. **Rationale:** the installed torch version (2.5.1) is below the 2.6 floor required by a `torch.load` security restriction (CVE-2025-32434) — attempting to load `BGEM3Embedder` the direct way silently failed and fell back to a lexical hashing-TF-IDF embedder with only a printed warning, discovered during a project audit. Upgrading torch risked destabilizing the already-verified Qwen3+LoRA serving stack, so BGE-M3 was instead served via Ollama (`ollama pull bge-m3`), which sidesteps `torch.load` entirely. **Tradeoff:** an additional runtime dependency (Ollama must be running for the embedder, independent of whether it's also serving the LLM) — accepted because it fully restores genuine semantic retrieval without touching the LLM serving path.

### 8.7 PostgreSQL for Research Persistence, Best-Effort

**Decision:** persist every screening/guidance call to PostgreSQL, but make the database entirely optional at runtime (`backend/database/db.py::init_db()` degrades gracefully — the API never blocks or fails a user-facing request because the database is unreachable). **Rationale:** research reproducibility benefits from a durable, queryable record of every prediction and guidance run, but a demo/thesis-defense environment should never fail a live request because of a database hiccup. **Tradeoff:** persistence is not guaranteed — an outage silently skips saving rather than retrying or queuing — acceptable for a research/demo tool, not acceptable as-is for a production clinical system.

### 8.8 Patient Profile and Explicit Personalization Cues in the Prompt

Covered in detail in §4.5 and §4.7 — retained here as a design decision: passing the raw profile (not just ML factors) and computing explicit per-section cues in Python were both adopted after observing that vaguer instructions produced generically similar text across patients. **Tradeoff:** longer prompts (mitigated by the compact-prompt fine-tune, §8.4) and some personalization logic duplicated in Python outside the ML model (acceptable, since it uses simple transparent threshold rules purely for prompt-hint purposes, not a second prediction).

### 8.9 Local Demo Mode

**Decision:** a fully offline `MockGuidanceModel` path, still passed through the real safety validator, labeled clearly as demo mode in the UI. **Rationale:** the system must be demonstrable on any machine, including one with no GPU or network. **Tradeoff:** demo text is deterministic and less natural than the real LLM — never used for any reported evaluation number.

---

## 9. Technology Stack — Deep Dive (Where / How / Why / Pros / Cons)

| Technology | Where used | How | Why chosen | Pros | Cons / Drawbacks |
|---|---|---|---|---|---|
| **scikit-learn (RandomForest + GradientBoosting)** | ML cascade, `model_package/*.joblib` | Two independently-fit stages inside one serialized `dict` package | Mature, interpretable, fast to train/serve, well-suited to tabular clinical data | Fast inference, no GPU needed, easy to inspect/debug, native `predict_proba` for calibrated-looking outputs | Ceiling accuracy below deep tabular methods on very large datasets; moderate-class recall (0.505) is the weakest link |
| **imbalanced-learn (RandomOverSampler / OBU resampling)** | Training-time only (baked into the saved model) | Rebalances the minority classes before fitting each stage | NHANES risk classes are heavily imbalanced | Meaningfully improves minority-class (moderate/high) recall vs. no resampling | Risk of overfitting the oversampled minority patterns if not paired with proper cross-validation (mitigated by the notebook's 5-fold × 3-repeat CV) |
| **joblib** | Model serialization (`model_package/*.joblib`) | Plain-`dict` package: preprocessor + model + thresholds + feature lists, no custom classes | Avoids fragile custom-pickle-class deserialization; the service trusts the package's own declared feature list over static config, preventing drift | Simple, fast load, self-describing package | Not a cross-language format; large files if models grow much bigger |
| **FastAPI + Uvicorn** | `backend/main.py` and all route modules | Async REST API, Pydantic request/response validation, lifespan-managed startup (`init_db()`) | Modern async Python framework with automatic OpenAPI docs and strong type validation | Fast, self-documenting (`/docs`), first-class Pydantic integration for strict schema enforcement | Async correctness discipline required if blocking ML/LLM calls aren't offloaded carefully |
| **Pydantic (`schemas.py`)** | Request/response validation | `extra="forbid"` + `ge`/`le` field constraints on all 14 features | Reject malformed input before it reaches the model, not after | Clear 422 errors, self-documenting schema | Schema must be kept in lock-step with `feature_config.py` and the trained model's own feature list (mitigated: `model_service.py` trusts the model package's own list at runtime) |
| **BGE-M3 (via Ollama)** | RAG query/chunk embedding (`llm/rag/embedder.py`, `ollama-bge-m3` spec) | Multilingual, multi-granularity dense embedding model, served through Ollama's REST API | State-of-the-art open dense embedder; serving through Ollama sidesteps a `torch.load` version restriction that broke the direct `sentence-transformers` path (§8.6) | Genuine semantic retrieval (not lexical matching), no GPU version conflicts, easy to swap/upgrade via Ollama | Requires Ollama running as an extra service; embedding quality still depends on chunk quality/granularity |
| **FAISS** | Vector search over the KB index | Exact/approximate nearest-neighbor search over the 162-chunk embedding matrix | Industry-standard, fast, well-tested vector search | Very fast at this KB scale, simple integration via the project's `VectorStore` wrapper | Overkill at 162 chunks (a brute-force fallback exists and is equally correct here) — its value is architectural headroom for a much larger KB |
| **Qwen3-4B-Instruct-2507** | LLM generation (`model_backends.py::QwenChatModel`) | Loaded via HuggingFace `transformers`, `device_map="auto"` | Strong instruction-following at a size (4B) that's fine-tunable and inference-affordable without a data-center GPU | Good instruction adherence, reasonable latency, open weights (auditable, self-hostable — no data leaves the machine) | Smaller than frontier closed models — quality/nuance ceiling is lower; still needs a capable GPU for comfortable latency |
| **PEFT (LoRA adapters, bf16)** | Phase-8 fine-tuning (`llm/finetune/qlora_train.py`, run with `--precision bf16`) | Freeze the bf16 base weights, train a small rank-16 adapter on top (no quantization — the 24 GB RTX 3090 held the full bf16 base) | Full fine-tuning of a 4B model is expensive and risks catastrophic forgetting of general instruction-following; LoRA trains ~a fraction of the parameters | Cheap to train (single run completed with a 24GB GPU tier), small adapter file (~132MB), swappable without touching the base model, enables the compact-prompt distillation objective | Adapter quality is bounded by training-data quality/size (540 examples here); LoRA can underperform full fine-tuning on tasks needing large representational shifts (not observed as a problem here, per Q1 results) |
| **Rule-based safety validator (pure Python)** | `llm/serving/safety_validator.py` | Regex + heuristic detectors, sentence-scoped, negation-aware | Deterministic guarantee > probabilistic judgment for a medical-safety gate (§8.5) | Zero extra inference cost, fully auditable, no false sense of security from "the LLM says it's fine" | Can miss genuinely novel unsafe phrasings outside its patterns — the ten scoring-detector fixes made during Q1 are direct evidence of this class of risk, all found by adversarial testing against real outputs |
| **PostgreSQL 18 + SQLAlchemy + Alembic** | Research persistence (`backend/database/`) | JSONB columns for flexible nested data (raw inputs, full prediction, full guidance) + typed columns for fast common queries (risk_class, confidence) | Durable, queryable, production-grade relational database with native JSON support; Alembic gives explicit, versioned schema control | Real SQL querying over research data, JSONB keeps the schema flexible without constant migrations for every new field | An extra service to install/run/manage; the app degrades gracefully without it, so its absence is silent unless `/db/health` is checked |
| **React 18 + Vite + Tailwind CSS** | Frontend | Component-based patient/admin views, hook-based local state, utility-first styling | Fast dev/build cycle (Vite), no heavyweight state-management library needed for this app's scope, rapid, consistent styling via Tailwind | Fast builds, small bundle (~200KB gzipped ~62KB), easy to reason about with plain hooks | No SSR/routing framework — acceptable for a single-page thesis demo app, would need revisiting for a larger multi-page product |

---

## 10. Unique Selling Points & Novelty (USP)

### 10.1 Routine, Low-Cost Blood-Test Screening Without eGFR

Existing ML CKD detection literature (e.g., Ramspek et al. 2021, Aljaaf et al. 2019) almost universally uses eGFR or serum creatinine as an input feature. Nephro-AI's model uses only vitals, routine low-cost blood tests, and comorbidity flags that are already part of a standard checkup — the key enabling property for cross-sectional, community-level pre-screening that doesn't require ordering the specialized confirmatory kidney test up front.

### 10.2 Two-Stage Cascade Tuned for the Clinically Worst Error

Rather than optimizing aggregate accuracy, the cascade's two independent thresholds are tuned specifically to minimize high→low misses (2 on the held-out test set) — the single most dangerous failure mode for a screening tool.

### 10.3 Factor-Aware Personalized RAG Retrieval

Most RAG-enabled health information systems retrieve top-k by vector similarity alone, which tends to return the same generic documents for every user in a narrow domain. Nephro-AI's retriever derives a prioritized topic list from each patient's own top ML factors and guarantees per-topic coverage — different patients at the same risk class receive different guideline grounding.

### 10.4 A Rigorously Measured, Six-Arm Causal Ablation of Every Safety Component

Rather than reporting one end-to-end evaluation number, the Q1 evaluation isolates and statistically tests the individual contribution of RAG, the safety validator, LoRA fine-tuning, and prompt compression, on the same 300-profile benchmark, with paired statistics (McNemar, Wilson CI, Holm correction) and full transparency about every scoring correction found and fixed along the way. Few deployed health-LLM systems publish this level of component-isolated, statistically-defended safety evidence.

### 10.5 Fine-Tuning as a Verified, Not Assumed, Improvement

The fine-tuned adapter's value is not claimed by assertion — it is measured against the base model on the identical benchmark (D vs B, E vs D, F vs E), with the honest finding that its safety benefit alone is not independently significant, while its quality/efficiency benefit (grounding, token cost) is measured and real. This kind of self-critical, component-level reporting is uncommon in applied LLM health-tooling literature.

### 10.6 Content-Aware, Not Blanket-Filtered, Knowledge Base Safety

Knowledge-base chunks with clinical-only numeric content are tagged `restricted_clinician_referral` rather than excluded outright — retrievable, but flagged so the model defers to a professional rather than citing them as direct advice, preserving usefulness without sacrificing safety.

### 10.7 Full-Stack, Dual-Audience Deployment

A complete web application with a calm patient-facing interface and a fully transparent researcher-facing admin interface (every probability, factor, retrieved chunk, validator decision, and evaluation number auditable) — bridging the common gap between an academic ML experiment and a demonstrable, inspectable system.

---

## 11. Limitations & Scope

1. **Screening, not diagnosis.** Confirmatory testing (eGFR, UACR) is always required; the system exists to trigger professional follow-up, not replace it.
2. **Population bias.** Trained on NHANES 2021–23 (US adult population); applicability to other populations is unverified and should be validated before any real deployment.
3. **Moderate-class performance.** Moderate recall (0.505) is the weakest metric, reflecting the genuine ambiguity of that category; a false moderate↔low is less dangerous than a false high→low, which the cascade is specifically tuned against.
4. **LLM explanation is not SHAP.** The "Main signals" explanation is a real, honestly-labeled model-agnostic local-perturbation method, not Shapley-value attribution, and does not capture feature interactions.
5. **Reliability score is a heuristic, not calibrated uncertainty.** It is deterministic and fully inspectable, but should not be described as a statistically validated confidence interval.
6. **Emergency-symptom detection is currently vitals-only in production.** The validator's symptom vocabulary is real and tested by the Q1 harness, but the live input schema has no free-text symptom field yet (§4.8) — this is a known, documented gap, not a hidden one.
7. **Q1 benchmark is synthetic.** All 300 profiles are model-scored rather than drawn from real patients; the evaluation measures internal safety-mechanism effectiveness and consistency, not real-world outcome validity.
8. **Rule-based safety detection has an inherent blind spot** to genuinely novel unsafe phrasings outside its tested patterns — mitigated, not eliminated, by the extensive adversarial-testing-and-fix cycle documented in §7 and `results/q1_llm_evaluation_summary.md`.
9. **No clinician review** of generated guidance has been performed in this phase.
10. **English only.** No multilingual support in the knowledge base or prompts.
11. **Cross-sectional design, no longitudinal tracking.** Both the training data (NHANES) and each screening itself are cross-sectional — a single point-in-time snapshot per person, not repeated measurements over time. The model has no notion of whether a patient's risk is rising, falling, or stable across visits, and the app does not implement trend/serial-measurement tracking.
12. **Database persistence is best-effort.** A database outage silently skips saving rather than queuing or retrying — acceptable for research/demo use, not for a production clinical record system as-is.

---

## 12. Citations & References

### Clinical Guidelines

[1] Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. *KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease.* Kidney International. 2024;105(Suppl 4S):S117–S314.

[2] Kidney Disease: Improving Global Outcomes (KDIGO) Blood Pressure Work Group. *KDIGO 2021 Clinical Practice Guideline for the Management of Blood Pressure in Chronic Kidney Disease.* Kidney International. 2021;99(3S):S1–S87.

[3] Kidney Disease: Improving Global Outcomes (KDIGO) Diabetes Work Group. *KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease.* Kidney International. 2022;102:S1–S127.

[4] American Diabetes Association. *Standards of Care in Diabetes—2026, Chapter 11: Chronic Kidney Disease and Risk Management.* Diabetes Care. 2026;49(S1).

[5] Ikizler TA, Burrowes JD, Byham-Gray LD, et al. *KDOQI Clinical Practice Guideline for Nutrition in CKD: 2020 Update.* American Journal of Kidney Diseases. 2020;76(3):S1–S107.

[6] Centers for Disease Control and Prevention (CDC). *Chronic Kidney Disease Basics.* 2024.

[7] Centers for Disease Control and Prevention (CDC). *Preventing Chronic Kidney Disease.* 2024.

[8] Centers for Disease Control and Prevention (CDC). *Risk Factors for Chronic Kidney Disease.* 2024.

[9] National Kidney Foundation (NKF). *Exercise and Chronic Kidney Disease* (patient education). 2026.

[10] Samarpan Kidney Foundation. *Save Your Kidneys — Complete Guide for Kidney Patients*, 2nd Edition. 2015.

### Training Data

[11] Centers for Disease Control and Prevention / National Center for Health Statistics (CDC/NCHS). *National Health and Nutrition Examination Survey (NHANES) 2021–2023 Datasets.* https://www.cdc.gov/nchs/nhanes/

### ML & CKD Detection Literature

[12] Ramspek CL, Steyerberg EW, Riley RD, et al. Prediction of kidney failure in patients with chronic kidney disease: a systematic review of prediction models. *Nephrology Dialysis Transplantation.* 2021;36:1066–1076.

[13] Aljaaf AJ, Al-Jumeily D, Hussain AJ, et al. Early prediction of chronic kidney disease using machine learning supported by predictive analytics. *2019 IEEE Congress on Evolutionary Computation (CEC)*. 2019.

[14] Chittora P, Chaurasia S, Chakrabarti P, et al. Prediction of chronic kidney disease — a machine learning perspective. *IEEE Access.* 2021;9:17312–17334.

[15] Yildirim P. Chronic kidney disease prediction on imbalanced data by multilayer perceptron. *2017 IEEE 41st Annual Computer Software and Applications Conference (COMPSAC)*. 2017.

### Imbalanced Learning

[16] Lemaître G, Nogueira F, Aridas CK. Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *Journal of Machine Learning Research.* 2017;18(17):1–5.

### Language Models & Fine-Tuning

[17] Qwen Team (Alibaba). *Qwen3 Technical Report.* 2025. Model: Qwen3-4B-Instruct-2507.

[18] Hu EJ, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR).* 2022.

[19] Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems.* 2023;36.

### Embeddings

[20] Chen J, Xiao S, Zhang P, et al. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv preprint arXiv:2309.07597. 2023.

### RAG

[21] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems.* 2020;33:9459–9474.

### LLM Safety in Healthcare

[22] Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. *Nature.* 2023;620:172–180.

[23] Thirunavukarasu AJ, Ting DSJ, Elangovan K, et al. Large language models in medicine. *Nature Medicine.* 2023;29:1930–1940.

[24] Li I, Pan J, Goldwasser J, et al. Neural natural language processing for unstructured data in electronic health records: a review. *Computer Science Review.* 2022;46:100511.

### Statistical Methods (Q1 Evaluation)

[25] McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika.* 1947;12(2):153–157.

[26] Wilson EB. Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association.* 1927;22(158):209–212.

[27] Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics.* 1979;6(2):65–70.

---

## 13. Frontend Component Reference

### 13.1 Component Inventory

| Component | File | Responsibility |
|---|---|---|
| `App` | `src/App.jsx` | Root: shared state (values, guidance, view), health polling, demo runner |
| `AppShell` | `src/components/AppShell.jsx` | Sidebar navigation, status bar, disclaimer footer |
| `PatientView` | `src/components/patient/PatientView.jsx` | Form + result display container |
| `PatientForm` | `src/components/patient/PatientForm.jsx` | 14-field input form, live completeness indicator |
| `PatientResult` | `src/components/patient/PatientResult.jsx` | Risk card, reliability, main signals, five-section education with citations, safety note |
| `AdminDashboard` | `src/components/admin/AdminDashboard.jsx` | Tabbed dashboard |
| `PredictionPanels` | `src/components/admin/PredictionPanels.jsx` | Probabilities, stage outputs, input quality, full explanation table |
| `LLMPanels` | `src/components/admin/LLMPanels.jsx` | RAG query/chunks/grounding, raw vs. final text, safety-validator audit |
| `EvaluationPanel` | `src/components/admin/EvaluationPanel.jsx` | Frozen ML metrics + final Q1 LLM ablation metrics (deployed-configuration, Arm F) |
| `ExportButtons` | `src/components/admin/ExportButtons.jsx` | JSON/text download and copy of the actual in-memory response |
| `DemoCases` | `src/components/DemoCases.jsx` | One-click demo inputs, run through the real live pipeline |
| `SystemHealth` | `src/components/SystemHealth.jsx` | Live backend/model/adapter/database status |
| `ui` | `src/components/ui.jsx` | Generic presentational primitives (Card, Badge, Tabs, JSON viewer, etc.) — no embedded data |

### 13.2 Key Data Files (Frontend)

| File | Contents |
|---|---|
| `src/data/featureSchema.js` | 14 feature definitions: label, unit, type, default, range, help text |
| `src/data/demoCases.js` | 3 realistic example payloads run through the live pipeline |
| `src/data/evaluationData.js` | Frozen ML test metrics + final Q1 evaluation metrics (deployed configuration) |
| `src/lib/risk.js` | Risk-level and reliability presentation helpers (color/label mapping, completeness %) |
| `src/lib/exportUtils.js` | JSON/text serialization for export |
| `src/api/client.js` | API wrapper — `predict`, `generateGuidance`, `health`, `llmHealth` |

### 13.3 Environment

```env
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

For the full, line-by-line mathematical justification of every displayed frontend element, see **`docs/FRONTEND_ELEMENTS_JUSTIFICATION.md`**.

---

## 14. Database Schema

The database layer is **configured and live** (PostgreSQL 18, verified end-to-end with a real request during project setup) but remains **best-effort at runtime** — the API never blocks or fails a screening if the database is unreachable.

### 14.1 Table: `screenings`

| Column | SQL Type | Purpose |
|---|---|---|
| `id` | INTEGER PRIMARY KEY (autoincrement) | Row identifier |
| `created_at` | TIMESTAMPTZ (`server_default = now()`) | When the screening was run |
| `source` | VARCHAR(32), indexed | `"predict"` or `"generate-guidance"` |
| `input_features` | JSONB | Raw 14-feature payload from the user |
| `n_missing_features` | INTEGER | Count of features that were imputed |
| `risk_class` | VARCHAR(16), indexed | `"low"` / `"moderate"` / `"high"` |
| `risk_class_display` | VARCHAR(64) | Human-readable label |
| `confidence` | FLOAT | Max class probability |
| `probabilities` | JSONB | `{low, moderate, high}` probability map |
| `prediction` | JSONB | Full `/predict` response |
| `guidance` | JSONB | Full `/generate-guidance` response (populated only for that endpoint) |
| `safety_status` | VARCHAR(16) | `"pass"` / `"sanitized"` / `"fallback"` |
| `llm_mode` | VARCHAR(32) | `"production"` / `"local_demo"` |
| `model_spec` | VARCHAR(128) | LLM model identifier (adapter status baked in) |

Indexed on `created_at`, `source`, and `risk_class` for fast research queries.

### 14.2 Setup (as actually configured)

```bash
# .env
DATABASE_URL=postgresql+psycopg2://postgres:<password>@localhost:5432/ckd_research
CKD_DB_AUTO_CREATE=false     # schema is managed explicitly via Alembic

# One-time setup
createdb ckd_research                 # or: psql -U postgres -c "CREATE DATABASE ckd_research;"
python -m alembic upgrade head        # applies backend/database/migrations/versions/0001_initial_screenings.py
```

Verified end-to-end: a live `/predict` call through the running FastAPI app produced a real row in `screenings`, matching the ORM schema exactly (JSONB columns confirmed on PostgreSQL, not the JSON fallback used only for SQLite).

---

## 15. Project Directory Structure

```
Nephro-AI/
├── backend/
│   ├── main.py                    # FastAPI app, CORS, routes, lifespan (init_db)
│   ├── schemas.py                 # Pydantic: PatientFeatures, PredictionResponse
│   ├── feature_config.py          # MODEL_FEATURES (14), FEATURE_METADATA, NUMERIC_DEFAULTS/RANGES
│   ├── model_service.py           # KidneyRiskModelService: load, score, explain, reliability
│   ├── llm_endpoint.py            # build_guidance_response(), health(), adapter_status()
│   └── database/
│       ├── models.py              # SQLAlchemy Screening ORM
│       ├── db.py                  # Engine, session factory, graceful degradation
│       ├── repository.py          # save_screening(), count_screenings()
│       └── migrations/            # Alembic version files
│
├── llm/
│   ├── serving/
│   │   ├── contract.py            # LLMInput, GuidanceOutput, parse_guidance()
│   │   ├── prompts.py             # Full + compact system prompts, safety rules, personalization cues
│   │   ├── llm_runtime.py         # GuidanceRuntime.generate_guidance() → GuidanceResult
│   │   ├── model_backends.py      # QwenChatModel (+ LoRA), OllamaChatModel, MockGuidanceModel
│   │   ├── safety_validator.py    # Multi-layer enforce(), scan_unsafe(), fallback_text()
│   │   └── citations.py           # Deterministic IEEE citation layer
│   ├── rag/
│   │   ├── retriever.py           # Factor-aware retrieval
│   │   ├── vector_store.py        # FAISS / brute-force search
│   │   ├── embed_index.py         # CLI: build/rebuild the embedding index
│   │   ├── embedder.py            # BGEM3Embedder, OllamaEmbedder, HashingTfidfEmbedder
│   │   ├── ingest.py              # PDF/markdown → chunks.jsonl
│   │   └── audit.py               # Knowledge base safety/coverage audit
│   ├── knowledge_base/
│   │   ├── raw/                   # Original source files
│   │   ├── processed/chunks.jsonl # 162 annotated chunks
│   │   └── index/                 # meta.json, vectors.json, bge_faiss.index (ollama-bge-m3)
│   ├── data/                      # Held-out evaluation profiles
│   ├── finetune/
│   │   ├── data.py                # Loss-masked SFT dataset construction
│   │   ├── qlora_train.py         # Phase-8 LoRA training script (bf16 used; optional 4-bit mode)
│   │   ├── evaluate_adapter.py
│   │   └── adapters/v1/adapter/   # Trained LoRA adapter (adapter_config.json, adapter_model.safetensors)
│   └── eval/
│       ├── score.py                     # Legacy Phase-6 scoring
│       └── q1/                          # Final six-arm ablation package
│           ├── benchmark.py             # 300-profile frozen benchmark generator
│           ├── detectors.py             # Rule-based safety-violation detectors
│           ├── metrics.py, stats.py     # Quality/grounding/efficiency + statistics
│           ├── run_arms.py              # Six-arm generation runner
│           └── report.py                # Scoring + tables + figures + summary
│
├── frontend/
│   └── src/ (see §13)
│
├── model_package/
│   ├── FINAL_MODEL_C_FULL_DEVELOPMENT.joblib   # deployed model
│   └── *.ipynb                                  # training/development notebooks
│
├── phase7_data/                    # RAFT/self-distillation training data (train.jsonl, val.jsonl, data_card.md)
├── q1_benchmark/                   # Frozen 300-profile Q1 benchmark
├── q1_runs/                        # Six-arm generation records
├── q1_report/                      # Generated tables, figures, machine-generated summary
├── results/
│   └── q1_llm_evaluation_summary.md   # Final, hand-authored Q1 evaluation writeup
│
├── docs/
│   ├── TECHNICAL_DOCUMENTATION.md          # This file
│   ├── HARDWARE_AND_ENVIRONMENT.md         # Verified GPU / VRAM / library versions / precision
│   └── FRONTEND_ELEMENTS_JUSTIFICATION.md  # Line-by-line math justification for every UI element
│
├── misc/docs/                      # Phase-by-phase LLM development documentation (Phases 1-9)
├── sample_data/                    # Sample /predict request payload
├── requirements.txt                # Backend (thesis MVP / local-demo)
├── requirements-llm.txt            # Production LLM/fine-tuning/RAG deps
├── alembic.ini
├── .env / .env.example
└── README.md
```

---

## 16. Deployment Modes Comparison

| Aspect | Local Demo (`local_demo`) | Ollama (`ollama`) | Production (`qwen`, deployed) |
|---|---|---|---|
| **LLM** | `MockGuidanceModel` (deterministic Python) | Qwen3-4B via Ollama REST API | Qwen3-4B via HuggingFace `transformers`, optionally + Phase-8 LoRA adapter |
| **Embedder** | HashingTFIDF (pure Python, 4096-d) | BGE-M3 via Ollama REST API (1024-d) | BGE-M3 via Ollama REST API (1024-d) |
| **Vector search** | Brute-force | Brute-force / FAISS | FAISS |
| **GPU required** | No | No | Recommended (CPU works, much slower) |
| **Prompt variant** | Full | Full | Compact (when adapter active) / Full (base only) |
| **Output quality** | Condition-aware, not natural language | Full LLM quality | Full LLM quality, fine-tuned + evaluated (Q1) |
| **Safety validator** | Same (runs in all modes) | Same | Same |
| **Env var** | `CKD_LLM_MODEL=local_demo` | `CKD_LLM_MODEL=ollama` | `CKD_LLM_MODEL=qwen` + `CKD_LLM_ADAPTER=llm/finetune/adapters/v1/adapter` |
| **Best for** | Any laptop, CI, offline demo | Local GPU laptops without the full ML stack | The deployed, evaluated configuration (Arm F of the Q1 ablation) |

---

## 17. Quick-Reference Numbers

| Parameter | Value |
|---|---|
| ML input features | 14 |
| Risk classes | 3 (low, moderate, high) |
| Stage-1 threshold | 0.44 |
| Stage-2 threshold | 0.50 |
| High→low misses (held-out test set) | 2 |
| ML accuracy / balanced accuracy | 0.7095 / 0.6602 |
| Knowledge base sources (active) | 10 |
| Knowledge base chunks | 162 |
| RAG retrieval k | 4 (default) |
| Output sections | 5 |
| Safety rules in the full system prompt | 9 |
| Validator layers | 6 content layers + 1 structural check |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Fine-tuning epochs / examples | 3 / 540 train + 60 val |
| Fine-tuning final train loss / best val loss | 0.133 / 0.155 |
| Prompt-token reduction (compact vs. full, fine-tuned) | 25% |
| Q1 evaluation benchmark size | 300 profiles (6 arms × same 300) |
| **Deployed configuration (Arm F): delivered critical-unsafe** | **0.0%** |
| **Deployed configuration (Arm F): emergency-redirect recall** | **100%** |
| Deployed configuration (Arm F): grounding (cited ≥1 source) | 99.7% |
| Deployed configuration (Arm F): personalization score | 1.99 / 2 |
| LLM generation temperature / top_p / max_new_tokens (live serving) | 0.3 / 0.9 / 700 |
| LLM decoding in the Q1 evaluation | Greedy (deterministic), seed 42 |
| **GPU model** | **NVIDIA GeForce RTX 3090** |
| **VRAM** | **24 GB** (7.62 GiB used by the served model + adapter) |
| **PyTorch version** | **2.5.1+cu121** (CUDA 12.1) |
| **Inference precision** | **bfloat16**, no quantization (LoRA adapter weights float32) |
| **transformers version** | **5.14.1** |
| PEFT version | 0.19.1 |
| Fine-tuning precision | bf16 LoRA, no quantization |
| BGE-M3 embedding dimension | 1024 |
| Database | PostgreSQL 18, configured and live |

---

*This documentation reflects the final thesis state as of September 2026, incorporating the completed Phase 8 LoRA fine-tuning (bf16), the final Q1 six-arm safety/quality ablation, the live PostgreSQL research-persistence setup, and the full-project consistency audit performed prior to defense.*
