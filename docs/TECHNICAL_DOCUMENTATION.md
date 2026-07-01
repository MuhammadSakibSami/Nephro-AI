# Nephro-AI: Technical Documentation
### Non-Invasive CKD Risk Screening with ML and Safety-Guarded, Guideline-Grounded LLM Education

**Version:** Thesis MVP 1.0  
**Authors:** Mohammad Walid Hasan  
**Institution:** Jahangirnagar University (JU) / IIT  
**Date:** June 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack](#3-tech-stack)
4. [How It Works — End-to-End](#4-how-it-works--end-to-end)
   - 4.1 [Patient Input & Feature Engineering](#41-patient-input--feature-engineering)
   - 4.2 [Two-Stage ML Cascade](#42-two-stage-ml-cascade)
   - 4.3 [LLM Context Building](#43-llm-context-building)
   - 4.4 [Guideline RAG Retrieval](#44-guideline-rag-retrieval)
   - 4.5 [Prompt Construction](#45-prompt-construction)
   - 4.6 [LLM Generation](#46-llm-generation)
   - 4.7 [Safety Validation](#47-safety-validation)
   - 4.8 [Frontend Rendering](#48-frontend-rendering)
5. [Knowledge Base](#5-knowledge-base)
6. [API Reference](#6-api-reference)
7. [Evaluation Results](#7-evaluation-results)
8. [Design Decisions & Tradeoffs](#8-design-decisions--tradeoffs)
9. [Unique Selling Points & Novelty (USP)](#9-unique-selling-points--novelty-usp)
10. [Limitations & Scope](#10-limitations--scope)
11. [Citations & References](#11-citations--references)

---

## 1. Project Overview

### 1.1 What is Nephro-AI?

Nephro-AI is a **non-invasive chronic kidney disease (CKD) risk screening and patient education system**. It combines a two-stage machine-learning cascade with a safety-guarded, retrieval-augmented large language model (LLM) to:

1. **Screen** for CKD risk (low / moderate / high) from 22 routine health inputs — *without requiring creatinine, eGFR, or UACR*, which are the standard confirmatory tests.
2. **Educate** the patient with personalized, plain-language kidney-health guidance that is grounded in clinical guidelines (KDIGO, ADA, NKF/KDOQI, CDC) and guaranteed by a multi-layer safety validator to contain no diagnosis, no medication advice, and no individualized nutrient targets.

### 1.2 Problem Statement

Chronic kidney disease affects approximately 800 million people globally. The majority of cases go undetected until advanced stages because:

- Standard confirmatory tests (eGFR, serum creatinine, UACR) require laboratory infrastructure and phlebotomy.
- Asymptomatic early-stage CKD produces no obvious symptoms.
- Patient health literacy about kidney risk is generally very low.

Existing ML-based CKD detection models almost universally rely on eGFR or serum creatinine as input features — making them useful for clinical confirmation but not for *community-level early screening*.

Additionally, most ML screening tools provide only a risk score. They do not explain what the risk means, what the patient should do about it, or when to seek care — all in language the patient can understand.

### 1.3 Solution

Nephro-AI addresses both gaps:

- **Non-invasive screening:** The ML model is trained entirely on non-creatinine, non-eGFR features drawn from NHANES 2021–23, making it deployable without laboratory access.
- **Explainable, personalized education:** The LLM pipeline translates the ML result + patient profile into five-section patient education, grounded in clinical guidelines, personalized to the patient's specific conditions (diabetes, hypertension, BMI, etc.), and safety-validated before delivery.

> **Scope disclaimer:** This is a screening and education tool. It does not diagnose CKD, prescribe treatment, or replace clinician judgment, eGFR/UACR testing, or medical evaluation.

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PATIENT INPUTS                           │
│   Age, Sex, BMI, Waist, Blood Pressure, Pulse, Glucose,        │
│   HbA1c, BUN, Uric Acid, Albumin, Na, K, Hemoglobin,          │
│   Diabetes, Hypertension, Cholesterol flags & medications       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼ POST /generate-guidance
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (backend/)                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             TWO-STAGE ML CASCADE                         │   │
│  │                                                          │   │
│  │  Stage 1: Random Forest (At-risk gate)                   │   │
│  │     → Outputs P(at-risk)                                 │   │
│  │     → Threshold: 0.44                                    │   │
│  │     → Low probability → classified LOW, exit             │   │
│  │                                                          │   │
│  │  Stage 2: Gradient Boosting (High-risk discriminator)    │   │
│  │     → Only runs when Stage 1 sends patient to Stage 2    │   │
│  │     → Outputs P(high | at-risk)                          │   │
│  │     → Threshold: 0.50                                    │   │
│  │     → Combined: low / moderate / high                    │   │
│  │                                                          │   │
│  │  Local Explanation (model-agnostic perturbation)         │   │
│  │     → Top 6 factors with direction + magnitude           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                      │                                           │
│                      ▼                                           │
│             risk_class · probabilities · key_factors ·          │
│             reliability · patient_inputs → llm_context          │
│                      │                                           │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM COMPONENT (llm/)                          │
│                                                                  │
│  ┌────────────────────┐    ┌────────────────────────────────┐   │
│  │   RAG RETRIEVER    │    │     PROMPT BUILDER             │   │
│  │                    │    │                                │   │
│  │ 1. Build query     │    │  System prompt                 │   │
│  │    from patient    │    │   - Role + scope               │   │
│  │    risk factors    │    │   - Safety rules (9 rules)     │   │
│  │                    │    │   - Output format spec         │   │
│  │ 2. Embed query     │    │                                │   │
│  │    (BGE-M3 via     │    │  User prompt                   │   │
│  │    Ollama or       │    │   - Screening result           │   │
│  │    TF-IDF)         │    │   - PATIENT PROFILE            │   │
│  │                    │    │   - Top ML factors             │   │
│  │ 3. Factor-aware    │    │   - Guideline chunks (k=4)     │   │
│  │    retrieval:      │    │   - Personalization cues       │   │
│  │    1 best chunk    │    │     (per-section hints)        │   │
│  │    per priority    │    │                                │   │
│  │    topic, then     │    └───────────────┬────────────────┘   │
│  │    fill to k=4     │                    │                     │
│  └────────────────────┘                    ▼                     │
│                                  ┌──────────────────────┐        │
│                                  │   LLM (generate)     │        │
│                                  │  Qwen3-4B (Ollama)   │        │
│                                  │  or MockGuidance     │        │
│                                  └──────────┬───────────┘        │
│                                             │                     │
│                                             ▼                     │
│                               ┌──────────────────────────┐       │
│                               │   SAFETY VALIDATOR       │       │
│                               │                          │       │
│                               │ Layer 1: Diagnosis block │       │
│                               │ Layer 2: Medication block│       │
│                               │ Layer 3: Supplement block│       │
│                               │ Layer 4: Nutrition rewrite│      │
│                               │ Layer 5: Lab reminder    │       │
│                               │ Layer 6: Emergency redir │       │
│                               │ Layer 7: Format check    │       │
│                               │                          │       │
│                               │ status: pass/sanitized/  │       │
│                               │         fallback         │       │
│                               └──────────┬───────────────┘       │
│                                          │                        │
└──────────────────────────────────────────┼────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND (frontend/)                    │
│                                                                  │
│  Patient View: Risk card · Key signals · 5-section education    │
│  Admin View:  Probabilities · Factors · RAG · Validator audit   │
│  Demo Cases:  Low / Moderate / High-emergency one-click cases   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Deployment Modes

| Mode | LLM | Embedder | RAG | Requires |
|---|---|---|---|---|
| **local_demo** | MockGuidanceModel (deterministic) | HashingTFIDF (pure Python) | Custom vector store | Nothing — runs anywhere |
| **ollama** | Qwen3-4B via Ollama REST API | BGE-M3 via Ollama REST API | Custom vector store | Ollama running locally |
| **production** | Qwen3-4B via HuggingFace transformers | BGE-M3 via sentence-transformers | FAISS index | GPU + CUDA |

The active mode is displayed in the UI. The safety validator runs in all modes.

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend API** | FastAPI 0.115 + Uvicorn 0.34 | REST API, routing, CORS, lifespan management |
| **ML Model** | scikit-learn + imbalanced-learn | Two-stage cascade training & inference |
| **Model serialization** | joblib | `.joblib` package containing model, features, thresholds |
| **Data processing** | pandas + numpy | Feature matrix construction, imputation |
| **Dense Embeddings** | BAAI/BGE-M3 via Ollama REST API | Semantic embedding for RAG |
| **Lexical Embeddings** | Pure-Python HashingTFIDF (stdlib) | Offline RAG fallback |
| **Vector Store** | Custom pure-Python + JSON | Serializable offline store; optional FAISS in prod |
| **LLM (production)** | Qwen3-4B-Instruct-2507 via Ollama | Guideline-grounded education generation |
| **LLM (demo)** | MockGuidanceModel (deterministic) | Offline pipeline testing |
| **Safety Validator** | Pure Python (stdlib only) | Post-generation rule enforcement |
| **Frontend** | React 18 + Vite 6 + Tailwind CSS 3 | Patient UI + Admin/Research dashboard |
| **Frontend state** | React hooks (useState, useEffect) | Local state, no Redux |
| **API client** | Axios | Frontend → backend communication |
| **Database** | PostgreSQL (optional, best-effort) | Screening persistence |
| **Config** | python-dotenv + VITE env | Backend and frontend environment config |
| **Platform** | Windows 10 (primary) / cross-platform | Runs on any OS; Windows `.bat` helpers provided |
| **Training data** | NHANES 2021–23 (CDC/NCHS) | Population-level health survey (US) |

### 3.1 Key Python Dependencies

```
fastapi==0.115.x
uvicorn==0.34.x
python-dotenv
joblib
scikit-learn
imbalanced-learn
pandas
numpy
requests           # Ollama REST calls
```

**Optional (production LLM):**
```
sentence-transformers   # BGE-M3 via sentence-transformers
torch                   # Required by sentence-transformers
transformers            # Qwen3 via HuggingFace
faiss-cpu / faiss-gpu   # Production vector search
```

### 3.2 Frontend Dependencies

```
react 18.x
react-dom 18.x
vite 6.x
tailwindcss 3.x
axios
lucide-react         # Icon library
```

---

## 4. How It Works — End-to-End

### 4.1 Patient Input & Feature Engineering

The user submits a form with **22 features**:

**Demographic & socioeconomic:**
| Feature | Type | Unit |
|---|---|---|
| age | Numeric | years |
| sex_female | Binary | 0/1 |
| income_poverty_ratio | Numeric | ratio |

**Anthropometric & vitals:**
| Feature | Type | Unit |
|---|---|---|
| bmi | Numeric | kg/m² |
| waist | Numeric | cm |
| mean_sbp | Numeric | mmHg |
| mean_dbp | Numeric | mmHg |
| mean_pulse | Numeric | bpm |

**Laboratory values:**
| Feature | Type | Unit |
|---|---|---|
| glucose | Numeric | mg/dL |
| hba1c | Numeric | % |
| blood_urea_nitrogen | Numeric | mg/dL |
| uric_acid | Numeric | mg/dL |
| serum_albumin | Numeric | g/dL |
| sodium | Numeric | mmol/L |
| potassium | Numeric | mmol/L |
| hemoglobin | Numeric | g/dL |

**Comorbidity flags (binary, Yes/No):**
| Feature | Description |
|---|---|
| diabetes_yes | Diagnosed diabetes |
| borderline_diabetes | Prediabetes / borderline |
| hypertension_doctor_told | Diagnosed hypertension |
| high_cholesterol_told | Diagnosed high cholesterol |
| taking_bp_medicine | Currently on BP medication |
| taking_cholesterol_medicine | Currently on cholesterol medication |

**Notably absent:** creatinine, eGFR, UACR, urine protein — the defining features of standard CKD diagnostics. Their exclusion is the key non-invasive property of the system.

The backend applies **live-input consistency rules** before sending to the model:
- If hypertension=No, then taking_bp_medicine defaults to 0 (no need to ask).
- If high_cholesterol=No, taking_cholesterol_medicine defaults to 0.
- If diabetes=Yes, borderline_diabetes defaults to 0 (can't have both).

Missing features are handled by the model's trained imputation pipeline (median/mode imputation fitted on the training set). Up to 5 missing features are tolerated; reliability degrades gracefully.

---

### 4.2 Two-Stage ML Cascade

The core ML model is a **two-stage cascade** trained on NHANES 2021–23 data and serialized as a `.joblib` package.

**Stage 1 — At-Risk Gate (Random Forest)**

- **Task:** Binary classification: low vs. at-risk (moderate+high combined).
- **Algorithm:** Random Forest (with imbalanced-learn SMOTE or class-weight balancing).
- **Output:** `P(at-risk)` — probability the patient is NOT low risk.
- **Threshold:** 0.44 (tuned to minimize dangerous high→low misses).
- **Decision:** If `P(at-risk) < 0.44` → classify as **low**, return immediately.

**Stage 2 — High-Risk Discriminator (Gradient Boosting)**

- **Task:** Binary classification within at-risk pool: moderate vs. high.
- **Algorithm:** Gradient Boosting Classifier.
- **Input:** A subset of features selected for high-vs-moderate discrimination.
- **Output:** `P(high | at-risk)` — conditional probability of high given at-risk.
- **Threshold:** 0.50.
- **Decision:** If `P(high | at-risk) ≥ 0.50` → classify as **high**, else **moderate**.

**Why two stages?**

Standard multi-class classifiers optimize overall accuracy, which tends to treat high-risk and low-risk mistakes equivalently. The cascade allows independent optimization:
- Stage 1 is tuned for **high sensitivity** (catch all at-risk patients).
- Stage 2 is tuned for **high specificity** (correctly separate high from moderate among those flagged).
- Result: Only **2 high→low misses** in the held-out test set, vs. more with a single model.

**Local Explanation (Model-Agnostic Perturbation)**

For each supplied feature, the backend temporarily replaces the patient's value with a reference value (population median) and measures the change in predicted-class probability. Features are ranked by their absolute contribution. The top 6 are surfaced as `key_model_factors` — the "key signals" shown to the patient and used to personalize the LLM guidance.

This is **not SHAP** — it is a fast, model-agnostic local perturbation, computationally cheap and interpretable.

**Reliability Scoring**

A reliability score (low / moderate / high) is computed from:
- Model confidence (max class probability)
- Probability margin between top two classes
- Number of missing features
- Proximity to decision boundaries

This is displayed in the UI and used in the prompt to flag uncertain predictions.

---

### 4.3 LLM Context Building

After scoring, the backend calls `_build_llm_context()` to package everything the LLM needs:

```json
{
  "purpose": "Generate non-clinical supportive lifestyle guidance...",
  "strict_safety_rules": [...],
  "model_output": {
    "screening_category": "high",
    "class_probabilities": {"low": 0.05, "moderate": 0.15, "high": 0.80},
    "confidence": 0.80,
    "reliability_level": "high",
    "reliability_notes": [...]
  },
  "key_model_factors": [
    {"feature": "mean_sbp", "label": "Mean systolic BP", "current_value": 160,
     "unit": "mmHg", "direction": "supports_prediction", "absolute_contribution": 0.18},
    ...
  ],
  "input_quality": {"n_supplied": 22, "n_missing": 0, ...},
  "patient_inputs": {"age": 65, "mean_sbp": 160, "diabetes_yes": 1, ...},
  "suggestion_style": {...}
}
```

The `patient_inputs` dict carries the full raw patient values — this is what enables per-patient LLM personalization.

---

### 4.4 Guideline RAG Retrieval

**Knowledge base construction:**

Clinical guideline PDFs and text files are chunked into ~150-word passages. Each chunk receives:
- `chunk_id`: unique identifier (e.g., `KDIGO2024_bp_001`)
- `topic`: semantic topic tag (bp / diabetes / nutrition / weight / activity / smoking / followup / general)
- `safety_label`: `safe_public` or `restricted_clinician_referral`
- `restricted`: boolean — True for chunks containing numeric nutrient targets or clinical-only content

Total: **118 chunks** across 10 source documents.

**Query construction:**

The retriever builds a query string from:
1. The risk class (`"high kidney disease risk"`)
2. Top factor phrases from ML explanation (`"Mean systolic BP 160 mmHg"`, `"HbA1c 8.5%"`)
3. Mapped query terms per feature (`mean_sbp` → `"high blood pressure"`, `hba1c` → `"high blood sugar diabetes"`)
4. A brief base intent anchor (`"kidney health lifestyle education"`)

Example query: `"high kidney disease risk. Mean systolic BP 160 mmHg. HbA1c 8.5 %. high blood pressure. high blood sugar diabetes. kidney health lifestyle education"`

**Factor-aware retrieval:**

Instead of returning the top-k chunks by raw similarity score (which collapses to the same generic chunks for everyone), the retriever uses a **priority-topic strategy**:

1. Map each top factor to a topic (e.g., `mean_sbp` → `"bp"`, `hba1c` → `"diabetes"`).
2. Fetch a large candidate pool (up to 24 chunks).
3. Take the **single best-scoring chunk per priority topic**, in factor-importance order.
4. Fill remaining slots with best-scoring leftovers.
5. For at-risk patients, guarantee a `"followup"` topic chunk is included.

This ensures a patient with hypertension as their top factor gets a bp-topic chunk, while a patient with diabetes as their top factor gets a diabetes-topic chunk — even if the two patients have the same risk class.

---

### 4.5 Prompt Construction

The prompt is built in two parts by `llm/serving/prompts.py`.

**System prompt** (stable, shared across all patients):
- Role and scope statement
- 9 safety rules (no diagnosis, no medication, no supplements, no numeric nutrient targets, personalization-allowed, lab-reminder requirement, emergency redirect, guideline-grounding, readability target)
- Output format specification with per-section hints

**User prompt** (fully patient-specific):

```
SCREENING RESULT
- Risk level: High kidney-risk screening category (high)
- Class probabilities: low=0.05, moderate=0.15, high=0.80
- Model confidence: 0.80
- Reliability: high
- Input completeness: All expected inputs were provided.
- Note: moderate/high result — lab/clinician follow-up is REQUIRED.

PATIENT PROFILE (actual health values — use to personalize every section)
  - Age: 65.0 years
  - Sex: Male
  - Systolic blood pressure: 160.0 mmHg
  - HbA1c (blood sugar control): 8.5 %
  - Has diabetes (diagnosed): Yes
  - Has high blood pressure (diagnosed): Yes
  - BMI: 32.0 kg/m²
  ...

TOP FACTORS THAT MOST INFLUENCED THIS RESULT
  - Mean systolic BP 160 mmHg (raises this risk level)
  - HbA1c 8.5% (raises this risk level)
  - Has diabetes (raises this risk level)

GUIDELINE EXCERPTS [k=4 retrieved chunks]
[KDIGO2024_bp_001] (source: KDIGO2024)
...high blood pressure management in CKD...

[KDIGO2024_diabetes_003] ...

Now write the kidney-risk education for this person using the required format.
PERSONALIZATION REQUIREMENTS — tailor EACH section to this specific patient:
- Summary: Name the high risk level. Briefly mention: Mean systolic BP 160 mmHg; HbA1c 8.5%.
- Food habits: focus on low-sugar choices because of elevated blood sugar; suggest reducing salt
  because of high blood pressure
- Rest and sleep: concise, general sleep-hygiene advice (7-9 hours).
- Daily routine: remind them to monitor blood pressure regularly; encourage daily blood-sugar checks
- Follow-up: urine/blood test for kidney function; HbA1c check; blood pressure review with doctor

Important: Do NOT write generic advice. Reference the patient's actual conditions. Follow every safety rule.
```

---

### 4.6 LLM Generation

**Production mode:** Qwen3-4B-Instruct-2507, served via Ollama or HuggingFace Transformers.

Key generation parameters:
- Temperature: **0.65** (for output variety without hallucination)
- max_tokens: **900** (sufficient for all 5 sections)
- top_p: 0.9

**Demo mode:** `MockGuidanceModel` — a deterministic Python class that parses the patient profile flags and numeric values from the rendered prompt and generates condition-specific text. No GPU or network required.

**Retry logic:** If the first generation fails the safety validator's structural check (missing sections), the prompt is re-sent with an appended format reminder. Maximum 1 retry before falling back to the safe fallback template.

---

### 4.7 Safety Validation

The safety validator (`llm/serving/safety_validator.py`) is a multi-layer rule engine that runs on every LLM output before it reaches the user.

**Layer 1 — Diagnosis block (Severe)**  
Pattern-matches phrases like "you have CKD", "you have kidney disease", "you have kidney failure". On detection: triggers regenerate → fallback.

**Layer 2 — Medication block (Severe)**  
Detects named drugs (RAAS agents, diuretics, statins, etc.), dosing language ("mg/day", "twice daily"), and start/stop/change instructions. On detection: regenerate → fallback.

**Layer 3 — Supplement block (Severe)**  
Detects recommendation to "take" supplements (fish oil, vitamin D, probiotics, etc.). On detection: regenerate → fallback.

**Layer 4 — Nutrition restriction rewrite (Moderate, Auto-fix)**  
Detects numeric nutrient targets (e.g., "2 g sodium per day", "potassium 2000 mg"). Rewrites those sentences to a dietitian-referral phrase. Status: `sanitized`.

**Layer 5 — Lab-reminder append (Moderate, Auto-fix)**  
For moderate/high risk results: verifies the follow-up section contains a mention of a clinician visit and kidney test. If absent, appends a standard reminder. Status: `sanitized`.

**Layer 6 — Emergency redirect (Moderate, Auto-fix)**  
If patient inputs show SBP ≥ 180 or DBP ≥ 120, or if the text contains emergency symptom phrases (chest pain, severe swelling, etc.), prepends an urgent-care redirect to the summary. Status: `sanitized`.

**Layer 7 — Structural check (triggers regenerate)**  
Verifies all 5 required sections are present. If any are missing: retry → fallback.

**Outcome states:**
- `pass`: All checks pass, output returned as-is (re-rendered clean).
- `sanitized`: Moderate issues auto-fixed. Output returned with fix annotations.
- `fallback`: Severe violation or structural failure. Safe template (also condition-aware) returned.

**Phase-6 result:** 0% final unsafe output rate on 30 adversarial profiles.

---

### 4.8 Frontend Rendering

The React 18 frontend has four sidebar areas:

**Patient View**
- Calm, non-technical form for data entry
- Risk card (Low / Moderate / High) with color coding
- Reliability badge
- "Key signals" — top 3 ML factors in plain terms
- 5-section education panel (Summary, Food habits, Rest and sleep, Daily routine, Follow-up reminder)
- Provenance line ("Based on N guideline source(s)")
- Mode badge (local demo vs. production)
- Disclaimer

**Admin / Research View**
- Prediction tab: raw probabilities, stage outputs, model confidence
- Input quality tab: supplied vs. missing features
- Model explanation tab: all factor contributions + directions
- LLM & RAG tab: query string, retrieved chunk IDs + scores + topics, grounding ratio
- Safety validator tab: violations detected, auto-fixes applied, validator status
- Evaluation metrics tab: frozen Phase-6 results
- Export: download full JSON response

**Demo Cases**
- One-click Low / Moderate / High-emergency pre-filled cases
- The high-emergency case shows the validator adding an urgent-care redirect (status = `sanitized`)

**System Health**
- Backend status, model path, runtime library versions
- LLM mode and availability
- Database connectivity and stored record count

---

## 5. Knowledge Base

### 5.1 Sources

| Source | Document | Organization | Year | Topics |
|---|---|---|---|---|
| KDIGO 2024 CKD Guideline | `KDIGO-2024-CKD-Guideline.pdf` | KDIGO | 2024 | nutrition, bp, activity, weight, smoking, general |
| KDIGO 2021 Blood Pressure in CKD | `KDIGO-2021-Blood-Pressure-in-CKD-Guideline.pdf` | KDIGO | 2021 | bp |
| KDIGO 2022 Diabetes Management in CKD (ADA) | `dc26s011.pdf` / `dci220027.pdf` | ADA / KDIGO | 2022 | diabetes |
| NKF KDOQI Nutrition | `kadoqi.pdf` | NKF | 2020 | nutrition |
| NKF Exercise Guidelines for CKD | `nkf_exercise_ckd_clean.md` | NKF | 2023 | activity |
| CDC — CKD Basics | `Chronic Kidney Disease Basics CDC.pdf` | CDC | 2024 | general |
| CDC — Preventing CKD | `Preventing Chronic Kidney Disease CDC.pdf` | CDC | 2024 | general, followup |
| CDC — Risk Factors for CKD | `Risk Factors for Chronic Kidney Disease CDC.pdf` | CDC | 2024 | general |
| NKF — Staying Active on Dialysis | `Staying Active on Dialysis...pdf` | NKF | 2023 | activity |

**Total chunks:** 118  
**Chunk length:** ~150 words each  
**Safety labels:** `safe_public` (unrestricted) or `restricted_clinician_referral` (dietitian/physician-only content)

### 5.2 Safety Scoping of the Knowledge Base

Chunks containing **numeric individualized targets** for protein, potassium, phosphorus, or sodium (e.g., "0.8 g/kg/day protein", "2 g sodium per day") are flagged `restricted=True`. The retrieval pipeline can return these chunks to the LLM with an explicit `[RESTRICTED: mention only as 'discuss with clinician/renal dietitian']` tag, preventing the model from citing them as direct patient advice.

This implements the KDOQI constraint: never give individualized nutrient targets to a patient without dietitian involvement.

### 5.3 Topic Tags

The knowledge base chunks are tagged with one of these topics:
`bp` · `diabetes` · `nutrition` · `weight` · `activity` · `smoking` · `sleep` · `followup` · `general`

These tags drive the **factor-aware retrieval** — patients with blood pressure as their top risk driver get `bp`-topic chunks; patients with diabetes as their driver get `diabetes`-topic chunks.

---

## 6. API Reference

All endpoints are served by FastAPI at `http://localhost:8000`. Interactive Swagger docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root — links to docs, health, predict, guidance |
| `GET` | `/health` | Backend + model status, runtime versions, database |
| `GET` | `/db/health` | Database connectivity + stored record count |
| `GET` | `/features` | Feature schema, metadata, numeric/binary lists |
| `GET` | `/model-info` | Model thresholds, selected feature lists |
| `POST` | `/predict` | Two-stage cascade for one patient — returns risk class, probabilities, explanation, llm_context |
| `POST` | `/reload-model` | Hot-reload the model joblib (for model updates) |
| `GET` | `/generate-guidance/health` | LLM mode and backend availability |
| `POST` | `/generate-guidance` | Full pipeline: predict → RAG → LLM → validator — returns guidance sections, provenance, audit trail |

### 6.1 POST /generate-guidance

**Request body:** Same 22-feature JSON as `/predict`.

**Response (abbreviated):**
```json
{
  "risk_class": "high",
  "mode": "local_demo",
  "status": "sanitized",
  "disclaimer": "This assistant provides personalized, guideline-grounded kidney-risk education...",
  "guidance": {
    "Summary": "Your screening shows a high risk level...",
    "Food habits": "Since your blood pressure is elevated, try to reduce salt...",
    "Rest and sleep": "Aim for 7 to 9 hours of sleep each night...",
    "Daily routine": "Check your blood pressure regularly and log the readings...",
    "Follow-up reminder": "Ask for a urine test and blood test for kidney function..."
  },
  "provenance": ["KDIGO", "CDC"],
  "grounding": ["KDIGO2024_bp_001", "KDIGO2024_diabetes_003"],
  "grounding_valid_ratio": 1.0,
  "query": "high kidney disease risk. Mean systolic BP 160 mmHg...",
  "retrieved_chunk_ids": ["KDIGO2024_bp_001", "KDIGO2024_diabetes_003", ...],
  "violations": [{"type": "emergency_redirect", "severity": "moderate", ...}],
  "n_attempts": 1,
  "prediction": { ... full /predict output ... }
}
```

---

## 7. Evaluation Results

### 7.1 ML Two-Stage Cascade

**Dataset:** NHANES 2021–23, held-out test split.

| Metric | Value |
|---|---|
| Accuracy | 0.719 |
| Balanced Accuracy | 0.664 |
| Macro F1 | 0.559 |
| Low recall | 0.751 |
| Moderate recall | 0.505 |
| High recall | 0.737 |
| **High→Low misses** | **2** |

The two most safety-critical errors are high→low misses (patient who needs follow-up is told they are low risk). Only 2 such errors on the held-out test set.

### 7.2 LLM — Phase 6 Evaluation

**Setup:** 30 held-out patient profiles (10 low, 10 moderate, 10 high) run through Qwen3-4B-Instruct-2507 + BGE-M3 RAG (k=4) + safety validator.

| Metric | Value | Target |
|---|---|---|
| Format adherence rate | **1.00** | ≥ 0.95 |
| Raw unsafe rate (pre-validator) | 0.10 | — |
| **Final unsafe rate (post-validator)** | **0.00** | 0.00 |
| Pass rate (no validator action needed) | 0.90 | — |
| Sanitized rate (auto-fixed) | 0.10 | — |
| Fallback rate | 0.00 | — |
| Retry rate | 0.00 | — |
| Mean factor relevance | 0.966 | ≥ 0.80 |
| Grounding rate (any) | 1.00 | ≥ 0.90 |
| Mean grounding valid ratio | 1.00 | — |
| Mean personalization score | 1.93 / 2 | ≥ 1.5 |
| Mean readability (Flesch-Kincaid grade) | **6.9** | ≤ 8 |
| Readability ≤ grade 8 rate | 1.00 | 1.00 |

**Key result:** The safety validator successfully intercepted all 10% raw-unsafe outputs (emergency-redirect cases where SBP was dangerously elevated) and converted them to safe outputs, achieving **0% final unsafe rate** on the full adversarial test set.

---

## 8. Design Decisions & Tradeoffs

### 8.1 Two-Stage Cascade vs. Single Multi-Class Classifier

**Decision:** Use a Random Forest gate followed by a Gradient Boosting discriminator rather than a single three-class classifier.

**Rationale:** Class imbalance in NHANES is severe (low >> moderate > high). A single model optimizes aggregate accuracy and tends to misclassify high-risk patients as moderate or low. The two-stage design lets Stage 1 be tuned purely for recall of at-risk cases, while Stage 2 handles the moderate/high split.

**Tradeoff:** Two models to train, validate, and serialize. Threshold tuning for two decision boundaries. Slightly more complex inference pipeline.

### 8.2 Non-Invasive Feature Set (No eGFR/Creatinine)

**Decision:** Exclude creatinine, eGFR, UACR, and urine protein from the feature set entirely.

**Rationale:** These are the gold-standard markers for CKD but require laboratory testing. Their exclusion makes the screening tool deployable in community health settings, pharmacies, or via mobile health workers with a blood pressure cuff and basic labs.

**Tradeoff:** The model is a screening tool, not a diagnostic tool. Its accuracy (71.9%) is lower than eGFR-based classifiers, but appropriate for population-level triage. The system is designed to trigger follow-up testing, not to replace it.

### 8.3 Retrieval-Augmented Generation (RAG) over Fine-Tuning Alone

**Decision:** Use RAG (embed + retrieve guideline chunks at inference time) as the primary mechanism for grounding the LLM's output, rather than relying solely on fine-tuning.

**Rationale:**
- RAG provides **traceable, updateable, auditable** knowledge grounding. Chunks cited by the model can be verified.
- Guidelines change (KDIGO updates annually). Updating RAG requires re-indexing, not retraining.
- Fine-tuning alone can cause the model to "hallucinate" guidelines with confidence.

**Tradeoff:** RAG adds latency (embedding + search per request). Retrieval errors (wrong chunks) can cause irrelevant citations. These are mitigated by the factor-aware retrieval strategy and the validator's grounding ratio check.

### 8.4 Rule-Based Safety Validator

**Decision:** Use a deterministic rule-based validator (regex + heuristics) rather than an LLM-as-judge approach.

**Rationale:**
- Deterministic: a regex that blocks "you have CKD" will never fail on that pattern.
- Fast: no additional LLM call required.
- Auditable: every decision is logged with the exact rule that triggered it.
- Medical safety requirement: a zero false-negative guarantee on specific patterns (diagnosis, medication) is more valuable than probabilistic LLM judgment.

**Tradeoff:** Rules cannot catch novel harmful patterns outside their definitions. Fine-tuning (planned post-thesis) is the complementary path to internalize safe behavior. An optional LLM-judge layer is architected but not activated in MVP.

### 8.5 Patient Profile in the Prompt (Not Just Key Factors)

**Decision:** Include the patient's full raw input values as a `PATIENT PROFILE` block in the user prompt, in addition to the ML explanation factors.

**Rationale:** Initial testing showed the LLM produced identical generic output for all patients at the same risk level. The ML factors convey which features *drove the risk prediction*, but the patient may also have conditions that are relevant to lifestyle education but didn't top the factor list. Providing the full profile gives the LLM the full picture.

**Tradeoff:** Longer prompt. Slightly higher token cost per request. Mitigated by the `max_tokens=900` cap on the LLM response.

### 8.6 Per-Section Personalization Cues

**Decision:** Compute explicit, condition-specific per-section instructions (`_build_personalization_cues`) and inject them as a `PERSONALIZATION REQUIREMENTS` block in the user prompt.

**Rationale:** The instruction "personalize it" is too vague for a language model. Explicitly stating "Food habits: focus on low-sugar choices because of elevated blood sugar; suggest reducing salt because of high blood pressure" gives the model actionable instructions per section.

**Tradeoff:** Requires condition-detection logic in Python (which duplicates some of what the ML model already knows). This logic uses simple threshold rules, not the model's calibrated probabilities — acceptable for a prompt-hint purpose.

### 8.7 Safety Rule 5 — Personalized Wellness vs. General Wellness

**Decision:** Safety rule #5 was changed from "Only give GENERAL wellness education" to "Give wellness education TAILORED to the patient's specific conditions."

**Rationale:** The original rule directly contradicted the personalization goal. The LLM is a rule-follower; when told "only give general advice" it correctly produces generic text. The revised rule maintains the prohibition on *diagnosis and treatment* while explicitly authorizing *condition-referenced wellness education*.

**Tradeoff:** Slightly wider scope of permitted content. Mitigated by the other 8 safety rules and the post-generation validator, which still block diagnosis, medication, and numeric targets.

### 8.8 Local Demo Mode

**Decision:** Build a `MockGuidanceModel` that produces condition-aware output without any LLM, and call this "local_demo" mode.

**Rationale:** The thesis must be demonstrable on any laptop, including those without a GPU or internet connection. The full safety pipeline still runs in demo mode, so the system's safety properties can be demonstrated without the production LLM.

**Tradeoff:** Demo output quality is deterministic and less natural than Qwen3. The UI clearly labels the active mode and the disclaimer explains the difference.

---

## 9. Unique Selling Points & Novelty (USP)

The following features represent the novel contributions of this thesis compared to existing literature:

### 9.1 Non-Invasive Screening Without eGFR

**What prior work does:** Existing ML CKD detection models (e.g., Ramspek et al. 2021, Aljaaf et al. 2019) use eGFR or serum creatinine as input features. These require blood tests with specific creatinine assays.

**What Nephro-AI does:** The model uses only vitals, standard labs (glucose, HbA1c, BUN, uric acid, albumin, electrolytes, hemoglobin), and comorbidity flags — all measurable without a nephrology-specific test. This is **the key enabling feature for community-level pre-screening**.

### 9.2 KDIGO-Aligned Two-Stage Cascade for Non-Invasive Risk

**What prior work does:** Most ML CKD classifiers use a binary (CKD / no-CKD) or multi-class approach with single-model training. Few are explicitly optimized for the KDIGO risk grid stratification (low / moderate / high / very high) with non-invasive inputs.

**What Nephro-AI does:** The two-stage cascade maps to KDIGO risk categories with explicit threshold tuning to minimize the clinically dangerous high→low miss rate. The cascade is trained on NHANES 2021–23 (the most recent available cycle, post-COVID).

### 9.3 Factor-Aware Personalized RAG Retrieval

**What prior work does:** Most RAG-enabled health information systems retrieve top-k documents by vector similarity alone — which for a narrow domain (kidney health) tends to return the same generic documents for all users.

**What Nephro-AI does:** The retriever uses the patient's **top ML factors** to derive a prioritized topic list (e.g., bp → diabetes → nutrition → followup) and guarantees one best-matching chunk per topic, in factor-importance order. This means **different patients receive different guideline grounding** even when they share the same risk class.

### 9.4 Safety-Scoped Knowledge Base with Content-Level Restriction Labels

**What prior work does:** LLM health chatbots either retrieve from unrestricted medical databases (risking dangerous specific advice) or heavily pre-filter content, reducing usefulness.

**What Nephro-AI does:** Every knowledge base chunk is annotated with a `safety_label`. Chunks with clinical-level content (numeric nutrient targets, dialysis protocols) are tagged `restricted_clinician_referral` rather than excluded entirely. They can still be retrieved but are flagged in the prompt so the model defers to a dietitian/clinician rather than citing them as direct advice. This is a **content-aware retrieval policy** rather than a blanket filter.

### 9.5 Multi-Layer Post-Generation Safety Validator with Measured Guarantee

**What prior work does:** Most deployed health LLM systems rely on either prompt engineering alone (no guarantee) or fine-tuning to internalize safety (expensive, hard to verify). Very few publish measured post-generation unsafe rates.

**What Nephro-AI does:** A 7-layer deterministic validator runs on every LLM output. The system reports a **measured 0% final unsafe output rate** on 30 adversarial profiles (including emergency-BP cases). The validator's actions are logged and surfaced in the Admin view, making safety behavior fully auditable.

### 9.6 Personalization via Condition-Aware Patient Profile Injection

**What prior work does:** LLM health education tools typically pass only a risk class or a generic condition summary. They don't feed the patient's actual numeric values and binary condition flags into the prompt with explicit per-section personalization requirements.

**What Nephro-AI does:** The user prompt contains:
- A `PATIENT PROFILE` block with all 22 patient input values, human-readable.
- A `PERSONALIZATION REQUIREMENTS` block with explicit, condition-derived instructions for each of the 5 output sections — computed from the patient's actual profile.

This ensures a diabetic hypertensive patient receives categorically different food/routine/follow-up guidance from a young patient with only borderline BMI.

### 9.7 Full-Stack End-to-End Patient-Facing + Admin-Facing System

**What prior work does:** Academic CKD ML papers demonstrate model performance metrics. They rarely produce deployable, patient-facing applications with administrative transparency tooling.

**What Nephro-AI does:** A complete web application with a calming patient-facing interface (simplified, no raw probabilities) and a researcher-facing admin interface exposing the full audit trail (probabilities, factors, retrieved chunks, validator decisions, frozen evaluation metrics). This bridges the gap between an academic ML experiment and a deployable screening tool.

---

## 10. Limitations & Scope

1. **Screening, not diagnosis.** The ML model outputs a risk class, not a CKD diagnosis. Confirmatory testing (eGFR, UACR) is always required. The system is designed to be a trigger for professional follow-up, not a replacement for it.

2. **Population bias.** The model is trained on NHANES 2021–23 (US adult population). Applicability to other populations (different ethnic backgrounds, countries, healthcare access patterns) is uncertain and should be validated before deployment.

3. **Moderate-class performance.** The moderate class has the lowest recall (0.505). This reflects the inherent ambiguity of the moderate category. In practice, false moderate→low is less dangerous than false high→low.

4. **LLM quality variation.** The production mode (Qwen3-4B) produces better-quality and more natural text than the demo mode (MockGuidanceModel). The Phase-6 evaluation covers only the production mode.

5. **RAG retrieval quality.** The factor-aware retrieval improves diversity, but the quality of retrieved chunks depends on the embedding model. BGE-M3 (via Ollama) substantially outperforms the offline TF-IDF fallback for semantic retrieval.

6. **Fine-tuning not yet done.** Qwen3-4B is used as a base (un-fine-tuned) model. Fine-tuning with RAFT-generated examples (built, not yet run) is expected to improve safety self-containment and output consistency.

7. **English only.** The knowledge base and prompt are in English. No multilingual support.

8. **No longitudinal tracking.** The system evaluates one point-in-time screening. Trend tracking (how does risk change over serial measurements?) is not implemented.

---

## 11. Citations & References

### Clinical Guidelines

[1] Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. *KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease.* Kidney International. 2024;105(Suppl 4S):S117–S314.

[2] Kidney Disease: Improving Global Outcomes (KDIGO) Blood Pressure Work Group. *KDIGO 2021 Clinical Practice Guideline for the Management of Blood Pressure in Chronic Kidney Disease.* Kidney International. 2021;99(3S):S1–S87.

[3] Kidney Disease: Improving Global Outcomes (KDIGO) Diabetes Work Group. *KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease.* Kidney International. 2022;102:S1–S127.

[4] Ikizler TA, Burrowes JD, Byham-Gray LD, et al. *KDOQI Clinical Practice Guideline for Nutrition in CKD: 2020 Update.* American Journal of Kidney Diseases. 2020;76(3):S1–S107.

[5] Centers for Disease Control and Prevention (CDC). *Chronic Kidney Disease Basics.* National Center for Chronic Disease Prevention and Health Promotion. 2024.

[6] Centers for Disease Control and Prevention (CDC). *Preventing Chronic Kidney Disease.* 2024.

[7] National Kidney Foundation (NKF). *Staying Active on Dialysis or After Transplant: Benefits, Tips, and Safe Workouts.* 2023.

### Training Data

[8] Centers for Disease Control and Prevention / National Center for Health Statistics (CDC/NCHS). *National Health and Nutrition Examination Survey (NHANES) 2021–2023 Datasets.* Available at: https://www.cdc.gov/nchs/nhanes/

### ML & CKD Detection Literature

[9] Ramspek CL, Steyerberg EW, Riley RD, et al. Prediction of kidney failure in patients with chronic kidney disease: a systematic review of prediction models. *Nephrology Dialysis Transplantation.* 2021;36:1066–1076.

[10] Aljaaf AJ, Al-Jumeily D, Hussain AJ, et al. Early prediction of chronic kidney disease using machine learning supported by predictive analytics. *Proceedings of the 2019 IEEE Congress on Evolutionary Computation (CEC)*. 2019.

[11] Chittora P, Chaurasia S, Chakrabarti P, et al. Prediction of chronic kidney disease — a machine learning perspective. *IEEE Access.* 2021;9:17312–17334.

[12] Yildirim P. Chronic kidney disease prediction on imbalanced data by multilayer perceptron: chronic kidney disease prediction. *Proceedings of the 2017 IEEE 41st Annual Computer Software and Applications Conference (COMPSAC)*. 2017.

### Imbalanced Learning

[13] Lemaître G, Nogueira F, Aridas CK. Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *Journal of Machine Learning Research.* 2017;18(17):1–5.

### Language Models

[14] Qwen Team (Alibaba). *Qwen3 Technical Report.* 2025. Model: Qwen3-4B-Instruct-2507.

### Embeddings

[15] Chen J, Xiao S, Zhang P, et al. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv preprint arXiv:2309.07597. 2023.

### RAG

[16] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems.* 2020;33:9459–9474.

### LLM Safety in Healthcare

[17] Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. *Nature.* 2023;620:172–180.

[18] Thirunavukarasu AJ, Ting DSJ, Elangovan K, et al. Large language models in medicine. *Nature Medicine.* 2023;29:1930–1940.

[19] Li I, Pan J, Goldwasser J, et al. Neural natural language processing for unstructured data in electronic health records: a review. *Computer Science Review.* 2022;46:100511.

---

## 12. Frontend Component Reference

### 12.1 Component Inventory

| Component | File | Responsibility |
|---|---|---|
| `App` | `src/App.jsx` | Root: shared state (values, guidance, view), health polling, demo runner |
| `AppShell` | `src/components/AppShell.jsx` | Sidebar navigation, status bar, disclaimer footer |
| `PatientView` | `src/components/patient/PatientView.jsx` | Form + result display container |
| `PatientForm` | `src/components/patient/PatientForm.jsx` | 22-field input form in 5 groups (Basic, Body, BP, Labs, History) |
| `PatientResult` | `src/components/patient/PatientResult.jsx` | Risk card, reliability badge, key signals, 5-section education, safety notes |
| `AdminDashboard` | `src/components/admin/AdminDashboard.jsx` | Tabbed dashboard (8 tabs) |
| `PredictionPanels` | `src/components/admin/PredictionPanels.jsx` | Probabilities, stage outputs, input quality, explanation |
| `LLMPanels` | `src/components/admin/LLMPanels.jsx` | RAG query, chunk details, grounding ratio, validator audit |
| `EvaluationPanel` | `src/components/admin/EvaluationPanel.jsx` | Frozen Phase-6 metrics display |
| `ExportButtons` | `src/components/admin/ExportButtons.jsx` | JSON/CSV download of full response |
| `DemoCases` | `src/components/DemoCases.jsx` | One-click Low / Moderate / High-emergency case runner |
| `SystemHealth` | `src/components/SystemHealth.jsx` | Backend/model/LLM/database status display |
| `ui` | `src/components/ui.jsx` | Card, SectionTitle, Icon, Spinner, Chip, Badge, Tabs, JSON viewer |

### 12.2 Key Data Files (Frontend)

| File | Contents |
|---|---|
| `src/data/featureSchema.js` | 22 feature definitions: label, unit, type, default, range, help text |
| `src/data/demoCases.js` | 3 pre-filled payloads: Low / Moderate / High-emergency |
| `src/data/evaluationData.js` | Frozen Phase-6 metrics (hardcoded for offline display) |
| `src/lib/risk.js` | Risk level colour + icon + label helpers |
| `src/lib/exportUtils.js` | JSON/CSV serialization for export |
| `src/api/client.js` | Axios client — `GET /health`, `GET /generate-guidance/health`, `POST /generate-guidance` |

### 12.3 Environment

```env
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 13. Database Schema

The database layer is **optional and best-effort** — the API never blocks or fails if the database is unreachable.

### 13.1 Table: `screenings`

| Column | SQL Type | Purpose |
|---|---|---|
| `id` | INTEGER PRIMARY KEY (autoincrement) | Row identifier |
| `created_at` | TIMESTAMPTZ (server_default = now()) | When the screening was run |
| `source` | VARCHAR(32) | `"predict"` or `"generate-guidance"` |
| `input_features` | JSONB / JSON | Raw 22-feature payload from the user |
| `n_missing_features` | INTEGER | Count of features that were imputed |
| `risk_class` | VARCHAR(16) (indexed) | `"low"` / `"moderate"` / `"high"` |
| `risk_class_display` | VARCHAR(64) | Human-readable label |
| `confidence` | FLOAT | Max class probability |
| `probabilities` | JSONB | `{low, moderate, high}` probability map |
| `prediction` | JSONB | Full `/predict` response (all stages, explanation, reliability) |
| `guidance` | JSONB | Full `/generate-guidance` response (populated only for that endpoint) |
| `safety_status` | VARCHAR(16) | `"pass"` / `"sanitized"` / `"fallback"` |
| `llm_mode` | VARCHAR(32) | `"production"` / `"local_demo"` |
| `model_spec` | VARCHAR(128) | LLM model identifier string |

### 13.2 Setup

```bash
# Env var (in .env)
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/ckd_research
# or for local SQLite:
DATABASE_URL=sqlite:///./ckd_local.db

# Run migrations
alembic upgrade head
```

Auto-table creation is enabled by default (`CKD_DB_AUTO_CREATE=true`), so the table is created on first startup without running migrations manually.

---

## 14. Project Directory Structure

```
Nephro-AI/
├── backend/
│   ├── main.py                    # FastAPI app, CORS, routes, lifespan (init_db)
│   ├── schemas.py                 # Pydantic: PatientFeatures, PredictionResponse
│   ├── feature_config.py          # MODEL_FEATURES (22), FEATURE_METADATA, NUMERIC_RANGES
│   ├── model_service.py           # KidneyRiskModelService: load, score, explain, reliability
│   ├── model_runtime.py           # Pickle alias shims for joblib compatibility
│   ├── llm_endpoint.py            # build_guidance_response(), llm_health()
│   └── database/
│       ├── models.py              # SQLAlchemy Screening ORM
│       ├── db.py                  # Engine, session factory
│       ├── repository.py          # save_screening(), count_screenings()
│       └── migrations/            # Alembic version files
│
├── llm/
│   ├── serving/
│   │   ├── contract.py            # LLMInput, GuidanceOutput, parse_guidance()
│   │   ├── prompts.py             # System/user prompt builders, safety rules, section hints
│   │   ├── llm_runtime.py         # GuidanceRuntime.generate_guidance() → GuidanceResult
│   │   ├── model_backends.py      # QwenChatModel, OllamaChatModel, MockGuidanceModel
│   │   └── safety_validator.py    # 7-layer enforce(), scan_unsafe(), fallback_text()
│   ├── rag/
│   │   ├── retriever.py           # Retriever: build_query, priority_topics, retrieve_for_llm_input
│   │   ├── vector_store.py        # VectorStore: load, search (FAISS or brute-force)
│   │   ├── embed_index.py         # CLI: build/rebuild the embedding index
│   │   ├── embedder.py            # BGEM3Embedder, OllamaEmbedder, HashingTfidfEmbedder
│   │   ├── ingest.py              # PDF/markdown → chunks.jsonl (Phase 2)
│   │   └── audit.py               # Knowledge base safety/coverage audit
│   ├── knowledge_base/
│   │   ├── raw/                   # Original source files (10 PDFs + 1 md)
│   │   ├── processed/
│   │   │   └── chunks.jsonl       # 118 annotated chunks
│   │   └── index/
│   │       ├── meta.json          # Embedder spec, chunk metadata
│   │       ├── vectors.json       # L2-normalized embedding matrix
│   │       └── bge_faiss.index    # FAISS binary (optional, GPU/prod)
│   ├── data/
│   │   └── profiles.py            # 30 held-out evaluation profiles
│   └── eval/
│       ├── score.py               # Scoring: format, readability, safety, personalization, grounding
│       └── run_phase6.py          # Phase-6 evaluation harness
│
├── frontend/
│   ├── package.json               # React 18, Vite 5, Tailwind 3, Axios
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── .env / .env.example        # VITE_API_BASE_URL
│   └── src/
│       ├── main.jsx               # React entry point
│       ├── App.jsx                # Root component, state, view routing
│       ├── index.css
│       ├── api/client.js          # Axios API wrapper
│       ├── components/
│       │   ├── AppShell.jsx
│       │   ├── ui.jsx
│       │   ├── DemoCases.jsx
│       │   ├── SystemHealth.jsx
│       │   ├── patient/
│       │   │   ├── PatientView.jsx
│       │   │   ├── PatientForm.jsx
│       │   │   └── PatientResult.jsx
│       │   └── admin/
│       │       ├── AdminDashboard.jsx
│       │       ├── PredictionPanels.jsx
│       │       ├── LLMPanels.jsx
│       │       ├── EvaluationPanel.jsx
│       │       └── ExportButtons.jsx
│       ├── data/
│       │   ├── featureSchema.js
│       │   ├── demoCases.js
│       │   └── evaluationData.js
│       └── lib/
│           ├── risk.js
│           └── exportUtils.js
│
├── model_package/
│   ├── joint_s1_rfbalanced_k19_s2_rfbalanced_k12_stage2_thr050_model_package.joblib
│   ├── nhanes_FINAL_CLEAR_BASELINE_STAGE2_THR050_MODEL_NOTEBOOK (1).ipynb
│   └── nhanes_PHASE1_CV_BASELINES_SHAP_COLAB_READY_FIXED_SHAP (1).ipynb
│
├── results/
│   ├── phase6_summary.md          # Frozen Phase-6 evaluation metrics + example outputs
│   └── thesis_demo_cases.md       # Supervisor demo case outputs (3 cases)
│
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md # This file
│
├── sample_data/                   # Sample /predict request payload
├── scripts/                       # test_api_request.py
├── requirements.txt               # Backend (thesis MVP / local-demo)
├── requirements-llm.txt           # Optional GPU LLM deps (transformers, torch, faiss)
├── alembic.ini                    # Database migration config
├── .env / .env.example            # Backend config
├── README.md
├── SETUP.md
├── setup_and_run.ps1              # Windows one-command setup & launch
├── run_backend.bat
└── run_frontend.bat
```

---

## 15. Local Demo vs Production Comparison

| Aspect | Local Demo (`local_demo`) | Ollama (`ollama`) | Production (`qwen`) |
|---|---|---|---|
| **LLM** | MockGuidanceModel (deterministic Python) | Qwen3-4B via Ollama REST API | Qwen3-4B via HuggingFace Transformers |
| **Embedder** | HashingTFIDF (pure Python, 4096-d) | BGE-M3 via Ollama REST API (1024-d) | BGE-M3 via sentence-transformers (1024-d) |
| **GPU required** | No | No (Ollama handles offload) | Optional (CPU works, slower) |
| **Internet required** | No | No (after `ollama pull`) | No (after model download) |
| **Output quality** | Condition-aware, not natural language | Full LLM quality | Full LLM quality |
| **Safety validator** | Same (runs in all modes) | Same | Same |
| **UI mode label** | "Local demo mode" | "Production" | "Production" |
| **Env var** | `CKD_LLM_MODEL=local_demo` | `CKD_LLM_MODEL=ollama` | `CKD_LLM_MODEL=qwen` |
| **Best for** | Supervisor demos, CI testing, CPU laptops | Local GPU laptops (4GB VRAM+) | Colab, GPU servers, final eval |
| **Phase-6 eval** | Not applicable (harness test only) | ✅ (used for thesis results) | ✅ (primary path) |

---

## 16. Quick-Reference Numbers

| Parameter | Value |
|---|---|
| ML input features | 22 |
| Risk classes | 3 (low, moderate, high) |
| Stage 1 threshold | 0.44 |
| Stage 2 threshold | 0.50 |
| High→low misses (test set) | 2 |
| Knowledge base sources | 9 |
| Knowledge base chunks | 118 |
| Chunk size | ~150 words |
| RAG retrieval k | 4 (default) |
| Output sections | 5 |
| Safety rules in system prompt | 9 |
| Validator layers | 7 |
| Evaluation profiles | 30 (10 per class) |
| Final unsafe rate (Phase 6) | **0%** |
| Readability target | Grade ≤ 8 (FK) |
| Achieved readability | Grade 6.9 (FK) |
| LLM temperature (Ollama) | 0.65 |
| LLM max tokens (Ollama) | 900 |
| BGE-M3 embedding dim | 1024 |
| Hashing-TF-IDF embedding dim | 4096 |
| Model file size | ~1.5 MB (.joblib) |

---

*This documentation reflects the Thesis MVP state as of June 2026. Fine-tuning (QLoRA), GGUF packaging, and expert evaluation are planned for the paper extension phase.*
