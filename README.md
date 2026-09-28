# Kidney-Risk Screening + Guideline-Grounded Education

A routine-blood-test-based, cross-sectional **chronic-kidney-disease (CKD) risk
screening** system that pairs a two-stage machine-learning model with a
**safety-guarded, guideline-grounded LLM** that turns the risk result into
personalized, plain-language kidney-health **education**.

- **ML model** — predicts a KDIGO-aligned risk level (low / moderate / high) from
  14 routine, low-cost inputs (vitals, standard blood tests, history) already
  covered by a general checkup — *no creatinine / eGFR / UACR required as
  inputs*. Each screening is a cross-sectional, point-in-time estimate, not a
  longitudinal diagnosis.
- **LLM component** — retrieves relevant clinical-guideline snippets (RAG), writes
  5-section education, and passes every output through a strict **safety
  validator** before the user sees it.
- **Web app** — a modern React dashboard with a **Patient view** and an
  **Admin / Research view**.

> **Scope & safety.** This is a **screening + education** tool. It does **not**
> diagnose CKD, prescribe treatment, or replace clinician consultation. The LLM
> never gives medication or restrictive-diet instructions.

---

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Using the app](#using-the-app)
- [Evaluation & results](#evaluation--results)
- [Project status & roadmap](#project-status--roadmap)
- [Documentation](#documentation)
- [Safety, scope & disclaimer](#safety-scope--disclaimer)
- [Data sources & acknowledgements](#data-sources--acknowledgements)

---

## Highlights

- **Two-stage cascade** (RandomForest gate → GradientBoosting) trained on
  **NHANES 2021–23**, optimized to minimize dangerous high→low misses.
- **Guideline RAG** over a safety-scoped knowledge base (KDIGO, ADA, NKF/KDOQI,
  CDC) — medication / dialysis / transplant / numeric-target content is **excluded
  from retrieval** by design.
- **Strict safety validator** — blocks diagnosis/medication/supplement content,
  rewrites nutrition restrictions to a dietitian referral, appends lab-test
  reminders, and adds urgent-care redirects for emergency vitals.
- **Runs anywhere** — a **local-demo mode** (no GPU, pure-Python RAG + safe
  generator) for laptops/demos, and an optional **production mode**
  (Qwen3-4B + BGE-M3) for GPU/Colab. The app labels the active mode.
- **Research-grade transparency** — the Admin view exposes probabilities, stage
  outputs, retrieved chunks, validator actions, and frozen evaluation metrics.

---

## Architecture

```
Patient inputs
   │
   ▼
Two-stage ML cascade  ── /predict ──►  risk class · probabilities · top factors · reliability
   │
   ▼
Guideline RAG  (BGE-M3 + FAISS  |  TF-IDF fallback offline)
   │   retrieves guideline chunks relevant to the patient's drivers
   ▼
Prompt builder ──►  LLM  (Qwen3-4B  |  local-demo generator)
   │
   ▼
Safety Validator  (rule layers → pass / sanitize / safe-fallback)
   │
   ▼
Safe, personalized kidney-risk education  ── /generate-guidance ──►  React UI
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML model | scikit-learn, imbalanced-learn (two-stage cascade, `.joblib`) |
| RAG | BGE-M3 embeddings + FAISS (prod) · pure-Python TF-IDF (offline) |
| LLM | Qwen3-4B-Instruct-2507 + LoRA adapter (prod) · deterministic safe generator (demo) |
| Fine-tuning | PEFT LoRA (r=16, α=32), trained in bf16 without quantization |
| Safety | Rule-based validator (pure Python) |
| Frontend | React 18 + Vite + Tailwind CSS |
| Database | PostgreSQL + SQLAlchemy + Alembic |
| Config | `.env` (python-dotenv) · `VITE_*` env |

### Compute environment (verified)

| Item | Value |
|---|---|
| GPU model | NVIDIA GeForce RTX 3090 |
| VRAM | 24 GB |
| PyTorch version | 2.5.1+cu121 (CUDA 12.1) |
| Inference precision | bfloat16, no quantization (LoRA adapter weights float32) |
| transformers version | 5.14.1 |

The same machine and library versions were used for fine-tuning, the final Q1
evaluation and live serving. Full record: [docs/HARDWARE_AND_ENVIRONMENT.md](docs/HARDWARE_AND_ENVIRONMENT.md).

---

## Project structure

```
<repo root>/
├── backend/                  # FastAPI: /predict, /generate-guidance, /health, ...
│   ├── main.py               #   app + routes (loads .env)
│   ├── model_service.py      #   Final Model C two-stage cascade loading + scoring
│   ├── llm_endpoint.py       #   wires RAG + LLM + validator into the API
│   ├── schemas.py            #   request/response models
│   └── feature_config.py     #   feature metadata
├── llm/                      # LLM component
│   ├── serving/              #   contract, prompts, runtime, safety validator, models
│   ├── rag/                  #   ingest, embed_index, retriever, vector store, audit
│   └── knowledge_base/       #   guideline sources + processed chunks + index
├── frontend/                 # React + Vite + Tailwind UI (Patient + Admin)  — see frontend/README.md
├── model_package/            # trained ML model (.joblib)
├── results/                  # frozen evaluation evidence (Phase-6 CSV/MD, demo cases)
├── sample_data/              # sample request payload
├── scripts/                  # helper scripts (test_api_request.py, ...)
├── .env / .env.example       # backend config (CKD_LLM_MODEL, CKD_MODEL_PATH, ...)
├── requirements.txt          # backend deps (thesis MVP / local-demo)
├── requirements-llm.txt      # optional production-LLM deps (GPU)
├── run_backend.bat           # start backend (Windows)
├── run_frontend.bat          # start frontend (Windows)
├── setup_and_run.ps1         # one-command setup + run (Windows)
├── SETUP.md                  # detailed setup & run guide
└── README.md                 # this file
```

---

## Quick start

**Prerequisites:** Python 3.10–3.12, Node.js ≥ 18, and the model `.joblib` in
`model_package/`. Default LLM mode is **local-demo** (no GPU needed).

### Windows — one command

```powershell
# from the repo root
./setup_and_run.ps1          # sets up + launches backend (:8000) and frontend (:5173)
```
Then open **http://localhost:5173**. (Flags: `-SetupOnly`, `-BackendOnly`, `-FrontendOnly`.)

### Manual (any OS)

```bash
# Backend (from the repo root)
python -m venv venv && venv\Scripts\activate        # source venv/bin/activate on mac/Linux
pip install -r requirements.txt
copy .env.example .env                               # cp on mac/Linux
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Frontend (second terminal)
cd frontend
copy .env.example .env                               # cp on mac/Linux
npm install
npm run dev                                          # http://localhost:5173
```

> Always start the backend **from the repo root** with the module path
> `backend.main:app`. Full instructions, troubleshooting, and the optional GPU
> setup are in **[SETUP.md](SETUP.md)**.

---

## Configuration

**Backend** (`.env` at the repo root — see `.env.example`):

| Variable | Values | Purpose |
|---|---|---|
| `CKD_LLM_MODEL` | `local_demo` (default) · `auto` · `qwen` | LLM backend / mode |
| `CKD_MODEL_PATH` | path (optional) | override the model `.joblib` location |
| `GEMINI_API_KEY` | key (optional) | only for the deferred data-generation scripts |

**Frontend** (`frontend/.env` — see `frontend/.env.example`):

| Variable | Example | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | backend URL the UI calls |

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend + ML model status, runtime versions |
| `GET` | `/features` | Model feature schema + metadata |
| `GET` | `/model-info` | Model thresholds / selected features |
| `POST` | `/predict` | Run the two-stage cascade for one patient |
| `POST` | `/reload-model` | Reload the model after replacing the `.joblib` |
| `POST` | `/generate-guidance` | Full pipeline: predict → RAG → LLM → safety validator |
| `GET` | `/generate-guidance/health` | LLM mode / model availability |

Interactive docs at **http://localhost:8000/docs**.

`POST /generate-guidance` accepts the same patient features as `/predict` and
returns: `risk_class`, `guidance` (5 sections), safety `status`, `mode`,
`disclaimer`, `provenance`, the full `prediction`, and admin metadata (retrieved
chunks, query, grounding ratio, validator `violations`).

<details>
<summary>Sample <code>/predict</code> payload</summary>

```json
{
  "age": 62, "income_poverty_ratio": 2.1, "waist": 105.0,
  "mean_sbp": 138.0, "mean_dbp": 82.0,
  "glucose": 118.0, "hba1c": 6.4, "blood_urea_nitrogen": 22.0, "uric_acid": 6.8,
  "serum_albumin": 4.0, "hemoglobin": 13.2,
  "diabetes_yes": 1, "hypertension_doctor_told": 1, "taking_bp_medicine": 1
}
```
</details>

---

## Using the app

The frontend has four areas (sidebar): **Patient View**, **Admin / Research**,
**Demo Cases**, **System Health**.

- **Patient View** — a calm, non-technical form → risk card, reliability, key
  signals (plain terms), 5-section education, and safety messages. Internals are
  hidden.
- **Admin / Research** — tabs for prediction details, input quality, model
  explanation, LLM/RAG details, safety validator, frozen evaluation metrics, and
  export tools.
- **Demo Cases** — one-click **Low / Moderate / High-emergency** cases. The
  high-emergency case shows the validator adding an **urgent-care redirect**
  (status = `sanitized`).

See **[frontend/README.md](frontend/README.md)** for UI details.

---

## Evaluation & results

Frozen evidence lives in [`results/`](results/).

**Final Model C — two-stage cascade** (deployed full-development fit, NHANES 2021–23 held-out test):

| Accuracy | Balanced acc | Macro F1 | Low recall | Moderate recall | High recall | High→Low miss |
|---|---|---|---|---|---|---|
| 0.709 | 0.660 | 0.556 | 0.739 | 0.505 | 0.737 | 2 |

**LLM — Phase 6** (real Qwen3-4B + BGE-M3 RAG + validator, 30 held-out profiles):

| Format | Final unsafe | Fallback/Retry | Factor relevance | Personalization | Grounding | Readability (FK) | Raw unsafe |
|---|---|---|---|---|---|---|---|
| 1.00 | 0.00 | 0.00 / 0.00 | 0.966 | 1.93 / 2 | 1.00 | 6.9 | 0.10\* |

\* *All raw-unsafe cases were handled by the validator (emergency redirect /
nutrition deferral); zero diagnoses or medication advice.*
**In the Phase-6 evaluation, the final unsafe output rate was 0% after safety
validation.** Details: [`results/phase6_summary.md`](results/phase6_summary.md).

---

## Project status & roadmap

**Thesis MVP — complete:**
- ✅ Two-stage ML cascade + evaluation
- ✅ Guideline RAG (safety-scoped knowledge base + audit)
- ✅ Safety validator (raw 78% → final **0%** unsafe on the adversarial set)
- ✅ LLM runtime + `/generate-guidance` endpoint with local-demo fallback
- ✅ Phase-6 real-Qwen3 evaluation (**frozen** as the thesis LLM result)
- ✅ React frontend (patient + admin)
- ✅ Phase-7 training-data generation (600 self-distilled, safety-gated examples)
- ✅ Phase-8 LoRA fine-tuning of Qwen3-4B (bf16, RTX 3090) — adapter live in production
- ✅ Final Q1 six-arm ablation (300 profiles): deployed configuration at 0% delivered
  critical-unsafe, 100% emergency-redirect recall — see
  [results/q1_llm_evaluation_summary.md](results/q1_llm_evaluation_summary.md)
- ✅ PostgreSQL research persistence

**Still open:**
- ⏸ Clinician / expert review of generated guidance
- ⏸ GGUF/Ollama packaging of the fine-tuned model

---

## Documentation

| Doc | What it covers |
|---|---|
| [SETUP.md](SETUP.md) | Full setup & run guide (Windows + cross-platform), troubleshooting, optional GPU LLM |
| [frontend/README.md](frontend/README.md) | Frontend usage, components, env config |
| [docs/HARDWARE_AND_ENVIRONMENT.md](docs/HARDWARE_AND_ENVIRONMENT.md) | Verified GPU, VRAM, PyTorch/transformers versions, inference and training precision |
| [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | Full end-to-end technical documentation |
| [results/q1_llm_evaluation_summary.md](results/q1_llm_evaluation_summary.md) | Final Q1 six-arm LLM evaluation |
| [results/phase6_summary.md](results/phase6_summary.md) | Earlier Phase-6 LLM evaluation (superseded by Q1) |
| [results/thesis_demo_cases.md](results/thesis_demo_cases.md) | The three supervisor demo cases |

---

## Safety, scope & disclaimer

- The system produces a **screening result**, not a diagnosis. The LLM produces
  **education only** — it does not diagnose, prescribe, or give medication /
  numeric-diet instructions.
- It does not claim absolute safety; it reports a **measured** result (0% final
  unsafe output rate in the Phase-6 evaluation after validation).
- `local_demo` mode is clearly distinguished from production `qwen` mode in the UI.
- For research and screening support only — it does not replace clinician
  judgment, eGFR/UACR testing, or medical evaluation.

---

## Data sources & acknowledgements

- **Training data:** NHANES 2021–23 (CDC/NCHS), publicly available.
- **Target definition:** KDIGO CKD risk grid (eGFR + albuminuria categories).
- **Guideline knowledge base:** KDIGO 2024 / 2021 BP / 2022 Diabetes-in-CKD, ADA
  Standards of Care, NKF KDOQI nutrition, and CDC patient-education materials —
  used under their respective terms, scoped to prevention/lifestyle content. CDC
  materials are US-government public domain (with attribution).

> Built as a thesis project on routine-blood-test-based, cross-sectional
> kidney-disease risk screening with ML + LLM-powered patient education.
