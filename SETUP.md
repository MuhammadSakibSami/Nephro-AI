# Setup & Run Guide — Kidney-Risk Screening App

This guide gets the whole project running on a fresh machine: the **FastAPI
backend** (ML screening + LLM education) and the **React frontend** (patient +
admin/research views).

> The app is a **screening + education** tool. It does not diagnose CKD,
> prescribe treatment, or replace a clinician.

There are two ways to set up:
- **A. One-command PowerShell script** (Windows) — easiest.
- **B. Manual steps** (any OS) — full control / understanding.

---

## 0. Prerequisites

| Tool | Version | Check |
|---|---|---|
| **Python** | 3.10 – 3.12 | `python --version` |
| **Node.js** | ≥ 18 (tested on 24) | `node --version` |
| **npm** | ≥ 9 | `npm --version` |
| Git | any | `git --version` |

You also need the trained **model package** `.joblib` in `model_package/`
(it ships with the repo as
`joint_s1_rfbalanced_k19_s2_rfbalanced_k12_stage2_thr050_model_package.joblib`).

**LLM mode:** by default the app runs in **local-demo mode** — no GPU and no heavy
ML libraries required. Production Qwen3 is optional (see §5).

---

## A. Quick start (Windows PowerShell)

From the repo root (the folder containing `backend/`, `frontend/`, `requirements.txt`):

```powershell
# Sets up the Python venv + backend deps, copies .env files, installs the
# frontend, then launches backend (:8000) and frontend (:5173) in new windows.
./setup_and_run.ps1
```

Useful flags:
```powershell
./setup_and_run.ps1 -SetupOnly      # install everything, don't launch
./setup_and_run.ps1 -BackendOnly    # set up + run only the backend
./setup_and_run.ps1 -FrontendOnly   # set up + run only the frontend
```

Then open **http://localhost:5173**.

> If PowerShell blocks the script: run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

---

## B. Manual setup

### B1. Backend

From the repo root:

```bash
# 1) Create + activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2) Install backend dependencies
pip install -r requirements.txt

# 3) Configure environment (loaded automatically by the backend)
copy .env.example .env           # Windows  (cp on mac/Linux)
#   -> edit .env: CKD_LLM_MODEL=local_demo  (default; no GPU needed)

# 4) Run the API  (IMPORTANT: run from the repo root, module = backend.main)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify: open **http://localhost:8000/docs** (Swagger), or
**http://localhost:8000/health**.

> **Run-location matters.** Always start the backend from the repo root with the
> module path `backend.main:app` (as above, or via `run_backend.bat`). Running it
> from a different folder will break package imports.

### B2. Frontend

In a **second terminal**:

```bash
cd frontend
copy .env.example .env           # Windows  (cp on mac/Linux)
#   -> .env: VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev                      # http://localhost:5173
```

Open **http://localhost:5173**.

---

## 3. Using the app (supervisor demo)

1. **Demo Cases** tab → click **Low / Moderate / High** → runs the full pipeline
   (ML → RAG → LLM → safety validator) and opens the patient result.
2. Switch to **Admin / Research** to inspect the same run: probabilities,
   stage-1/2, retrieved RAG chunks, safety-validator actions, and the frozen
   thesis evaluation metrics.
3. The **High / emergency-BP** case shows the validator adding an **urgent-care
   redirect** (status = `sanitized`).
4. **System Health** tab shows backend / model / LLM-mode status.

---

## 4. Environment variables (summary)

**Backend** — `./.env` (repo root, see `.env.example`):

| Variable | Values | Purpose |
|---|---|---|
| `CKD_LLM_MODEL` | `local_demo` (default) · `auto` · `qwen` | LLM backend / mode |
| `CKD_MODEL_PATH` | path (optional) | override the model `.joblib` location |
| `GEMINI_API_KEY` | key (optional) | only for the deferred Phase-7 data scripts |

**Frontend** — `frontend/.env` (see `frontend/.env.example`):

| Variable | Example | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | backend URL the UI calls |

---

## 5. Optional: production Qwen3 + BGE-M3 (GPU / Colab)

Local-demo mode is enough for the thesis demo. For the real model:

```bash
pip install -r requirements-llm.txt        # torch, transformers, sentence-transformers, faiss
python -m llm.rag.embed_index              # build the BGE-M3 + FAISS index
# set CKD_LLM_MODEL=qwen in .env, then start the backend
```

This needs a CUDA GPU and downloads ~8 GB on first run. The frontend is unchanged;
the UI badge switches from `local demo` to `qwen`.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `attempted relative import beyond top-level package` | Start the backend from the **repo root** with `uvicorn backend.main:app` (not from inside `backend/`). |
| `Model package not found` | Put the `.joblib` in `model_package/`, or set `CKD_MODEL_PATH` in `.env`. |
| Frontend shows "Could not reach the screening service" | Backend not running, or `VITE_API_BASE_URL` wrong. Restart `npm run dev` after editing `frontend/.env`. |
| LLM badge says `local demo` but you wanted `qwen` | Install `requirements-llm.txt`, build the index, set `CKD_LLM_MODEL=qwen`. |
| `uvicorn`/`npm` not found | Activate the venv (`venv\Scripts\activate`) / install Node.js. |
| PowerShell won't run the script | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 7. Project layout (relevant parts)

```
<repo root>/
├── backend/                # FastAPI: /predict, /generate-guidance, /health
├── llm/                    # RAG + prompts + safety validator + runtime
├── frontend/               # React + Vite + Tailwind UI  (npm)
│   ├── .env / .env.example # VITE_API_BASE_URL
│   └── README.md
├── model_package/          # trained ML model (.joblib)
├── .env / .env.example     # backend config (CKD_LLM_MODEL, ...)
├── requirements.txt        # backend deps (thesis MVP)
├── requirements-llm.txt    # optional production-LLM deps (GPU)
├── run_backend.bat         # start backend
├── run_frontend.bat        # start frontend
├── setup_and_run.ps1       # one-command setup + run (Windows)
└── SETUP.md                # this file
```
