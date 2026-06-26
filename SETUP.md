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
| `CKD_LLM_MODEL` | `local_demo` · `ollama` · `qwen` · `auto` | LLM backend / mode |
| `OLLAMA_MODEL` | model tag (optional) | Ollama model to use (default: `qwen3:4b-instruct-2507-q4_K_M`) |
| `OLLAMA_BASE_URL` | URL (optional) | Ollama server URL (default: `http://localhost:11434`) |
| `DATABASE_URL` | SQLAlchemy URL (optional) | PostgreSQL connection for saving screenings. Unset = persistence off (see §6) |
| `CKD_DB_AUTO_CREATE` | `true` (default) · `false` | auto-create tables on startup vs. manage with Alembic |
| `CKD_MODEL_PATH` | path (optional) | override the ML model `.joblib` location |
| `GEMINI_API_KEY` | key (optional) | only for the deferred Phase-7 data scripts |

**Frontend** — `frontend/.env` (see `frontend/.env.example`):

| Variable | Example | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | backend URL the UI calls |

---

## 5. Optional: production Qwen3 (two paths)

Local-demo mode is enough for the thesis demo. When you want the real model,
choose the path that matches your hardware:

### 5a. Ollama — local laptop with a 4 GB GPU (e.g. RTX 3050 Ti)

No Python ML libraries needed. Ollama handles model loading and GPU offload.
The Q4_K_M quantization of Qwen3-4B needs ~3 GB VRAM — fits a 4 GB card.

```powershell
# 1. Install Ollama (one-time): https://ollama.com/download
# 2. Pull the model (~2.4 GB download, one-time):
ollama pull qwen3:4b-instruct-2507-q4_K_M
# 3. Edit .env:
#      CKD_LLM_MODEL=ollama
# 4. Start the backend normally — it connects to Ollama automatically.
```

Ollama starts its server automatically on Windows once installed.
To switch back to local-demo: change `.env` to `CKD_LLM_MODEL=local_demo` and
restart the backend. No other code changes.

> Use the **official Ollama library model** (`ollama pull qwen3:4b-instruct-2507-q4_K_M`),
> not community-uploaded variants — the library version is quantized by the Ollama team.

#### Sharper, more personalized retrieval (recommended with Ollama)

By default the RAG index uses a lightweight pure-Python (TF-IDF) embedder. It runs
anywhere but retrieves coarsely, so the guideline grounding — and therefore the
generated education — can look similar across patients. For noticeably more
**personalized** output, rebuild the index with **BGE-M3 served by Ollama** (no
PyTorch needed):

```powershell
ollama pull bge-m3
python -m llm.rag.embed_index --embedder ollama-bge-m3   # one-time rebuild
```

The embedder used is recorded in the index, so the backend automatically uses
BGE-M3 for queries too (keep `bge-m3` pulled wherever the backend runs). To
revert: `python -m llm.rag.embed_index --embedder hashing-tfidf`.

### 5b. HuggingFace transformers — Colab or dedicated GPU server (≥8 GB VRAM)

```bash
pip install -r requirements-llm.txt        # torch, transformers, sentence-transformers, faiss
python -m llm.rag.embed_index              # build the BGE-M3 + FAISS index
# set CKD_LLM_MODEL=qwen in .env, then start the backend
```

Downloads the full Qwen3-4B-Instruct-2507 model (~8 GB) on first run.
Requires a CUDA GPU with ≥8 GB VRAM (or CPU offload in Colab with enough RAM).
The frontend and safety pipeline are unchanged; the UI badge switches from
`local demo` to the model spec.

---

## 6. Database (PostgreSQL) — saving the research dataset

Persistence is **optional and best-effort**: if `DATABASE_URL` is not set (or the
database is unreachable), the API still runs and screening works — saving is just
skipped. When configured, **every `/predict` and `/generate-guidance` call is
stored** (patient inputs + ML prediction + LLM guidance) in a `screenings` table.

### 6a. One-time setup

```sql
-- In psql (or pgAdmin), create a database and a user:
CREATE DATABASE ckd_research;
CREATE USER ckd_user WITH PASSWORD 'change_me';
GRANT ALL PRIVILEGES ON DATABASE ckd_research TO ckd_user;
```

Then set the connection string in `./.env` (repo root):

```ini
DATABASE_URL=postgresql+psycopg2://ckd_user:change_me@localhost:5432/ckd_research
```

The PostgreSQL driver (`psycopg2-binary`) ships in `requirements.txt`. Start the
backend — on startup it connects, creates the table (because `CKD_DB_AUTO_CREATE`
defaults to `true`), and begins saving. Verify:

```
GET http://localhost:8000/db/health
  -> { "available": true, "configured": true, "stored_screenings": <n>, ... }
```

### 6b. Migrations with Alembic (recommended for the final research)

For controlled schema changes (instead of auto-create), use the bundled Alembic
setup. Set `CKD_DB_AUTO_CREATE=false`, then from the repo root (venv active):

```bash
alembic upgrade head                                   # apply the schema
# after you edit backend/database/models.py:
alembic revision --autogenerate -m "describe change"   # generate a migration
alembic upgrade head                                   # apply it
```

`DATABASE_URL` is read from `.env`/the environment — it is never stored in
`alembic.ini`.

### 6c. What gets stored

One row per screening call in `screenings`: `created_at`, `source`
(`predict` | `generate-guidance`), the raw `input_features` (JSONB), `risk_class`
/ `confidence` / `probabilities`, the full `prediction`, and (for guidance calls)
the `guidance` sections + `safety_status` + `llm_mode` + `model_spec`. JSON
columns mean new fields can be added later without a migration; to add new
**tables** (e.g. patients, feedback), add a model in
`backend/database/models.py` and generate a migration.

> A typical UI run calls `/generate-guidance` (which screens internally). If the
> admin view also calls `/predict`, you'll get two rows — distinguishable by the
> `source` column. That's intended: one row per API call.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `attempted relative import beyond top-level package` | Start the backend from the **repo root** with `uvicorn backend.main:app` (not from inside `backend/`). |
| `Model package not found` | Put the `.joblib` in `model_package/`, or set `CKD_MODEL_PATH` in `.env`. |
| Frontend shows "Could not reach the screening service" | Backend not running, or `VITE_API_BASE_URL` wrong. Restart `npm run dev` after editing `frontend/.env`. |
| LLM badge says `local demo` but you wanted `ollama` | Install Ollama, run `ollama pull qwen3:4b-instruct-2507-q4_K_M`, set `CKD_LLM_MODEL=ollama`, restart backend. |
| LLM badge says `local demo` but you wanted `qwen` | Install `requirements-llm.txt`, build the index, set `CKD_LLM_MODEL=qwen`. |
| Ollama connection refused (`ConnectionError`) | Ollama server isn't running. Start it: `ollama serve` (or reopen the Ollama desktop app). |
| `uvicorn`/`npm` not found | Activate the venv (`venv\Scripts\activate`) / install Node.js. |
| PowerShell won't run the script | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `/db/health` shows `available: false` | Check `DATABASE_URL` in `.env` and that PostgreSQL is running. The `error` field explains why. Screening still works without it. |
| `psycopg2` / driver build errors | `requirements.txt` uses `psycopg2-binary` (no build needed). Ensure the venv deps are installed. |
| Alembic: "DATABASE_URL is not set" | Set it in `.env` (or the shell) before running `alembic`. |

---

## 8. Project layout (relevant parts)

```
<repo root>/
├── backend/                # FastAPI: /predict, /generate-guidance, /health
│   └── database/           # SQLAlchemy models + repository + Alembic migrations
├── alembic.ini             # Alembic config (reads DATABASE_URL from env)
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
