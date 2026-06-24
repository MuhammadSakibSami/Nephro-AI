# Kidney-Risk Screening — Frontend

A modern, thesis-demo-ready frontend for the non-invasive kidney-risk screening
project. Two interfaces in one app:

- **Patient View** — simple, calm, non-technical: enter routine values → risk
  result + personalized, guideline-grounded kidney-risk **education**.
- **Admin / Research View** — model internals, RAG/LLM details, safety validator,
  and the frozen thesis evaluation metrics.

Plus **Demo Cases** (one-click supervisor cases) and **System Health**.

> The LLM component provides personalized, guideline-grounded kidney-risk
> **education** after ML-based screening. It does **not** diagnose CKD, prescribe
> treatment, or replace clinician consultation.

Stack: React 18 + Vite + Tailwind CSS. No backend code here — it talks to the
existing FastAPI backend.

---

## 1. Prerequisites

- **Node.js ≥ 18** (tested on Node 24).
- The project **backend running** (FastAPI) with the ML model + LLM endpoints:
  - `POST /predict`
  - `POST /generate-guidance`
  - `GET /generate-guidance/health`, `GET /health`

The backend works in either LLM mode:
- `qwen` (production, GPU) — set `CKD_LLM_MODEL=qwen` when serving.
- `local_demo` (no GPU) — automatic fallback; the app labels it clearly.

---

## 2. Run the frontend

```bash
cd frontend
cp .env.example .env          # set VITE_API_BASE_URL if backend isn't on :8000
npm install
npm run dev                   # http://localhost:5173
```

Production build:
```bash
npm run build && npm run preview
```

### Configure the backend URL
`.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```

---

## 3. Run the backend (for reference)

From the project root (needs the ML stack: fastapi, uvicorn, numpy, pandas,
scikit-learn, imbalanced-learn, joblib):

```bash
# optional: pick the LLM model (defaults to "auto" -> qwen if available, else local_demo)
export CKD_LLM_MODEL=local_demo
uvicorn ckd_risk_app.backend.main:app --reload --port 8000
```

CORS is already open (`allow_origins=["*"]`) so the Vite dev server can call it.

---

## 4. Using it for the demo

1. **Demo Cases** tab → click **Low / Moderate / High** → it runs the full
   pipeline and opens the patient result.
2. Switch to **Admin / Research** to show the same run's internals (prediction,
   stage-1/2, RAG chunks, safety validator actions, evaluation metrics).
3. The **high / emergency-BP** case shows the safety validator adding an
   **urgent-care redirect** (status = `sanitized`).
4. **Export / Tools** tab → download prediction/guidance JSON or copy the
   patient / admin reports.

The **Evaluation** tab works even without a backend (static frozen metrics).

---

## 5. Project structure

```
src/
  api/client.js                 # central API client (.env base URL)
  data/
    featureSchema.js            # form fields (mapped to backend feature keys)
    demoCases.js                # 3 supervisor demo payloads
    evaluationData.js           # frozen ML + LLM Phase-6 metrics
  lib/{risk,exportUtils}.js     # presentation + export helpers
  components/
    AppShell.jsx                # sidebar, status bar, disclaimer footer
    ui.jsx                      # cards, tabs, chips, icons, json viewer
    patient/                    # PatientView, PatientForm, PatientResult
    admin/                      # AdminDashboard + panels + ExportButtons
    DemoCases.jsx, SystemHealth.jsx
  App.jsx                       # shared state + view routing
```

---

## 6. Notes for the thesis/paper

- The app never claims the model **diagnoses** CKD — everything is framed as a
  **screening result** + **education**.
- It does not claim absolute safety. It states the measured result:
  *"In our Phase 6 evaluation, the final unsafe output rate was 0% after safety
  validation."*
- `local_demo` mode is clearly separated from real `qwen` mode in the UI.
- Patient view deliberately hides raw probabilities, thresholds, chunk IDs,
  validator labels, and debug metadata — those live only in the Admin view.
