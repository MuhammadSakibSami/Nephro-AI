# CKD Risk Screening App

FastAPI backend + Streamlit frontend for the final NHANES two-stage kidney-risk screening cascade.

## What this app does

The user inputs the 22 model features in the Streamlit frontend. The frontend sends the values to the FastAPI backend. The backend loads your saved `.joblib` final model package, runs the two-stage cascade, and returns:

- final risk class: `low`, `moderate`, or `high`
- class probabilities
- Stage-1 low/at-risk gate probability
- Stage-2 high-risk probability among at-risk cases
- missing-feature audit
- model threshold information

This is a **screening-support tool**, not a diagnostic medical device.

## Project structure

```text
ckd_risk_app/
├── backend/
│   ├── main.py              # FastAPI endpoints
│   ├── model_service.py     # model loading + prediction logic
│   ├── model_runtime.py     # custom classes needed for joblib loading
│   ├── schemas.py           # API request/response schemas
│   └── feature_config.py    # feature metadata used by API and frontend
├── frontend/
│   └── app.py               # Streamlit UI
├── model_package/
│   └── PUT_MODEL_PACKAGE_HERE.txt
├── sample_data/
│   └── sample_patient.json
├── scripts/
│   ├── test_api_request.py
│   ├── run_backend.sh
│   └── run_frontend.sh
├── run_backend.bat
├── run_frontend.bat
├── requirements.txt
└── README.md
```

## Step 1: Create environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Step 2: Copy the final model package

Run your final notebook first. It should save:

```text
nhanes_FINAL_CLEAR_STAGE2_THR050_outputs/model_package/joint_s1_rfbalanced_k19_s2_rfbalanced_k12_stage2_thr050_model_package.joblib
```

Copy that file into this project:

```text
ckd_risk_app/model_package/joint_s1_rfbalanced_k19_s2_rfbalanced_k12_stage2_thr050_model_package.joblib
```

Alternative: set an environment variable:

```bash
export CKD_MODEL_PATH="/full/path/to/joint_s1_rfbalanced_k19_s2_rfbalanced_k12_stage2_thr050_model_package.joblib"
```

Windows PowerShell:

```powershell
$env:CKD_MODEL_PATH="C:\full\path\to\joint_s1_rfbalanced_k19_s2_rfbalanced_k12_stage2_thr050_model_package.joblib"
```

## Step 3: Run FastAPI backend

From project root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Check health:

```text
http://127.0.0.1:8000/health
```

## Step 4: Run Streamlit frontend

Open a second terminal from project root:

```bash
streamlit run frontend/app.py
```

Then open the Streamlit URL shown in the terminal.

## Test the API

After backend is running:

```bash
python scripts/test_api_request.py
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend and model loading status |
| GET | `/features` | Return model feature schema |
| GET | `/model-info` | Return model thresholds/features from package |
| POST | `/predict` | Predict one patient |
| POST | `/reload-model` | Reload model after replacing joblib file |

## Prediction payload example

```json
{
  "age": 62,
  "sex_female": 0,
  "income_poverty_ratio": 2.1,
  "bmi": 30.2,
  "waist": 105.0,
  "mean_sbp": 138.0,
  "mean_dbp": 82.0,
  "mean_pulse": 76.0,
  "glucose": 118.0,
  "hba1c": 6.4,
  "blood_urea_nitrogen": 22.0,
  "uric_acid": 6.8,
  "serum_albumin": 4.0,
  "sodium": 140.0,
  "potassium": 4.4,
  "hemoglobin": 13.2,
  "diabetes_yes": 1,
  "borderline_diabetes": 0,
  "hypertension_doctor_told": 1,
  "high_cholesterol_told": 1,
  "taking_bp_medicine": 1,
  "taking_cholesterol_medicine": 1
}
```

## Important model note

The model package itself is not included in this code bundle unless you copy it from the notebook output. The uploaded file was a notebook, not the trained `.joblib` package. The backend will show `model_not_loaded` until the `.joblib` file is present or `CKD_MODEL_PATH` is set.

## Clinical note

This app is for research and screening support only. It does not diagnose CKD and should not replace clinician judgment, eGFR/UACR testing, or medical evaluation.
