from __future__ import annotations

# Load backend .env (repo-root/.env) BEFORE importing model_service, so that
# CKD_MODEL_PATH / CKD_LLM_MODEL are available at import time. Safe no-op if
# python-dotenv is not installed (env vars can still be set in the shell).
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .feature_config import MODEL_FEATURES, FEATURE_METADATA, NUMERIC_FEATURES, BINARY_FEATURES
from .schemas import PatientFeatures, PredictionResponse
from .model_service import service, ModelNotLoadedError
from .llm_endpoint import build_guidance_response, health as llm_health
from .database import init_db, db_status, save_screening, count_screenings


def _runtime_versions():
    versions = {}
    for mod_name in ["numpy", "pandas", "sklearn", "imblearn", "joblib"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[mod_name] = f"not available: {exc}"
    return versions

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort: connects to PostgreSQL if DATABASE_URL is set. If it isn't
    # reachable, the API still runs and persistence is skipped (see /db/health).
    init_db()
    yield


app = FastAPI(
    title="CKD Risk Screening API",
    description="FastAPI backend for the final NHANES two-stage kidney-risk screening cascade.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "CKD Risk Screening API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "generate_guidance": "/generate-guidance",
        "llm_context": "returned inside /predict response",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if service.loaded else "model_not_loaded",
        "model_loaded": service.loaded,
        "model_path": str(service.model_path),
        "load_error": service.load_error,
        "runtime_versions": _runtime_versions(),
        "database": db_status(),
    }


@app.get("/db/health")
def db_health():
    """Database connectivity + stored-record count (best-effort)."""
    status = db_status()
    status["stored_screenings"] = count_screenings()
    return status


@app.post("/reload-model")
def reload_model():
    service.load()
    return health()


@app.get("/features")
def features():
    return {
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "metadata": FEATURE_METADATA,
    }


@app.get("/model-info")
def model_info():
    if not service.loaded:
        raise HTTPException(status_code=503, detail=service.load_error)
    return service.model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    try:
        inputs = features.model_dump()
        prediction = service.predict(inputs)
        # Best-effort persistence; never blocks or fails the response.
        save_screening(source="predict", input_features=inputs, prediction=prediction)
        return prediction
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.get("/generate-guidance/health")
def generate_guidance_health():
    return llm_health()


@app.post("/generate-guidance")
def generate_guidance(features: PatientFeatures):
    """Run ML screening, then produce safe, guideline-grounded kidney-risk education.

    Pipeline: /predict → RAG retrieval → prompt → LLM (Qwen3 or local-demo) →
    safety validator. The response carries `mode` ("production" | "local_demo"),
    a `disclaimer`, the guidance `sections`, `provenance`, and the safety `status`.
    """
    try:
        inputs = features.model_dump()
        prediction = service.predict(inputs)
        response = build_guidance_response(prediction)
        # Best-effort persistence: full record (inputs + prediction + guidance).
        save_screening(source="generate-guidance", input_features=inputs,
                       prediction=prediction, guidance_response=response)
        return response
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Guidance generation failed: {exc}")
