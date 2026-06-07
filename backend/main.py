from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .feature_config import MODEL_FEATURES, FEATURE_METADATA, NUMERIC_FEATURES, BINARY_FEATURES
from .schemas import PatientFeatures, PredictionResponse
from .model_service import service, ModelNotLoadedError


def _runtime_versions():
    versions = {}
    for mod_name in ["numpy", "pandas", "sklearn", "imblearn", "joblib"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[mod_name] = f"not available: {exc}"
    return versions

app = FastAPI(
    title="CKD Risk Screening API",
    description="FastAPI backend for the final NHANES two-stage kidney-risk screening cascade.",
    version="1.0.0",
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
    }


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
        return service.predict(features.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
