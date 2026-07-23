from __future__ import annotations

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class PatientFeatures(BaseModel):
    """Single-patient input features for Final Model C (14 inputs).

    Missing values are allowed and imputed by the trained pipeline. Every field
    here is consumed by the model — the deployed Model C uses exactly these 14
    inputs (13-feature Stage-1 + direct-six Stage-2, 14 unique).
    """

    model_config = ConfigDict(extra="forbid")

    age: Optional[float] = Field(None, ge=0, le=120)
    income_poverty_ratio: Optional[float] = Field(None, ge=0, le=10)
    waist: Optional[float] = Field(None, ge=20, le=250)
    mean_sbp: Optional[float] = Field(None, ge=40, le=300)
    mean_dbp: Optional[float] = Field(None, ge=20, le=200)
    glucose: Optional[float] = Field(None, ge=10, le=1000)
    hba1c: Optional[float] = Field(None, ge=2, le=25)
    blood_urea_nitrogen: Optional[float] = Field(None, ge=0, le=300)
    uric_acid: Optional[float] = Field(None, ge=0, le=50)
    serum_albumin: Optional[float] = Field(None, ge=0, le=10)
    hemoglobin: Optional[float] = Field(None, ge=1, le=30)
    diabetes_yes: Optional[int] = Field(None, ge=0, le=1)
    hypertension_doctor_told: Optional[int] = Field(None, ge=0, le=1)
    taking_bp_medicine: Optional[int] = Field(None, ge=0, le=1)


class PredictionResponse(BaseModel):
    risk_class: str
    risk_class_display: str
    probabilities: Dict[str, float]
    confidence: float
    stage1: Dict[str, Any]
    stage2: Dict[str, Any]
    input_quality: Dict[str, Any]
    model_info: Dict[str, Any]
    reliability: Dict[str, Any] = {}
    explanation: Dict[str, Any] = {}
    llm_context: Dict[str, Any] = {}
    clinical_note: str
