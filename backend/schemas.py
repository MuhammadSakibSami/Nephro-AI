from __future__ import annotations

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class PatientFeatures(BaseModel):
    """Single-patient input features. Missing values are allowed and imputed by the trained pipeline."""

    model_config = ConfigDict(extra="forbid")

    age: Optional[float] = Field(None, ge=0, le=120)
    sex_female: Optional[int] = Field(None, ge=0, le=1)
    income_poverty_ratio: Optional[float] = Field(None, ge=0, le=10)
    bmi: Optional[float] = Field(None, ge=5, le=100)
    waist: Optional[float] = Field(None, ge=20, le=250)
    mean_sbp: Optional[float] = Field(None, ge=40, le=300)
    mean_dbp: Optional[float] = Field(None, ge=20, le=200)
    mean_pulse: Optional[float] = Field(None, ge=20, le=220)
    glucose: Optional[float] = Field(None, ge=10, le=1000)
    hba1c: Optional[float] = Field(None, ge=2, le=25)
    blood_urea_nitrogen: Optional[float] = Field(None, ge=0, le=300)
    uric_acid: Optional[float] = Field(None, ge=0, le=50)
    serum_albumin: Optional[float] = Field(None, ge=0, le=10)
    sodium: Optional[float] = Field(None, ge=80, le=220)
    potassium: Optional[float] = Field(None, ge=1, le=12)
    hemoglobin: Optional[float] = Field(None, ge=1, le=30)
    diabetes_yes: Optional[int] = Field(None, ge=0, le=1)
    borderline_diabetes: Optional[int] = Field(None, ge=0, le=1)
    hypertension_doctor_told: Optional[int] = Field(None, ge=0, le=1)
    high_cholesterol_told: Optional[int] = Field(None, ge=0, le=1)
    taking_bp_medicine: Optional[int] = Field(None, ge=0, le=1)
    taking_cholesterol_medicine: Optional[int] = Field(None, ge=0, le=1)


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
