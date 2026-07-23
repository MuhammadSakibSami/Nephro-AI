"""Feature schema and UI metadata for the CKD risk screening app.

Aligned to the deployed **Final Model C** (`FINAL_MODEL_C_FULL_DEVELOPMENT.joblib`).
Model C consumes exactly 14 patient inputs (13-feature Stage-1 Random Forest +
direct-six Stage-2 Gradient Boosting; 14 unique inputs across both stages). Every
feature listed here is scored by the model — there are no unused/context-only
fields.
"""
from __future__ import annotations

LABEL_ORDER = ["low", "moderate", "high"]

# The 14 unique patient inputs consumed by Final Model C.
# (Stage-1 uses 13 of these; Stage-2 uses 6; the union is 14.)
MODEL_FEATURES = [
    "age", "income_poverty_ratio",
    "waist", "mean_sbp", "mean_dbp",
    "glucose", "hba1c", "blood_urea_nitrogen", "uric_acid",
    "serum_albumin", "hemoglobin",
    "diabetes_yes", "hypertension_doctor_told", "taking_bp_medicine",
]

NUMERIC_FEATURES = [
    "age", "income_poverty_ratio", "waist", "mean_sbp", "mean_dbp",
    "glucose", "hba1c", "blood_urea_nitrogen", "uric_acid",
    "serum_albumin", "hemoglobin",
]

BINARY_FEATURES = [
    "diabetes_yes", "hypertension_doctor_told", "taking_bp_medicine",
]

# Stage membership (informational; the model package is the source of truth).
STAGE1_FEATURES = [
    "age", "blood_urea_nitrogen", "hba1c", "taking_bp_medicine", "mean_sbp",
    "glucose", "hypertension_doctor_told", "uric_acid", "diabetes_yes",
    "income_poverty_ratio", "mean_dbp", "hemoglobin", "waist",
]
STAGE2_FEATURES = [
    "blood_urea_nitrogen", "uric_acid", "hemoglobin", "serum_albumin",
    "mean_sbp", "age",
]

FEATURE_METADATA = {
    "age": {"label": "Age", "unit": "years", "help": "Participant age."},
    "income_poverty_ratio": {"label": "Income-to-poverty ratio", "unit": "ratio", "help": "NHANES family income-to-poverty ratio. Optional if unavailable."},
    "waist": {"label": "Waist circumference", "unit": "cm", "help": "Waist circumference."},
    "mean_sbp": {"label": "Mean systolic BP", "unit": "mmHg", "help": "Average systolic blood pressure."},
    "mean_dbp": {"label": "Mean diastolic BP", "unit": "mmHg", "help": "Average diastolic blood pressure."},
    "glucose": {"label": "Glucose", "unit": "mg/dL", "help": "Serum/plasma glucose as prepared in the dataset."},
    "hba1c": {"label": "HbA1c", "unit": "%", "help": "Glycohemoglobin."},
    "blood_urea_nitrogen": {"label": "Blood urea nitrogen", "unit": "mg/dL", "help": "BUN."},
    "uric_acid": {"label": "Uric acid", "unit": "mg/dL", "help": "Serum uric acid."},
    "serum_albumin": {"label": "Serum albumin", "unit": "g/dL", "help": "Serum albumin."},
    "hemoglobin": {"label": "Hemoglobin", "unit": "g/dL", "help": "Blood hemoglobin."},
    "diabetes_yes": {"label": "Doctor told diabetes", "unit": "", "help": "Yes=1, No=0."},
    "hypertension_doctor_told": {"label": "Doctor told hypertension", "unit": "", "help": "Doctor/health professional told high blood pressure."},
    "taking_bp_medicine": {"label": "Taking BP medicine", "unit": "", "help": "Blood pressure medicine use."},
}

NUMERIC_DEFAULTS = {
    "age": 55.0,
    "income_poverty_ratio": 2.0,
    "waist": 98.0,
    "mean_sbp": 125.0,
    "mean_dbp": 78.0,
    "glucose": 100.0,
    "hba1c": 5.6,
    "blood_urea_nitrogen": 14.0,
    "uric_acid": 5.5,
    "serum_albumin": 4.2,
    "hemoglobin": 14.0,
}

NUMERIC_RANGES = {
    "age": (18.0, 85.0),
    "income_poverty_ratio": (0.0, 5.0),
    "waist": (40.0, 180.0),
    "mean_sbp": (70.0, 240.0),
    "mean_dbp": (35.0, 140.0),
    "glucose": (40.0, 500.0),
    "hba1c": (3.5, 15.0),
    "blood_urea_nitrogen": (2.0, 120.0),
    "uric_acid": (1.0, 20.0),
    "serum_albumin": (1.0, 6.0),
    "hemoglobin": (5.0, 22.0),
}
