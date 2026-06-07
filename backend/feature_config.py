"""Feature schema and UI metadata for the CKD risk screening app."""
from __future__ import annotations

LABEL_ORDER = ["low", "moderate", "high"]

MODEL_FEATURES = [
    "age", "sex_female", "income_poverty_ratio",
    "bmi", "waist", "mean_sbp", "mean_dbp", "mean_pulse",
    "glucose", "hba1c", "blood_urea_nitrogen", "uric_acid",
    "serum_albumin", "sodium", "potassium", "hemoglobin",
    "diabetes_yes", "borderline_diabetes", "hypertension_doctor_told",
    "high_cholesterol_told", "taking_bp_medicine", "taking_cholesterol_medicine",
]

NUMERIC_FEATURES = [
    "age", "income_poverty_ratio", "bmi", "waist", "mean_sbp", "mean_dbp", "mean_pulse",
    "glucose", "hba1c", "blood_urea_nitrogen", "uric_acid", "serum_albumin",
    "sodium", "potassium", "hemoglobin",
]

BINARY_FEATURES = [
    "sex_female", "diabetes_yes", "borderline_diabetes", "hypertension_doctor_told",
    "high_cholesterol_told", "taking_bp_medicine", "taking_cholesterol_medicine",
]

FEATURE_METADATA = {
    "age": {"label": "Age", "unit": "years", "help": "Participant age."},
    "sex_female": {"label": "Sex", "unit": "", "help": "Female=1, Male=0."},
    "income_poverty_ratio": {"label": "Income-to-poverty ratio", "unit": "ratio", "help": "NHANES family income-to-poverty ratio. Optional if unavailable."},
    "bmi": {"label": "Body mass index", "unit": "kg/m²", "help": "BMI."},
    "waist": {"label": "Waist circumference", "unit": "cm", "help": "Waist circumference."},
    "mean_sbp": {"label": "Mean systolic BP", "unit": "mmHg", "help": "Average systolic blood pressure."},
    "mean_dbp": {"label": "Mean diastolic BP", "unit": "mmHg", "help": "Average diastolic blood pressure."},
    "mean_pulse": {"label": "Mean pulse", "unit": "beats/min", "help": "Average pulse rate."},
    "glucose": {"label": "Glucose", "unit": "mg/dL", "help": "Serum/plasma glucose as prepared in the dataset."},
    "hba1c": {"label": "HbA1c", "unit": "%", "help": "Glycohemoglobin."},
    "blood_urea_nitrogen": {"label": "Blood urea nitrogen", "unit": "mg/dL", "help": "BUN."},
    "uric_acid": {"label": "Uric acid", "unit": "mg/dL", "help": "Serum uric acid."},
    "serum_albumin": {"label": "Serum albumin", "unit": "g/dL", "help": "Serum albumin."},
    "sodium": {"label": "Sodium", "unit": "mmol/L", "help": "Serum sodium."},
    "potassium": {"label": "Potassium", "unit": "mmol/L", "help": "Serum potassium."},
    "hemoglobin": {"label": "Hemoglobin", "unit": "g/dL", "help": "Blood hemoglobin."},
    "diabetes_yes": {"label": "Doctor told diabetes", "unit": "", "help": "Yes=1, No/borderline/no diabetes=0."},
    "borderline_diabetes": {"label": "Borderline diabetes", "unit": "", "help": "Borderline diabetes/prediabetes indicator."},
    "hypertension_doctor_told": {"label": "Doctor told hypertension", "unit": "", "help": "Doctor/health professional told high blood pressure."},
    "high_cholesterol_told": {"label": "Doctor told high cholesterol", "unit": "", "help": "Doctor/health professional told high cholesterol."},
    "taking_bp_medicine": {"label": "Taking BP medicine", "unit": "", "help": "Blood pressure medicine use."},
    "taking_cholesterol_medicine": {"label": "Taking cholesterol medicine", "unit": "", "help": "Cholesterol medicine use."},
}

NUMERIC_DEFAULTS = {
    "age": 55.0,
    "income_poverty_ratio": 2.0,
    "bmi": 28.0,
    "waist": 98.0,
    "mean_sbp": 125.0,
    "mean_dbp": 78.0,
    "mean_pulse": 72.0,
    "glucose": 100.0,
    "hba1c": 5.6,
    "blood_urea_nitrogen": 14.0,
    "uric_acid": 5.5,
    "serum_albumin": 4.2,
    "sodium": 140.0,
    "potassium": 4.2,
    "hemoglobin": 14.0,
}

NUMERIC_RANGES = {
    "age": (18.0, 85.0),
    "income_poverty_ratio": (0.0, 5.0),
    "bmi": (12.0, 70.0),
    "waist": (40.0, 180.0),
    "mean_sbp": (70.0, 240.0),
    "mean_dbp": (35.0, 140.0),
    "mean_pulse": (35.0, 160.0),
    "glucose": (40.0, 500.0),
    "hba1c": (3.5, 15.0),
    "blood_urea_nitrogen": (2.0, 120.0),
    "uric_acid": (1.0, 20.0),
    "serum_albumin": (1.0, 6.0),
    "sodium": (115.0, 170.0),
    "potassium": (2.0, 8.0),
    "hemoglobin": (5.0, 22.0),
}
