"""LLM Component — Held-out Evaluation Profiles (LLM_PLAN.md Phase 6).

30 synthetic, held-out `/predict`-shaped profiles (10 low / 10 moderate / 10
high) with mixed clinical drivers (blood pressure, glucose/diabetes, waist,
kidney labs, uric acid). These are the LLM-evaluation profiles — they are NOT
used for RAFT training (Phase 7), keeping the LLM eval clean.

Aligned to the deployed Final Model C: profiles use only the model's 14 input
features (see `backend.feature_config.MODEL_FEATURES`). Every driver/override is
one of those 14, so the LLM conditions on exactly what the deployed model scores.

Each profile is a `/predict` response in the exact shape produced by the backend
(`model_service`), so `LLMInput.from_predict_response` consumes it directly.

Notes:
  * Profiles are deterministic (fixed values) for reproducibility.
  * "smoking" and "activity" are KB/education themes, not ML model features, so
    they are not encoded as model factors here; the LLM may still address them.
  * Two high-risk profiles use emergency-level BP (SBP >= 180) to exercise the
    validator's emergency-redirect path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..serving.contract import RISK_CLASSES  # noqa: F401  (kept for clarity)

try:
    from ...backend.feature_config import FEATURE_METADATA
except Exception:  # pragma: no cover - relative import fails outside a parent package
    try:
        from backend.feature_config import FEATURE_METADATA  # repo-root on sys.path
    except Exception:
        FEATURE_METADATA = {}

EVAL_PROFILES_PATH = Path(__file__).resolve().parent / "eval_profiles.jsonl"

_PROBS = {
    "low": {"low": 0.80, "moderate": 0.14, "high": 0.06},
    "moderate": {"low": 0.30, "moderate": 0.48, "high": 0.22},
    "high": {"low": 0.08, "moderate": 0.25, "high": 0.67},
}

_BASELINE = {
    "age": 50, "income_poverty_ratio": 2.5, "waist": 92.0,
    "mean_sbp": 122.0, "mean_dbp": 78.0, "glucose": 95.0, "hba1c": 5.4,
    "blood_urea_nitrogen": 13.0, "uric_acid": 5.2, "serum_albumin": 4.3, "hemoglobin": 14.0,
    "diabetes_yes": 0, "hypertension_doctor_told": 0, "taking_bp_medicine": 0,
}

# (profile_id, risk, drivers[ordered: (feature, value)], overrides)
# Drivers/overrides use only the 14 Final Model C inputs (waist replaces the old
# BMI driver; no sex/pulse/sodium/potassium/cholesterol/borderline).
_SPECS: List[Tuple[str, str, List[Tuple[str, float]], Dict[str, Any]]] = [
    # ----- LOW (10) -----
    ("L01", "low", [("waist", 96.0)], {"age": 38}),
    ("L02", "low", [("mean_sbp", 128.0)], {"age": 44}),
    ("L03", "low", [("hba1c", 5.8)], {"age": 41, "glucose": 104}),
    ("L04", "low", [("waist", 97.0), ("uric_acid", 6.2)], {"age": 47}),
    ("L05", "low", [("age", 63.0)], {"glucose": 100}),
    ("L06", "low", [("uric_acid", 6.3)], {"age": 36}),
    ("L07", "low", [("waist", 90.0)], {"age": 29}),
    ("L08", "low", [("glucose", 103.0)], {"age": 52}),
    ("L09", "low", [("mean_sbp", 126.0), ("waist", 95.0)], {"age": 49}),
    ("L10", "low", [("hemoglobin", 13.1)], {"age": 33}),
    # ----- MODERATE (10) -----
    ("M01", "moderate", [("mean_sbp", 150.0), ("hypertension_doctor_told", 1)], {"age": 57}),
    ("M02", "moderate", [("hba1c", 7.6), ("diabetes_yes", 1)], {"age": 60, "glucose": 168}),
    ("M03", "moderate", [("waist", 110.0), ("mean_sbp", 142.0)], {"age": 45}),
    ("M04", "moderate", [("mean_sbp", 148.0), ("waist", 106.0)], {"age": 54}),
    ("M05", "moderate", [("glucose", 162.0), ("hba1c", 7.2)], {"age": 58, "diabetes_yes": 1}),
    ("M06", "moderate", [("blood_urea_nitrogen", 24.0), ("uric_acid", 7.6)], {"age": 61}),
    ("M07", "moderate", [("mean_sbp", 152.0), ("mean_dbp", 96.0)], {"age": 50}),
    ("M08", "moderate", [("uric_acid", 8.2), ("waist", 104.0)], {"age": 48}),
    ("M09", "moderate", [("mean_sbp", 146.0), ("mean_dbp", 90.0)], {"age": 55, "hypertension_doctor_told": 1, "taking_bp_medicine": 1}),
    ("M10", "moderate", [("hba1c", 7.8), ("waist", 108.0), ("mean_sbp", 144.0)], {"age": 59, "diabetes_yes": 1}),
    # ----- HIGH (10) -----
    ("H01", "high", [("mean_sbp", 168.0), ("hba1c", 8.6), ("diabetes_yes", 1)], {"age": 64}),
    ("H02", "high", [("mean_sbp", 185.0), ("hypertension_doctor_told", 1)], {"age": 67}),  # emergency
    ("H03", "high", [("hba1c", 9.2), ("blood_urea_nitrogen", 30.0), ("diabetes_yes", 1)], {"age": 62, "glucose": 215}),
    ("H04", "high", [("mean_sbp", 172.0), ("uric_acid", 9.0), ("serum_albumin", 3.4)], {"age": 65}),
    ("H05", "high", [("blood_urea_nitrogen", 34.0), ("serum_albumin", 3.2), ("hemoglobin", 10.5)], {"age": 70}),
    ("H06", "high", [("hba1c", 8.8), ("uric_acid", 8.6), ("waist", 116.0)], {"age": 56, "diabetes_yes": 1}),
    ("H07", "high", [("mean_sbp", 190.0), ("mean_dbp", 118.0)], {"age": 68}),  # emergency
    ("H08", "high", [("mean_sbp", 166.0), ("waist", 118.0), ("hba1c", 8.2), ("diabetes_yes", 1)], {"age": 60}),
    ("H09", "high", [("blood_urea_nitrogen", 38.0), ("serum_albumin", 3.0)], {"age": 72}),
    ("H10", "high", [("hba1c", 9.5), ("mean_sbp", 170.0), ("blood_urea_nitrogen", 28.0), ("diabetes_yes", 1)], {"age": 63}),
]


def make_factor(feature: str, value: float, contrib: float = 0.1) -> Dict[str, Any]:
    meta = FEATURE_METADATA.get(feature, {})
    return {
        "feature": feature,
        "label": meta.get("label", feature),
        "unit": meta.get("unit", ""),
        "current_value": float(value),
        "reference_value": None,
        "direction": "supports_prediction",
        "absolute_contribution": contrib,
    }


_factor = make_factor  # backward-compatible alias


def build_predict_profile(pid: str, risk: str, drivers: List[Tuple[str, float]],
                          over: Dict[str, Any] = None,
                          probs: Dict[str, float] = None) -> Dict[str, Any]:
    """Build one `/predict`-shaped profile. Reused by held-out + training profiles."""
    over = over or {}
    inputs = {**_BASELINE, **{f: v for f, v in drivers}, **over}
    factors = [make_factor(f, v, round(0.16 - 0.025 * i, 3)) for i, (f, v) in enumerate(drivers)]
    probs = probs or _PROBS[risk]
    conf = max(probs.values())
    n_features = len(inputs)
    return {
        "profile_id": pid,
        "risk_class": risk,
        "risk_class_display": f"{risk.capitalize()} risk",
        "probabilities": probs,
        "confidence": conf,
        "stage1": {"threshold": 0.44}, "stage2": {"threshold": 0.50},
        "model_info": {"stage1_threshold": 0.44, "stage2_threshold": 0.50},
        "llm_context": {
            "model_output": {
                "screening_category": risk, "class_probabilities": probs,
                "confidence": conf, "reliability_level": "moderate", "reliability_notes": [],
            },
            "key_model_factors": factors,
            "input_quality": {
                "n_model_features": n_features, "n_supplied_features": n_features, "n_missing_features": 0,
                "missing_features": [], "feature_labels": {},
            },
            "patient_inputs": inputs,
            "strict_safety_rules": [], "suggestion_style": {},
        },
    }


def held_out_profiles() -> List[Dict[str, Any]]:
    """Return the 30 held-out /predict-shaped evaluation profiles."""
    return [build_predict_profile(pid, risk, drivers, over) for pid, risk, drivers, over in _SPECS]


def save_profiles(path: Path = EVAL_PROFILES_PATH) -> Path:
    profiles = held_out_profiles()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    p = save_profiles()
    profs = held_out_profiles()
    from collections import Counter
    print(f"Wrote {len(profs)} profiles -> {p}")
    print("Distribution:", dict(Counter(x["risk_class"] for x in profs)))
