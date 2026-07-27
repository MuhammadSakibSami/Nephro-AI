"""LLM Component — Phase 9 fresh held-out evaluation profiles.

Unlike the Phase 6 set (`llm/data/profiles.py`), these profiles are **not**
hand-labeled. `risk_class`, `probabilities`, `confidence`, and
`key_model_factors` all come from actually calling the trained ML model
(`KidneyRiskModelService.predict`) on a sampled synthetic patient input, so
the LLM ablation is scored against genuine model output rather than a
fabricated stand-in.

A fresh, disjoint set from both the Phase 6 eval profiles (30, hand-authored)
and the Phase 7 training profiles (600, risk-conditioned sampling) — a
different seed and a from-real-output bucketing strategy (sample broadly,
classify by the model's actual prediction) rather than assigning a risk
label up front and reverse-engineering inputs to match it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

try:
    from ...backend.feature_config import NUMERIC_DEFAULTS, BINARY_FEATURES
    from ...backend.model_service import KidneyRiskModelService
except ImportError:  # pragma: no cover - repo-root on sys.path
    from backend.feature_config import NUMERIC_DEFAULTS, BINARY_FEATURES
    from backend.model_service import KidneyRiskModelService

PHASE9_PROFILES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "eval_profiles_phase9.jsonl"
)

_BASELINE: Dict[str, Any] = {**NUMERIC_DEFAULTS, **{f: 0 for f in BINARY_FEATURES}}

# Broad per-feature sampling ranges spanning low -> high risk (a superset of
# train_profiles.py's per-risk-class bands) so real /predict output covers
# the full risk spectrum instead of only what we'd expect it to be.
_RANGES: Dict[str, tuple] = {
    "waist":                (75.0, 134.0),
    "mean_sbp":              (108.0, 205.0),
    "mean_dbp":              (66.0, 126.0),
    "glucose":               (85.0, 270.0),
    "hba1c":                 (5.0, 10.8),
    "blood_urea_nitrogen":   (8.0, 45.0),
    "uric_acid":             (4.0, 11.0),
    "serum_albumin":         (2.7, 4.7),
    "hemoglobin":            (8.5, 15.2),
}
_AGE_RANGE = (28.0, 78.0)

# feature -> (correlated binary flag, threshold beyond which the flag turns on)
_BINARY_CORRELATES = {
    "mean_sbp": ("hypertension_doctor_told", 140.0),
    "hba1c": ("diabetes_yes", 7.0),
    "glucose": ("diabetes_yes", 150.0),
}


def _round(feature: str, value: float) -> float:
    return round(value, 1) if feature in {"hba1c", "uric_acid", "serum_albumin", "hemoglobin"} else round(value)


def _sample_payload(rng: random.Random, force_emergency: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(_BASELINE)
    payload["age"] = round(rng.uniform(*_AGE_RANGE))

    n_active = rng.randint(2, 6)
    active = rng.sample(list(_RANGES.keys()), n_active)
    for feat in active:
        lo, hi = _RANGES[feat]
        payload[feat] = _round(feat, rng.uniform(lo, hi))

    if force_emergency:
        payload["mean_sbp"] = _round("mean_sbp", rng.uniform(182.0, 202.0))
        payload["mean_dbp"] = _round("mean_dbp", rng.uniform(112.0, 124.0))

    for feat, (flag, thr) in _BINARY_CORRELATES.items():
        if payload.get(feat, 0) >= thr:
            payload[flag] = 1
    if payload.get("hypertension_doctor_told") and rng.random() < 0.6:
        payload["taking_bp_medicine"] = 1

    return payload


def generate_phase9_profiles(n_per_class: int = 10, seed: int = 2099,
                              n_emergency: int = 2, max_attempts: int = 4000
                              ) -> List[Dict[str, Any]]:
    """Sample broadly, score with the REAL model, bucket by its actual output
    until each risk class has `n_per_class` profiles (rejection sampling)."""
    rng = random.Random(seed)
    svc = KidneyRiskModelService()
    quotas = {"low": n_per_class, "moderate": n_per_class, "high": n_per_class}
    buckets: Dict[str, List[Dict[str, Any]]] = {"low": [], "moderate": [], "high": []}
    emergency_found = 0

    for _ in range(max_attempts):
        if all(len(buckets[r]) >= quotas[r] for r in quotas):
            break
        force_em = emergency_found < n_emergency and rng.random() < 0.15
        payload = _sample_payload(rng, force_emergency=force_em)
        try:
            resp = svc.predict(payload)
        except Exception:
            continue
        risk = resp["risk_class"]
        if len(buckets[risk]) >= quotas[risk]:
            continue
        is_emergency = payload.get("mean_sbp", 0) >= 180 or payload.get("mean_dbp", 0) >= 120
        if is_emergency:
            emergency_found += 1
        resp["profile_id"] = f"P9_{risk[:1].upper()}{len(buckets[risk]) + 1:02d}"
        buckets[risk].append(resp)
    else:
        missing = {r: quotas[r] - len(buckets[r]) for r in quotas if len(buckets[r]) < quotas[r]}
        if missing:
            raise RuntimeError(f"Exhausted {max_attempts} sampling attempts; still short: {missing}")

    profiles = buckets["low"] + buckets["moderate"] + buckets["high"]
    rng.shuffle(profiles)
    return profiles


def save_phase9_profiles(path: Path = PHASE9_PROFILES_PATH, **kwargs) -> Path:
    profiles = generate_phase9_profiles(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def load_phase9_profiles(path: Path = PHASE9_PROFILES_PATH) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    from collections import Counter

    p = save_phase9_profiles()
    profs = load_phase9_profiles(p)
    print(f"Wrote {len(profs)} profiles -> {p}")
    print("Distribution:", dict(Counter(x["risk_class"] for x in profs)))
    n_emerg = sum(
        1 for x in profs
        if x["llm_context"]["patient_inputs"].get("mean_sbp", 0) >= 180
        or x["llm_context"]["patient_inputs"].get("mean_dbp", 0) >= 120
    )
    print("emergency-BP profiles:", n_emerg)
