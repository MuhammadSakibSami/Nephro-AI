"""LLM Component — Training-distribution profiles (LLM_PLAN.md Phase 7).

Generates many diverse, synthetic `/predict`-shaped profiles for RAFT training
data generation. Drawn from the TRAINING distribution and kept SEPARATE from the
30 frozen held-out eval profiles (`profiles.held_out_profiles`) — the eval set is
reserved for the Phase-9 base-vs-fine-tuned ablation.

Profiles are seeded (reproducible) but randomized for variety: per profile we
sample a risk class, 1–4 clinical drivers with risk-appropriate values, and
matching binary flags. ~1/3 of profiles per class; a slice of high-risk profiles
use emergency-level vitals so the model learns to self-add emergency redirects.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from .profiles import build_predict_profile

RISKS = ["low", "moderate", "high"]

# Continuous drivers: per-risk value ranges (low/moderate/high).
# For albumin & hemoglobin a LOWER value is worse, so "high"-risk ranges are low.
_DRIVER_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "mean_sbp":            {"low": (116, 132), "moderate": (140, 158), "high": (160, 196)},
    "mean_dbp":            {"low": (70, 82),   "moderate": (88, 98),   "high": (100, 122)},
    "hba1c":               {"low": (5.1, 5.9), "moderate": (6.8, 7.9), "high": (8.2, 10.6)},
    "glucose":             {"low": (88, 108),  "moderate": (140, 178), "high": (190, 265)},
    "bmi":                 {"low": (22, 28),   "moderate": (31, 35),   "high": (35, 43)},
    "waist":               {"low": (78, 96),   "moderate": (101, 114), "high": (112, 132)},
    "blood_urea_nitrogen": {"low": (9, 16),    "moderate": (22, 28),   "high": (30, 44)},
    "uric_acid":           {"low": (4.3, 6.3), "moderate": (7.2, 8.4), "high": (8.6, 10.8)},
    "serum_albumin":       {"low": (4.0, 4.6), "moderate": (3.6, 3.9), "high": (2.8, 3.5)},
    "hemoglobin":          {"low": (13.0, 15.0), "moderate": (11.0, 12.6), "high": (8.8, 10.9)},
}
# Pool of continuous driver features to sample from.
_CONT_FEATURES = list(_DRIVER_RANGES.keys())

# Binary flags paired with related continuous drivers.
_BINARY_PAIRS = {
    "mean_sbp": "hypertension_doctor_told",
    "hba1c": "diabetes_yes",
    "glucose": "diabetes_yes",
}

_N_DRIVERS = {"low": (1, 2), "moderate": (1, 2), "high": (2, 4)}
_EMERGENCY_HIGH_FRACTION = 0.20   # share of high-risk profiles with crisis vitals


def _round(feature: str, value: float) -> float:
    return round(value, 1) if feature in {"hba1c", "uric_acid", "serum_albumin", "hemoglobin"} else round(value)


def _jittered_probs(rng: random.Random, risk: str) -> Dict[str, float]:
    if risk == "low":
        hi = rng.uniform(0.03, 0.10); mod = rng.uniform(0.08, 0.20)
        return _norm({"low": 1 - hi - mod, "moderate": mod, "high": hi})
    if risk == "moderate":
        low = rng.uniform(0.25, 0.38); hi = rng.uniform(0.15, 0.28)
        return _norm({"low": low, "moderate": 1 - low - hi, "high": hi})
    low = rng.uniform(0.05, 0.15); mod = rng.uniform(0.18, 0.30)
    return _norm({"low": low, "moderate": mod, "high": 1 - low - mod})


def _norm(p: Dict[str, float]) -> Dict[str, float]:
    s = sum(p.values())
    return {k: round(v / s, 3) for k, v in p.items()}


def _make_one(rng: random.Random, pid: str, risk: str) -> Dict[str, Any]:
    n = rng.randint(*_N_DRIVERS[risk])
    feats = rng.sample(_CONT_FEATURES, n)

    # Emergency override for a slice of high-risk profiles.
    emergency = risk == "high" and rng.random() < _EMERGENCY_HIGH_FRACTION
    if emergency and "mean_sbp" not in feats:
        feats[0] = "mean_sbp"

    drivers: List[Tuple[str, float]] = []
    over: Dict[str, Any] = {"age": rng.randint(30, 75), "sex_female": rng.randint(0, 1)}
    for f in feats:
        lo, hi = _DRIVER_RANGES[f][risk]
        val = _round(f, rng.uniform(lo, hi))
        if emergency and f == "mean_sbp":
            val = float(rng.randint(182, 205))
        drivers.append((f, val))
        flag = _BINARY_PAIRS.get(f)
        if flag:
            over[flag] = 1
    if emergency:
        over["mean_dbp"] = float(rng.randint(116, 126))

    return build_predict_profile(pid, risk, drivers, over, probs=_jittered_probs(rng, risk))


def generate_train_profiles(n: int = 600, seed: int = 7) -> List[Dict[str, Any]]:
    """Return `n` synthetic training profiles, ~evenly split across risk classes."""
    rng = random.Random(seed)
    per = n // 3
    counts = {"low": per, "moderate": per, "high": n - 2 * per}
    profiles: List[Dict[str, Any]] = []
    for risk in RISKS:
        for i in range(counts[risk]):
            pid = f"T_{risk[:1].upper()}{i:04d}"
            profiles.append(_make_one(rng, pid, risk))
    rng.shuffle(profiles)
    return profiles


if __name__ == "__main__":
    from collections import Counter
    profs = generate_train_profiles(60, seed=1)
    print("n:", len(profs), "dist:", dict(Counter(p["risk_class"] for p in profs)))
    n_emerg = sum(1 for p in profs if (p["llm_context"]["patient_inputs"].get("mean_sbp") or 0) >= 180)
    print("emergency-BP profiles:", n_emerg)
    print("sample drivers:", [f["feature"] for f in profs[0]["llm_context"]["key_model_factors"]])
