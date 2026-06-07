from __future__ import annotations

import math
from html import escape
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Kidney Risk Screening Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://127.0.0.1:8000"

CLASS_ORDER = ["low", "moderate", "high"]
CLASS_LABEL = {"low": "Low", "moderate": "Moderate", "high": "High"}

CLASS_META = {
    "low": {
        "title": "Low screening category",
        "badge": "Low",
        "summary": "The current inputs are closest to the low kidney-risk screening pattern learned by the model.",
        "css": "low",
    },
    "moderate": {
        "title": "Moderate screening category",
        "badge": "Moderate",
        "summary": "The current inputs suggest an at-risk pattern, but not a strong high-risk pattern by the cascade model.",
        "css": "moderate",
    },
    "high": {
        "title": "High screening category",
        "badge": "High",
        "summary": "The current inputs are closest to a higher kidney-risk screening pattern learned by the model.",
        "css": "high",
    },
}

FEATURES = [
    "age",
    "sex_female",
    "income_poverty_ratio",
    "bmi",
    "waist",
    "mean_sbp",
    "mean_dbp",
    "mean_pulse",
    "glucose",
    "hba1c",
    "blood_urea_nitrogen",
    "uric_acid",
    "serum_albumin",
    "sodium",
    "potassium",
    "hemoglobin",
    "diabetes_yes",
    "borderline_diabetes",
    "hypertension_doctor_told",
    "high_cholesterol_told",
    "taking_bp_medicine",
    "taking_cholesterol_medicine",
]

FEATURE_META: Dict[str, Dict[str, Any]] = {
    "age": {"label": "Age", "unit": "years", "default": 55.0, "min": 18.0, "max": 85.0, "step": 1.0, "group": "profile"},
    "sex_female": {"label": "Sex", "unit": "", "group": "profile", "type": "sex"},
    "income_poverty_ratio": {"label": "Income-to-poverty ratio", "unit": "ratio", "default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1, "group": "profile"},
    "bmi": {"label": "Body mass index", "unit": "kg/m²", "default": 28.0, "min": 12.0, "max": 70.0, "step": 0.1, "group": "profile"},
    "waist": {"label": "Waist circumference", "unit": "cm", "default": 98.0, "min": 40.0, "max": 180.0, "step": 0.1, "group": "profile"},
    "mean_sbp": {"label": "Mean systolic BP", "unit": "mmHg", "default": 125.0, "min": 70.0, "max": 240.0, "step": 1.0, "group": "bp"},
    "mean_dbp": {"label": "Mean diastolic BP", "unit": "mmHg", "default": 78.0, "min": 35.0, "max": 140.0, "step": 1.0, "group": "bp"},
    "mean_pulse": {"label": "Mean pulse", "unit": "beats/min", "default": 72.0, "min": 35.0, "max": 160.0, "step": 1.0, "group": "bp"},
    "glucose": {"label": "Glucose", "unit": "mg/dL", "default": 100.0, "min": 40.0, "max": 500.0, "step": 1.0, "group": "labs"},
    "hba1c": {"label": "HbA1c", "unit": "%", "default": 5.6, "min": 3.5, "max": 15.0, "step": 0.1, "group": "labs"},
    "blood_urea_nitrogen": {"label": "Blood urea nitrogen", "unit": "mg/dL", "default": 14.0, "min": 2.0, "max": 120.0, "step": 0.1, "group": "labs"},
    "uric_acid": {"label": "Uric acid", "unit": "mg/dL", "default": 5.5, "min": 1.0, "max": 20.0, "step": 0.1, "group": "labs"},
    "serum_albumin": {"label": "Serum albumin", "unit": "g/dL", "default": 4.2, "min": 1.0, "max": 6.0, "step": 0.1, "group": "labs"},
    "sodium": {"label": "Sodium", "unit": "mmol/L", "default": 140.0, "min": 115.0, "max": 170.0, "step": 0.1, "group": "labs"},
    "potassium": {"label": "Potassium", "unit": "mmol/L", "default": 4.2, "min": 2.0, "max": 8.0, "step": 0.1, "group": "labs"},
    "hemoglobin": {"label": "Hemoglobin", "unit": "g/dL", "default": 14.0, "min": 5.0, "max": 22.0, "step": 0.1, "group": "labs"},
    "diabetes_yes": {"label": "Doctor told diabetes", "unit": "", "group": "history", "type": "binary"},
    "borderline_diabetes": {"label": "Borderline diabetes", "unit": "", "group": "history", "type": "binary"},
    "hypertension_doctor_told": {"label": "Doctor told hypertension", "unit": "", "group": "history", "type": "binary"},
    "high_cholesterol_told": {"label": "Doctor told high cholesterol", "unit": "", "group": "history", "type": "binary"},
    "taking_bp_medicine": {"label": "Taking BP medicine", "unit": "", "group": "history", "type": "binary"},
    "taking_cholesterol_medicine": {"label": "Taking cholesterol medicine", "unit": "", "group": "history", "type": "binary"},
}

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] input {
        color: #0f172a !important;
        background: #ffffff !important;
    }

    .main-wrap {
        max-width: 1240px;
        margin: 0 auto;
        padding-bottom: 40px;
    }

    .hero {
        background:
            radial-gradient(circle at 12% 20%, rgba(255,255,255,.20), transparent 28%),
            linear-gradient(135deg, #0f766e 0%, #164e63 55%, #1e3a5f 100%);
        border-radius: 30px;
        padding: 38px 42px;
        color: #ffffff;
        box-shadow: 0 24px 60px rgba(15, 23, 42, .20);
        margin-bottom: 22px;
    }
    .hero .tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 13px;
        border-radius: 999px;
        background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.20);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 15px;
    }
    .hero h1 {
        margin: 0 0 12px 0;
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: -.045em;
    }
    .hero p {
        color: rgba(255,255,255,.88);
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 850px;
        margin: 0;
    }

    .soft-warning {
        display: flex;
        gap: 12px;
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #78350f;
        padding: 16px 18px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(120,53,15,.05);
    }

    .glass-card {
        background: rgba(255,255,255,.94);
        border: 1px solid #e5e7eb;
        border-radius: 26px;
        padding: 24px;
        box-shadow: 0 16px 42px rgba(15,23,42,.08);
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 900;
        color: #111827;
        letter-spacing: -.025em;
        margin: 0 0 6px 0;
    }
    .section-subtitle {
        color: #64748b;
        font-size: .96rem;
        margin-bottom: 16px;
    }

    .result-card {
        position: relative;
        overflow: hidden;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 30px;
        padding: 30px 32px;
        box-shadow: 0 24px 62px rgba(15,23,42,.12);
        margin-top: 12px;
    }
    .result-card:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 9px;
        height: 100%;
        background: #64748b;
    }
    .result-card.low:before { background: #15803d; }
    .result-card.moderate:before { background: #b45309; }
    .result-card.high:before { background: #b91c1c; }

    .result-kicker {
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .10em;
        color: #64748b;
        font-weight: 900;
        margin-bottom: 12px;
    }
    .result-title {
        font-size: 2.15rem;
        line-height: 1.15;
        font-weight: 900;
        letter-spacing: -.04em;
        color: #111827;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-flex;
        border-radius: 999px;
        padding: 8px 14px;
        font-size: 1rem;
        font-weight: 900;
        margin-left: 10px;
        vertical-align: middle;
    }
    .badge.low { color: #166534; background: #dcfce7; }
    .badge.moderate { color: #92400e; background: #fef3c7; }
    .badge.high { color: #991b1b; background: #fee2e2; }
    .clinical-note {
        border-top: 1px solid #e5e7eb;
        margin-top: 22px;
        padding-top: 18px;
        color: #64748b;
        font-size: .92rem;
        line-height: 1.65;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 16px 38px rgba(15,23,42,.08);
        min-height: 136px;
    }
    .metric-label {
        color: #64748b;
        font-size: .88rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .metric-value {
        color: #111827;
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -.045em;
    }
    .metric-foot {
        border-top: 1px solid #e5e7eb;
        padding-top: 12px;
        margin-top: 18px;
        color: #64748b;
        font-size: .86rem;
    }

    .prob-row {
        display: grid;
        grid-template-columns: 110px 1fr 72px;
        align-items: center;
        gap: 13px;
        margin: 13px 0;
    }
    .prob-label {
        color: #111827;
        font-weight: 900;
    }
    .prob-track {
        height: 14px;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
    }
    .prob-fill.low { background: linear-gradient(90deg, #86efac, #15803d); }
    .prob-fill.moderate { background: linear-gradient(90deg, #fbbf24, #b45309); }
    .prob-fill.high { background: linear-gradient(90deg, #f87171, #b91c1c); }
    .prob-value {
        color: #111827;
        font-weight: 900;
        text-align: right;
    }

    .factor-row {
        display: grid;
        grid-template-columns: 1fr 98px;
        gap: 12px;
        padding: 13px 0;
        border-bottom: 1px solid #e5e7eb;
    }
    .factor-row:last-child { border-bottom: none; }
    .factor-name {
        color: #111827;
        font-weight: 850;
    }
    .factor-desc {
        color: #64748b;
        font-size: .86rem;
        margin-top: 3px;
    }
    .factor-chip {
        justify-self: end;
        align-self: start;
        border-radius: 999px;
        padding: 7px 10px;
        font-size: .78rem;
        font-weight: 900;
        background: #eef2ff;
        color: #3730a3;
    }
    .factor-chip.negative {
        background: #f1f5f9;
        color: #475569;
    }

    .llm-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 22px;
        padding: 18px;
        color: #334155;
        font-size: .95rem;
        line-height: 1.65;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: .84rem;
        font-weight: 900;
    }
    .status-ok { color: #bbf7d0; background: rgba(34,197,94,.14); border: 1px solid rgba(187,247,208,.25); }
    .status-warn { color: #fde68a; background: rgba(245,158,11,.16); border: 1px solid rgba(253,230,138,.25); }
    .status-bad { color: #fecaca; background: rgba(239,68,68,.16); border: 1px solid rgba(254,202,202,.25); }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #eaf0f7;
        border-radius: 16px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        font-weight: 850;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        box-shadow: 0 5px 15px rgba(15,23,42,.08);
    }
    button[kind="primary"] {
        border-radius: 14px !important;
        min-height: 3.05rem;
        font-weight: 900 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helper functions
# ============================================================
def to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def fmt_pct(value: Any) -> str:
    return f"{to_float(value) * 100:.1f}%"


def api_get(base_url: str, path: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(base_url.rstrip("/") + path, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def api_predict(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(base_url.rstrip("/") + "/predict", json=payload, timeout=40)
    response.raise_for_status()
    return response.json()


def numeric_feature_input(name: str) -> Optional[float]:
    meta = FEATURE_META[name]
    label = str(meta["label"])
    unit = str(meta.get("unit", ""))
    label_with_unit = f"{label} ({unit})" if unit else label
    unknown = st.checkbox(f"Unknown: {label}", key=f"unknown_{name}")
    value = st.number_input(
        label_with_unit,
        min_value=float(meta["min"]),
        max_value=float(meta["max"]),
        value=float(meta["default"]),
        step=float(meta["step"]),
        disabled=unknown,
        key=f"value_{name}",
    )
    return None if unknown else float(value)


def binary_feature_input(name: str) -> Optional[int]:
    meta = FEATURE_META[name]
    value = st.radio(
        str(meta["label"]),
        ["No", "Yes", "Unknown"],
        index=0,
        horizontal=True,
        key=f"binary_{name}",
    )
    if value == "Unknown":
        return None
    return 1 if value == "Yes" else 0


def sex_input() -> Optional[int]:
    value = st.radio("Sex", ["Male", "Female", "Unknown"], index=0, horizontal=True, key="sex_input")
    if value == "Unknown":
        return None
    return 1 if value == "Female" else 0


def reliability_label(confidence: float, n_missing: int) -> Tuple[str, str]:
    if n_missing >= 6 or confidence < 0.50:
        return "Low", "Low reliability: result should be interpreted with caution."
    if n_missing >= 3 or confidence < 0.65:
        return "Moderate", "Moderate reliability: useful as a screening signal, but not definitive."
    return "High", "High reliability relative to the available model inputs."


def extract_probabilities(result: Dict[str, Any]) -> Dict[str, float]:
    raw = result.get("probabilities", {}) or {}
    return {cls: to_float(raw.get(cls, raw.get(CLASS_LABEL[cls], 0.0))) for cls in CLASS_ORDER}


def render_probability_profile(probabilities: Dict[str, float]) -> None:
    html_parts = []
    for cls in CLASS_ORDER:
        value = max(0.0, min(1.0, probabilities.get(cls, 0.0)))
        html_parts.append(
            f"""
            <div class="prob-row">
                <div class="prob-label">{escape(CLASS_LABEL[cls])}</div>
                <div class="prob-track"><div class="prob-fill {cls}" style="width:{value * 100:.1f}%"></div></div>
                <div class="prob-value">{value * 100:.1f}%</div>
            </div>
            """
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def get_top_factors(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    explanation = result.get("explanation", {}) or {}
    factors = explanation.get("top_factors", []) or []
    cleaned: list[Dict[str, Any]] = []
    for item in factors[:8]:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature", ""))
        label = str(item.get("label") or FEATURE_META.get(feature, {}).get("label") or feature)
        direction = str(item.get("direction") or item.get("effect") or "model signal")
        impact = item.get("impact", item.get("delta", item.get("change", None)))
        cleaned.append({"feature": feature, "label": label, "direction": direction, "impact": impact})
    return cleaned


def render_factor_list(result: Dict[str, Any], risk_class: str) -> None:
    factors = get_top_factors(result)
    if not factors:
        st.markdown(
            """
            <div class="llm-box">
                Local feature explanation was not returned by the backend. The final category and probabilities are still available.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for item in factors:
        impact = item.get("impact")
        impact_text = "Signal"
        chip_class = ""
        if impact is not None:
            impact_value = to_float(impact, 0.0)
            impact_text = f"{impact_value * 100:+.1f} pp"
            chip_class = "negative" if impact_value < 0 else ""
        description = item.get("direction") or "Influenced the model output"
        rows.append(
            f"""
            <div class="factor-row">
                <div>
                    <div class="factor-name">{escape(str(item['label']))}</div>
                    <div class="factor-desc">{escape(str(description))}</div>
                </div>
                <div class="factor-chip {chip_class}">{escape(impact_text)}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def build_payload() -> Dict[str, Any]:
    st.markdown('<div class="section-title">Patient input features</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter available values. Use Unknown for unavailable values instead of guessing.</div>',
        unsafe_allow_html=True,
    )

    payload: Dict[str, Any] = {}
    tab1, tab2, tab3, tab4 = st.tabs(["Profile & body", "Blood pressure", "Laboratory markers", "Medical history"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            payload["age"] = numeric_feature_input("age")
            payload["sex_female"] = sex_input()
        with c2:
            payload["bmi"] = numeric_feature_input("bmi")
            payload["waist"] = numeric_feature_input("waist")
        with c3:
            payload["income_poverty_ratio"] = numeric_feature_input("income_poverty_ratio")
            st.info("Income-to-poverty ratio is optional. Mark Unknown if unavailable.", icon="ℹ️")

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            payload["mean_sbp"] = numeric_feature_input("mean_sbp")
            payload["mean_dbp"] = numeric_feature_input("mean_dbp")
        with c2:
            payload["mean_pulse"] = numeric_feature_input("mean_pulse")
            payload["hypertension_doctor_told"] = binary_feature_input("hypertension_doctor_told")
        with c3:
            payload["taking_bp_medicine"] = binary_feature_input("taking_bp_medicine")
            st.info("Structural missingness is handled by the backend when applicable.", icon="ℹ️")

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            payload["glucose"] = numeric_feature_input("glucose")
            payload["hba1c"] = numeric_feature_input("hba1c")
        with c2:
            payload["blood_urea_nitrogen"] = numeric_feature_input("blood_urea_nitrogen")
            payload["uric_acid"] = numeric_feature_input("uric_acid")
        with c3:
            payload["serum_albumin"] = numeric_feature_input("serum_albumin")
            payload["hemoglobin"] = numeric_feature_input("hemoglobin")
        c4, c5 = st.columns(2)
        with c4:
            payload["sodium"] = numeric_feature_input("sodium")
        with c5:
            payload["potassium"] = numeric_feature_input("potassium")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            payload["diabetes_yes"] = binary_feature_input("diabetes_yes")
            payload["borderline_diabetes"] = binary_feature_input("borderline_diabetes")
            if payload.get("diabetes_yes") == 1 and payload.get("borderline_diabetes") == 1:
                st.warning("Usually diabetes and borderline diabetes should not both be marked Yes.", icon="⚠️")
        with c2:
            payload["high_cholesterol_told"] = binary_feature_input("high_cholesterol_told")
            payload["taking_cholesterol_medicine"] = binary_feature_input("taking_cholesterol_medicine")

    return payload


def render_result(result: Dict[str, Any]) -> None:
    risk_class = str(result.get("risk_class", "low")).lower()
    if risk_class not in CLASS_META:
        risk_class = "low"
    meta = CLASS_META[risk_class]
    probabilities = extract_probabilities(result)
    confidence = to_float(result.get("confidence"), max(probabilities.values()) if probabilities else 0.0)
    input_quality = result.get("input_quality", {}) or {}
    n_missing = int(input_quality.get("n_missing_features", 0) or 0)
    total_features = int(input_quality.get("n_total_features", len(FEATURES)) or len(FEATURES))
    reliability, reliability_note = reliability_label(confidence, n_missing)
    clinical_note = str(result.get("clinical_note") or "This output is for kidney-risk screening support only. It is not a medical diagnosis and should not replace clinical assessment, eGFR/UACR testing, or clinician judgment.")

    st.markdown(
        f"""
        <div class="result-card {meta['css']}">
            <div class="result-kicker">Model prediction output</div>
            <div class="result-title">
                {escape(meta['title'])}
                <span class="badge {meta['css']}">{escape(meta['badge'])}</span>
            </div>
            <div style="color:#475569; font-size:1rem; line-height:1.65; max-width:900px;">
                {escape(meta['summary'])}
            </div>
            <div class="clinical-note">{escape(clinical_note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Model confidence</div><div class="metric-value">{fmt_pct(confidence)}</div><div class="metric-foot">Highest class probability</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Output reliability</div><div class="metric-value">{escape(reliability)}</div><div class="metric-foot">{escape(reliability_note)}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        complete = max(0, total_features - n_missing)
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Input completeness</div><div class="metric-value">{complete}/{total_features}</div><div class="metric-foot">Unknown values are handled by the model pipeline</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Class probability profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Displayed in clinical order: Low → Moderate → High.</div>', unsafe_allow_html=True)
        render_probability_profile(probabilities)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key model signals for this result</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Local model-behavior explanation based on feature perturbation. This is not a causal clinical explanation.</div>',
            unsafe_allow_html=True,
        )
        render_factor_list(result, risk_class)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">LLM wellness handoff</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="llm-box">
            The model output can be passed to a separate LLM module for non-clinical wellness suggestions only.
            The LLM should use the screening category, confidence, probability profile, input completeness, and key model signals.
            It must not diagnose CKD, prescribe medicine, recommend medication changes, provide treatment plans, or replace eGFR/UACR testing or clinician judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("View structured LLM context from backend", expanded=False):
        st.json(result.get("llm_context", {}))
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Developer view: raw backend response", expanded=False):
        st.json(result)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🩺 Kidney Risk App")
    st.caption("FastAPI backend + Streamlit frontend")
    api_url = st.text_input("Backend API URL", value=DEFAULT_API_URL)

    health = api_get(api_url, "/health")
    if health is None:
        st.markdown('<span class="status-pill status-bad">● Backend offline</span>', unsafe_allow_html=True)
        st.caption("Run: uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload")
    elif health.get("model_loaded"):
        st.markdown('<span class="status-pill status-ok">● Backend connected · Model loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-warn">● Backend connected · Model not loaded</span>', unsafe_allow_html=True)
        st.caption(str(health.get("load_error") or "No model load error returned."))

    with st.expander("Backend health", expanded=False):
        st.json(health or {})

    st.divider()
    st.markdown("#### User-facing output")
    st.caption("Final category")
    st.caption("Confidence + reliability")
    st.caption("Class probabilities")
    st.caption("Feature signals")
    st.caption("LLM wellness handoff")

# ============================================================
# Main application
# ============================================================
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="tag">NHANES cascade model · LLM-ready screening support</div>
        <h1>Kidney Risk Screening Support</h1>
        <p>
            Generate a low, moderate, or high kidney-risk screening category from routine demographic,
            blood pressure, metabolic, and laboratory indicators. The output is designed for screening support
            and safe handoff to a non-clinical wellness suggestion module.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="soft-warning">
        <div style="font-size:1.25rem;">⚠️</div>
        <div><b>Screening support only.</b> This app does not diagnose CKD and must not replace eGFR/UACR testing, clinician assessment, or clinical judgment.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
payload = build_payload()
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
button_col, clear_col, hint_col = st.columns([0.25, 0.18, 0.57])
with button_col:
    predict_clicked = st.button("Generate prediction", type="primary", use_container_width=True)
with clear_col:
    clear_clicked = st.button("Clear result", use_container_width=True)
with hint_col:
    st.caption("Use real measured values when available. Keep unavailable values as Unknown instead of guessing.")

if clear_clicked:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_payload", None)
    st.rerun()

if predict_clicked:
    if health is None:
        st.error("Backend is not connected. Start the FastAPI backend first.")
    elif not health.get("model_loaded"):
        st.error(f"Backend is connected, but the model is not loaded: {health.get('load_error')}")
    else:
        with st.spinner("Running kidney-risk screening model..."):
            try:
                prediction = api_predict(api_url, payload)
                st.session_state["last_result"] = prediction
                st.session_state["last_payload"] = payload
            except requests.HTTPError as exc:
                try:
                    message = exc.response.json().get("detail", str(exc))
                except Exception:
                    message = str(exc)
                st.error(f"Prediction failed: {message}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

if "last_result" in st.session_state:
    render_result(st.session_state["last_result"])

with st.expander("Developer view: last API payload", expanded=False):
    st.json(st.session_state.get("last_payload", {}))

st.markdown(
    """
    <div style="color:#64748b; font-size:.9rem; margin-top:22px; line-height:1.6;">
        Research prototype for a two-stage kidney-risk screening model. Before any real-world use, calibration review,
        external validation, monitoring, and clinician oversight are required.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)
