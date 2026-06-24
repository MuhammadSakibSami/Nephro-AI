// Risk + reliability presentation helpers.

export const RISK_META = {
  low: {
    key: "low", label: "Low",
    text: "text-risk-low", bg: "bg-risk-lowbg", ring: "ring-risk-lowring",
    dot: "bg-risk-low", border: "border-green-200",
    blurb: "Your screening suggests a low kidney-risk level right now.",
  },
  moderate: {
    key: "moderate", label: "Moderate",
    text: "text-risk-mod", bg: "bg-risk-modbg", ring: "ring-risk-modring",
    dot: "bg-risk-mod", border: "border-amber-200",
    blurb: "Your screening suggests a moderate kidney-risk level worth attention.",
  },
  high: {
    key: "high", label: "High",
    text: "text-risk-high", bg: "bg-risk-highbg", ring: "ring-risk-highring",
    dot: "bg-risk-high", border: "border-red-200",
    blurb: "Your screening suggests a high kidney-risk level — please follow up with a clinician.",
  },
};

export function riskMeta(riskClass) {
  return RISK_META[String(riskClass || "").toLowerCase()] || RISK_META.low;
}

export const RELIABILITY_LABEL = {
  high: "High reliability",
  moderate: "Medium reliability",
  low: "Low reliability",
};

export function reliabilityColor(level) {
  const l = String(level || "").toLowerCase();
  if (l === "high") return "text-risk-low";
  if (l === "moderate") return "text-risk-mod";
  return "text-ink-500";
}

// Human-friendly factor name from a backend factor object.
export function factorName(factor) {
  if (factor?.label) return factor.label;
  const map = {
    mean_sbp: "Blood pressure (systolic)", mean_dbp: "Blood pressure (diastolic)",
    hba1c: "Blood sugar (HbA1c)", glucose: "Glucose", bmi: "Body mass index",
    waist: "Waist size", blood_urea_nitrogen: "Kidney blood test (BUN)",
    uric_acid: "Uric acid", serum_albumin: "Albumin", hemoglobin: "Hemoglobin",
    diabetes_yes: "Diabetes history", hypertension_doctor_told: "High blood pressure history",
  };
  return map[factor?.feature] || factor?.feature || "Factor";
}

export function factorDirection(factor) {
  return factor?.direction === "pulls_away_from_prediction" ? "lowers" : "raises";
}

export function completenessPct(prediction) {
  const iq = prediction?.input_quality;
  if (!iq) return null;
  const supplied = iq.n_supplied_features;
  const total = iq.n_model_features || (iq.n_supplied_features + iq.n_missing_features);
  if (!total) return null;
  return Math.round((supplied / total) * 100);
}
