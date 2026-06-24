// Patient input form schema. Field `key` values match the backend MODEL_FEATURES
// exactly. `patient_id` is intentionally NOT a backend field (the API forbids
// extras) — it is kept frontend-only for the session.

export const NUMBER = "number";
export const BINARY = "binary"; // yes/no -> 1/0
export const SELECT = "select";

export const FORM_SECTIONS = [
  {
    id: "basic",
    title: "Basic profile",
    icon: "user",
    fields: [
      { key: "patient_id", label: "Patient / Session ID", type: "text", optional: true,
        backendOmit: true, placeholder: "e.g. DEMO-001", help: "Optional. Not sent to the model." },
      { key: "age", label: "Age", unit: "years", type: NUMBER, min: 0, max: 120,
        normal: [18, 85], placeholder: "55" },
      { key: "sex_female", label: "Sex", type: SELECT,
        options: [{ value: 1, label: "Female" }, { value: 0, label: "Male" }] },
      { key: "income_poverty_ratio", label: "Income-to-poverty ratio", unit: "ratio",
        type: NUMBER, min: 0, max: 10, optional: true, placeholder: "2.5",
        help: "Optional NHANES socioeconomic ratio." },
    ],
  },
  {
    id: "body",
    title: "Body measurements",
    icon: "ruler",
    fields: [
      { key: "bmi", label: "Body mass index (BMI)", unit: "kg/m²", type: NUMBER,
        min: 5, max: 100, normal: [18.5, 25], placeholder: "28" },
      { key: "waist", label: "Waist circumference", unit: "cm", type: NUMBER,
        min: 20, max: 250, normal: [70, 100], placeholder: "94" },
    ],
  },
  {
    id: "bp",
    title: "Blood pressure",
    icon: "heart",
    fields: [
      { key: "mean_sbp", label: "Systolic BP", unit: "mmHg", type: NUMBER,
        min: 40, max: 300, normal: [90, 120], placeholder: "120" },
      { key: "mean_dbp", label: "Diastolic BP", unit: "mmHg", type: NUMBER,
        min: 20, max: 200, normal: [60, 80], placeholder: "78" },
      { key: "mean_pulse", label: "Pulse", unit: "bpm", type: NUMBER,
        min: 20, max: 220, normal: [60, 100], placeholder: "72" },
    ],
  },
  {
    id: "labs",
    title: "Routine lab values",
    icon: "flask",
    fields: [
      { key: "glucose", label: "Glucose", unit: "mg/dL", type: NUMBER,
        min: 10, max: 1000, normal: [70, 100], placeholder: "95" },
      { key: "hba1c", label: "HbA1c", unit: "%", type: NUMBER,
        min: 2, max: 25, normal: [4, 5.7], placeholder: "5.5" },
      { key: "blood_urea_nitrogen", label: "Blood urea nitrogen (BUN)", unit: "mg/dL",
        type: NUMBER, min: 0, max: 300, normal: [7, 20], placeholder: "14" },
      { key: "uric_acid", label: "Uric acid", unit: "mg/dL", type: NUMBER,
        min: 0, max: 50, normal: [3.5, 7.2], placeholder: "5.5" },
      { key: "serum_albumin", label: "Serum albumin", unit: "g/dL", type: NUMBER,
        min: 0, max: 10, normal: [3.5, 5.0], placeholder: "4.2" },
      { key: "sodium", label: "Sodium", unit: "mmol/L", type: NUMBER,
        min: 80, max: 220, normal: [135, 145], placeholder: "140" },
      { key: "potassium", label: "Potassium", unit: "mmol/L", type: NUMBER,
        min: 1, max: 12, normal: [3.5, 5.1], placeholder: "4.2" },
      { key: "hemoglobin", label: "Hemoglobin", unit: "g/dL", type: NUMBER,
        min: 1, max: 30, normal: [12, 17], placeholder: "14" },
    ],
  },
  {
    id: "history",
    title: "Medical history",
    icon: "clipboard",
    fields: [
      { key: "diabetes_yes", label: "Diagnosed diabetes", type: BINARY },
      { key: "borderline_diabetes", label: "Borderline / prediabetes", type: BINARY },
      { key: "hypertension_doctor_told", label: "Diagnosed high blood pressure", type: BINARY },
      { key: "high_cholesterol_told", label: "Diagnosed high cholesterol", type: BINARY },
      { key: "taking_bp_medicine", label: "Taking blood-pressure medicine", type: BINARY },
      { key: "taking_cholesterol_medicine", label: "Taking cholesterol medicine", type: BINARY },
    ],
  },
];

// All backend feature keys (for completeness calc), excluding frontend-only fields.
export const BACKEND_KEYS = FORM_SECTIONS.flatMap((s) =>
  s.fields.filter((f) => !f.backendOmit).map((f) => f.key)
);

export const FIELD_BY_KEY = Object.fromEntries(
  FORM_SECTIONS.flatMap((s) => s.fields).map((f) => [f.key, f])
);

// Build the backend payload from form values (numbers parsed, omits frontend-only
// and empty fields so the backend imputes them).
export function buildPayload(values) {
  const payload = {};
  for (const section of FORM_SECTIONS) {
    for (const f of section.fields) {
      if (f.backendOmit) continue;
      const raw = values[f.key];
      if (raw === undefined || raw === "" || raw === null) continue;
      payload[f.key] = f.type === NUMBER ? Number(raw) : Number(raw);
    }
  }
  return payload;
}

// Returns { errors: {key: msg}, missing: [keys], suppliedCount, requiredCount }
export function validateValues(values) {
  const errors = {};
  let supplied = 0;
  let required = 0;
  const missing = [];
  for (const section of FORM_SECTIONS) {
    for (const f of section.fields) {
      if (f.backendOmit) continue;
      required += f.optional ? 0 : 1;
      const raw = values[f.key];
      const empty = raw === undefined || raw === "" || raw === null;
      if (empty) {
        if (!f.optional) missing.push(f.key);
        continue;
      }
      supplied += 1;
      if (f.type === NUMBER) {
        const n = Number(raw);
        if (Number.isNaN(n)) errors[f.key] = "Enter a number";
        else if (f.min !== undefined && n < f.min) errors[f.key] = `Must be ≥ ${f.min}`;
        else if (f.max !== undefined && n > f.max) errors[f.key] = `Must be ≤ ${f.max}`;
      }
    }
  }
  return { errors, missing, suppliedCount: supplied, requiredCount: required };
}
