// Three supervisor demo cases. Values are full, realistic payloads chosen to push
// the real ML model toward each risk level. The high case uses emergency-level BP
// (SBP 188 / DBP 116) so the safety validator adds an urgent-care redirect.

export const DEMO_CASES = [
  {
    id: "low",
    title: "Low risk",
    subtitle: "Healthy adult — preventive education",
    expected: "Low risk · validator status: pass",
    accent: "low",
    values: {
      patient_id: "DEMO-LOW",
      age: 42, sex_female: 1, income_poverty_ratio: 3,
      bmi: 26, waist: 88,
      mean_sbp: 118, mean_dbp: 76, mean_pulse: 72,
      glucose: 92, hba1c: 5.3, blood_urea_nitrogen: 12, uric_acid: 5.0,
      serum_albumin: 4.4, sodium: 140, potassium: 4.2, hemoglobin: 14,
      diabetes_yes: 0, borderline_diabetes: 0, hypertension_doctor_told: 0,
      high_cholesterol_told: 0, taking_bp_medicine: 0, taking_cholesterol_medicine: 0,
    },
  },
  {
    id: "moderate",
    title: "Moderate risk",
    subtitle: "Raised blood pressure + glucose",
    expected: "Moderate risk · personalized to BP/glucose · clinician + lab reminder",
    accent: "mod",
    values: {
      patient_id: "DEMO-MODERATE",
      age: 58, sex_female: 1, income_poverty_ratio: 2,
      bmi: 32, waist: 104,
      mean_sbp: 150, mean_dbp: 92, mean_pulse: 78,
      glucose: 168, hba1c: 7.6, blood_urea_nitrogen: 18, uric_acid: 6.6,
      serum_albumin: 4.0, sodium: 139, potassium: 4.4, hemoglobin: 13.4,
      diabetes_yes: 1, borderline_diabetes: 0, hypertension_doctor_told: 1,
      high_cholesterol_told: 1, taking_bp_medicine: 1, taking_cholesterol_medicine: 0,
    },
  },
  {
    id: "high",
    title: "High risk · emergency BP",
    subtitle: "Validator adds urgent-care redirect",
    expected: "High risk · validator status: sanitized · urgent-care redirect added",
    accent: "high",
    values: {
      patient_id: "DEMO-HIGH-EMERGENCY",
      age: 66, sex_female: 0, income_poverty_ratio: 1.5,
      bmi: 34, waist: 114,
      mean_sbp: 188, mean_dbp: 116, mean_pulse: 86,
      glucose: 214, hba1c: 8.9, blood_urea_nitrogen: 31, uric_acid: 8.6,
      serum_albumin: 3.4, sodium: 138, potassium: 4.9, hemoglobin: 11.4,
      diabetes_yes: 1, borderline_diabetes: 0, hypertension_doctor_told: 1,
      high_cholesterol_told: 1, taking_bp_medicine: 1, taking_cholesterol_medicine: 1,
    },
  },
];
