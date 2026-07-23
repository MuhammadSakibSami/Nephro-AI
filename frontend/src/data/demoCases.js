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
      age: 42, income_poverty_ratio: 3, waist: 88,
      mean_sbp: 118, mean_dbp: 76,
      glucose: 92, hba1c: 5.3, blood_urea_nitrogen: 12, uric_acid: 5.0,
      serum_albumin: 4.4, hemoglobin: 14,
      diabetes_yes: 0, hypertension_doctor_told: 0, taking_bp_medicine: 0,
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
      age: 54, income_poverty_ratio: 2.5, waist: 100,
      mean_sbp: 140, mean_dbp: 88,
      glucose: 150, hba1c: 7.0, blood_urea_nitrogen: 15, uric_acid: 5.6,
      serum_albumin: 4.3, hemoglobin: 14.2,
      diabetes_yes: 1, hypertension_doctor_told: 1, taking_bp_medicine: 1,
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
      age: 66, income_poverty_ratio: 1.5, waist: 114,
      mean_sbp: 188, mean_dbp: 116,
      glucose: 214, hba1c: 8.9, blood_urea_nitrogen: 31, uric_acid: 8.6,
      serum_albumin: 3.4, hemoglobin: 11.4,
      diabetes_yes: 1, hypertension_doctor_told: 1, taking_bp_medicine: 1,
    },
  },
];
