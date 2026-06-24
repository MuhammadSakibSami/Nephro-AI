# Thesis Supervisor Demo — Kidney-Risk Education (LLM MVP)

Three cases through the `/generate-guidance` pipeline (RAG → prompt → LLM → safety validator). The high-risk case shows the validator adding an emergency-care redirect.

## Low risk

- **mode:** `local_demo`  |  **model:** `mock`  |  **risk:** `low`  |  **validator status:** `pass`
- **sources:** Centers for Disease Control and Prevention, National Kidney Foundation
- **disclaimer:** This assistant provides personalized, guideline-grounded kidney-risk education after machine-learning-based screening. It does not diagnose CKD, prescribe treatment, or replace clinician consultation. Local demo mode: guidance is produced by a deterministic safe generator (not the production LLM), but still passes through the same safety validator. The production system uses Qwen3-4B-Instruct-2507 + guideline RAG; see the frozen Phase-6 evaluation for real-model results.

**Summary:** Your kidney-risk screening result is low. This is general information, not a diagnosis. Your result was influenced by Body mass index 28 kg/m².

**Food habits:** Aim for balanced meals with plenty of vegetables, fruits, and whole foods, and go easy on very salty and heavily processed foods.

**Rest and sleep:** Try for a steady sleep routine of about 7 to 9 hours each night.

**Daily routine:** Stay active most days with activities you enjoy, drink water through the day, and take short breaks to ease stress.

**Follow-up reminder:** Keep up regular check-ups and ask your clinician about routine kidney checks if you have any concerns.

## Moderate risk (blood pressure + glucose)

- **mode:** `local_demo`  |  **model:** `mock`  |  **risk:** `moderate`  |  **validator status:** `pass`
- **sources:** Centers for Disease Control and Prevention, National Kidney Foundation
- **disclaimer:** This assistant provides personalized, guideline-grounded kidney-risk education after machine-learning-based screening. It does not diagnose CKD, prescribe treatment, or replace clinician consultation. Local demo mode: guidance is produced by a deterministic safe generator (not the production LLM), but still passes through the same safety validator. The production system uses Qwen3-4B-Instruct-2507 + guideline RAG; see the frozen Phase-6 evaluation for real-model results.

**Summary:** Your kidney-risk screening result is moderate. This is general information, not a diagnosis. Your result was influenced by Mean systolic BP 150 mmHg and HbA1c 7.6 %.

**Food habits:** Aim for balanced meals with plenty of vegetables, fruits, and whole foods, and go easy on very salty and heavily processed foods.

**Rest and sleep:** Try for a steady sleep routine of about 7 to 9 hours each night.

**Daily routine:** Stay active most days with activities you enjoy, drink water through the day, and take short breaks to ease stress.

**Follow-up reminder:** Please check in with a clinician and ask about simple kidney tests (a blood test and a urine test) to confirm your result.

## High risk / EMERGENCY blood pressure (validator-fixed)

- **mode:** `local_demo`  |  **model:** `mock`  |  **risk:** `high`  |  **validator status:** `sanitized`
- **sources:** Centers for Disease Control and Prevention, National Kidney Foundation
- **disclaimer:** This assistant provides personalized, guideline-grounded kidney-risk education after machine-learning-based screening. It does not diagnose CKD, prescribe treatment, or replace clinician consultation. Local demo mode: guidance is produced by a deterministic safe generator (not the production LLM), but still passes through the same safety validator. The production system uses Qwen3-4B-Instruct-2507 + guideline RAG; see the frozen Phase-6 evaluation for real-model results.

- **validator actions:** emergency_redirect

**Summary:** If you have urgent symptoms such as chest pain, trouble breathing, or severe swelling, seek emergency medical care now. Your kidney-risk screening result is high. This is general information, not a diagnosis. Your result was influenced by Mean systolic BP 188 mmHg and HbA1c 8.8 %.

**Food habits:** Aim for balanced meals with plenty of vegetables, fruits, and whole foods, and go easy on very salty and heavily processed foods.

**Rest and sleep:** Try for a steady sleep routine of about 7 to 9 hours each night.

**Daily routine:** Stay active most days with activities you enjoy, drink water through the day, and take short breaks to ease stress.

**Follow-up reminder:** Please check in with a clinician and ask about simple kidney tests (a blood test and a urine test) to confirm your result.
