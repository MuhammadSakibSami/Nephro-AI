# Phase 7 — RAFT Training Data — Data Card

- Date: 2026-07-23
- Profiles generated: 12 | examples kept: 12 (pass rate 1.0)
- Split: train=11, val=1
- Class mix (kept): low=4, moderate=4, high=4
- Teacher mix: {'self': 12} (Gemini share 0.0; model gemini-2.5-flash)
- Retrieval: k_golden=4, distractors_per_example=2
- Drops: {}

## Method
- Targets generated under the FULL prompt; examples pair the COMPACT prompt with the gold (prompt-compression objective).
- Bulk targets via self-distillation (base Qwen3 + RAG + validator); a subset via Gemini (free tier) for harder/diverse/emergency cases.
- **Every gold answer passed `scan_unsafe`** (0 unsafe), all 5 sections present, mentions the risk level. Unicode normalized to ASCII.
- Teacher used only on synthetic, de-identified profiles; no real patient data; no external API at inference.
- Profiles drawn from the training distribution; the 30 held-out eval profiles and the ML test set were NOT used.

## Example (first kept)

- profile T_H0002 (high), teacher self:qwen:Qwen/Qwen3-4B-Instruct-2507

**Assistant target:**
```
Summary:
You have a high risk for kidney problems based on your body measurements and blood values. This means your waist is large (130 cm) and your blood protein level (serum albumin) is a bit low, which can be signs of possible kidney stress. This is not a diagnosis, but it shows you should take steps to protect your kidneys.

Food habits:
Since your waist is large and your blood protein is low, it's important to eat balanced meals that don't add extra strain on your kidneys. Focus on eating plenty of fruits, vegetables, whole grains, and lean proteins like chicken or beans. Avoid too much salt, especially if you have high blood pressure (even if you don't have it yet), because high salt can affect kidney function. Since your blood sugar is slightly high (95 mg/dL), try to eat steady portions of carbs to keep your blood sugar steady. Avoid processed foods and sugary drinks. Talk to a c
```