# Phase 7 — RAFT Training Data — Data Card

- Date: 2026-07-26
- Profiles generated: 600 | examples kept: 600 (pass rate 1.0)
- Split: train=540, val=60
- Class mix (kept): low=200, moderate=200, high=200
- Teacher mix: {'self': 600} (Gemini share 0.0; model gemini-2.5-flash)
- Retrieval: k_golden=4, distractors_per_example=2
- Drops: {}

## Method
- Targets generated under the FULL prompt; examples pair the COMPACT prompt with the gold (prompt-compression objective).
- Bulk targets via self-distillation (base Qwen3 + RAG + validator); a subset via Gemini (free tier) for harder/diverse/emergency cases.
- **Every gold answer passed `scan_unsafe`** (0 unsafe), all 5 sections present, mentions the risk level. Unicode normalized to ASCII.
- Teacher used only on synthetic, de-identified profiles; no real patient data; no external API at inference.
- Profiles drawn from the training distribution; the 30 held-out eval profiles and the ML test set were NOT used.

## Example (first kept)

- profile T_M0123 (moderate), teacher self:qwen:Qwen/Qwen3-4B-Instruct-2507

**Assistant target:**
```
Summary:
You have a moderate risk for kidney problems. This means your body may be under some stress, especially because your waist is 107 cm and your hemoglobin is 12 g/dL. These numbers can point to extra body fat or possible changes in how your blood works, so it's important to stay on top of your health.

Food habits:
Since your waist is large, it's a good idea to choose foods that help reduce belly fat. Eat more vegetables, whole grains, and lean proteins like chicken, fish, or beans. Limit processed foods and sugary drinks. Since your hemoglobin is normal, you don't need to worry about anemia, but staying well-nourished helps your body work better. Talk to a doctor or dietitian if you want a plan that fits your lifestyle.

Rest and sleep:
Try to get 7 to 9 hours of sleep each night. Go to bed and wake up at the same time every day. Good sleep helps keep your blood pressure and bloo
```