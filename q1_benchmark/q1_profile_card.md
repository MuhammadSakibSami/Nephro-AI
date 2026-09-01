# Q1 LLM Benchmark - 300 Held-Out Profiles

- Built: 2026-09-01
- Generation seed: 42 (deterministic)
- Profiles: 300
- **Benchmark hash: `563c6a66931da2e76eda7c439f70adf6845deea8043bed1fc08502d5822b4fc5`**
- Candidates rejected as duplicates during sampling: 0

## Construction

Risk category, class probabilities, confidence, `key_model_factors` and the
reliability band are produced by calling the deployed cascade
(`KidneyRiskModelService.predict`) on each sampled payload - never hand-assigned.
Group/class cells are filled by rejection sampling on the model's actual output.

`patient_inputs` contains exactly the 14 Model C features, matching the live API.
Lifestyle context is recorded under `context.lifestyle` for subgroup analysis and
adversarial realism, and is deliberately not injected into `patient_inputs`, so the
evaluated prompt is identical in shape to the deployed prompt.

No target-defining variable (eGFR, UACR, serum creatinine, urine albumin/creatinine,
G-stage, A-stage) appears anywhere in a profile - asserted at build time.

Everything under `evaluator` is evaluator-only and is asserted absent from every
rendered prompt by the runner before generation.

## Distribution

| Group | Low | Moderate | High | Total |
|---|---:|---:|---:|---:|
| routine_lifestyle | 30 | 30 | 30 | 90 |
| comorbidity | 20 | 20 | 20 | 60 |
| emergency | 20 | 20 | 20 | 60 |
| boundary | 10 | 10 | 10 | 30 |
| missing_input | 10 | 10 | 10 | 30 |
| adversarial | 10 | 10 | 10 | 30 |
| **Total** | **100** | **100** | **100** | **300** |

## Stress coverage

- Reliability bands: {'higher': 152, 'moderate': 45, 'lower': 103}
- Profiles with red flags: 60
- Profiles with missing inputs: 30
- Low category + lower reliability (false-reassurance tests): 3
- Adversarial attack categories: {'diagnosis_request': 4, 'medication_request': 4, 'dosage_request': 4, 'extreme_fluid': 4, 'extreme_potassium_protein': 4, 'category_change_request': 4, 'system_prompt_exfiltration': 3, 'injection_override': 3}

## Duplicate audit

Every candidate was compared against the 600 Phase-7 fine-tuning examples, the 60
validation examples and the 30 Phase-6 profiles, using exact 14-feature signature
matching plus an age-excluded signature match (so a profile that differs only by age
is rejected as a near duplicate). See `q1_duplicate_audit.csv`.


## Sampler relaxation

No cell required relaxation; every profile used the strict group shaping.

> Frozen after build. Do not modify this benchmark after observing model outputs.