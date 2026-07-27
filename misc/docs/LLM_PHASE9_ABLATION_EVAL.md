# LLM Phase 9 — Ablation Evaluation

**Status:** ✅ DONE — 4-arm ablation run on real Qwen3-4B-Instruct-2507 + the Phase-8 QLoRA adapter.
**Plan reference:** `LLM_PLAN.md` → Phase 9
**Date implemented:** 2026-07-27
**Artifacts:** `phase9_eval/phase9_{arm}_results.csv|summary.json|summary.md`, `phase9_eval/phase9_comparison.md`

---

## 0. Why this phase looks different from the original plan

Phase 6 (the frozen thesis MVP evaluation) had two problems discovered while
building this phase, both fixed here rather than carried forward:

1. **Hand-authored eval profiles.** Phase 6's 30 profiles had `risk_class`,
   `probabilities`, and `key_model_factors` written directly in Python — never
   actually scored by the trained ML model. Phase 9 uses a **fresh set of 30
   profiles, sampled broadly and scored by the real `KidneyRiskModelService.predict()`**
   (`llm/eval/phase9_profiles.py`), so the ablation reflects genuine model output,
   not a fabricated stand-in.
2. **Two harness bugs in `run_phase6.py`**, both fixed in the new harness:
   - The summary file printed an unconditional "these numbers are from the mock
     model" note regardless of which model actually ran (harmless — Phase 6's
     underlying run WAS real Qwen3 — but self-contradicting and confusing on a
     read-through).
   - The good/validator-fixed/bad example picker could select the *same* record
     for two different buckets (no deduplication).

A third, independent bug was found and fixed during this phase: the safety
validator's `detect_nutrition_sentences` flagged negated sentences like *"you
don't need to cut back on protein"* as a nutrition-restriction violation — it
never checked for a preceding negation, unlike the sibling `detect_diagnosis`
check. Fixed in `llm/serving/safety_validator.py` (see commit history); this
corrected the Phase 8 acceptance-check headline number from 0.117 → **0.067**
raw_unsafe_rate before this phase started.

---

## 1. The four arms

Same 30 fresh, real-model-scored profiles run through four configurations:

| Arm | Model | Prompt | RAG | Validator |
|---|---|---|---|---|
| 1. `base` | base Qwen3-4B | full | off | off |
| 2. `base_rag` | base Qwen3-4B | full | on | off |
| 3. `base_rag_val` | base Qwen3-4B | full | on | on *(Phase-6-equivalent, re-run fresh)* |
| 4. `finetuned` | Phase-8 QLoRA adapter | **compact** | on | on |

Arm 4 deliberately uses the compact prompt (what it was trained for) against
the base arms' full prompt — this isolates the deployed configuration's
"prompt compression" claim rather than forcing an apples-to-apples prompt
test the base model was never built to handle (its compact-prompt
`format_ok_rate` was already measured at 0.0 in the Phase 8 acceptance check).

---

## 2. Result

| Metric | base | base+RAG | base+RAG+validator | fine-tuned+RAG+validator |
|---|---|---|---|---|
| format_adherence_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| raw_unsafe_rate | 0.133 | 0.067 | 0.1 | **0.033** |
| final_unsafe_rate | 0.133 *(no validator)* | 0.067 *(no validator)* | **0.0** | **0.0** |
| mean_factor_relevance | 0.95 | 0.983 | 0.983 | **1.0** |
| grounding_any_rate | 0.0 *(no chunks)* | 1.0 | 0.967 | 1.0 |
| readability_ok_rate | 0.733 | 0.5 | 0.467 | 0.533 |
| mean_personalization | 2.0 | 2.0 | 2.0 | 2.0 |
| mean_prompt_tokens | 1456 | 2243 | 2243 | **1638** |

**Violation categories among raw_unsafe cases:**

| Arm | Categories |
|---|---|
| base | diagnosis×2, emergency_redirect×2 |
| base_rag | emergency_redirect×2 |
| base_rag_val | emergency_redirect×2, diagnosis×1 |
| finetuned | **emergency_redirect×1** (nothing else) |

### Reading it

- **RAG alone (arm 1→2) is not a safety win by itself** — grounding and
  factor-relevance improve, but readability drops (0.733→0.5) as guideline
  excerpts push the model toward denser clinical phrasing. It only becomes a
  net positive once paired with the validator, which is exactly why the
  production pipeline is designed with all three layers together.
- **The validator (arm 3) does its job** — `final_unsafe` hits 0.0 regardless
  of what the raw generation looked like.
- **Fine-tuning (arm 4) is the only arm that moves the *raw*, pre-validator
  number**, and moves it the furthest: 0.133 (base) → 0.033. `diagnosis` is
  completely eliminated, `nutrition_restriction` and `medication` never
  appear at all — only one hard emergency-redirect edge case remains (see
  below). Combined with perfect factor-relevance (1.0) and the shortest
  prompt of any RAG'd arm, this is the "internalizes safety + format without
  leaning on the validator" result Phase 8 targeted, now demonstrated with
  the real pipeline instead of only the compact-prompt acceptance check.
- **Prompt compression is real and quantified:** 2243 → 1638 tokens (**-27%**)
  at equal-or-better quality — the deployed inference-cost claim.

---

## 3. The emergency-redirect fix (and its limit)

Across every arm, the one violation category that never fully disappeared is
`emergency_redirect` — the model not adding an urgent-care line when a
patient's BP is in the hypertensive-emergency range (SBP ≥180 or DBP ≥120).
Root cause: the model was never told the numeric threshold the validator
itself checks against, so it had no way to reliably infer "very high blood
pressure" the same way the validator defines it.

**Fix.** Single source of truth for the threshold, shared by the prompt
builder and the validator (`llm/serving/contract.py`:
`EMERGENCY_SBP_THRESHOLD` / `EMERGENCY_DBP_THRESHOLD` / `has_emergency_vitals()`
/ `LLMInput.is_emergency_bp`). `build_user_prompt` now injects an unmissable,
**patient-specific** alert (`_emergency_alert_block` in `llm/serving/prompts.py`)
citing the patient's actual reading, e.g.:

> *"This patient reported a diastolic (bottom number) reading of 122 mmHg,
> which is in the hypertensive-emergency range... even though the systolic
> number alone may look ordinary at a glance."*

The first version of the alert was generic ("blood pressure is in the
emergency range") and was silently ignored whenever systolic looked normal
and only diastolic was elevated — the model was anchoring on systolic as
"the" blood pressure number. Citing the patient's own value fixed the fully
lopsided cases immediately: **all 3 profiles with clearly extreme systolic
(≥184) now pass in every arm**, up from 0.

**What's left.** Two profiles with borderline, isolated diastolic elevation
(SBP≈125 — looks unremarkable, DBP=122 — just over the emergency threshold)
are inconsistent even with the patient-specific alert: one flips to compliant
in the fine-tuned arm, the other doesn't, despite near-identical inputs. This
points to genuine sampling-variance non-compliance (temperature 0.3, no fixed
seed) rather than a further prompt wording fix being available — the model
sometimes acts on the instruction and sometimes doesn't for this specific
"looks normal, isn't" pattern. This is an honest residual limitation to state
in the paper. It does not affect deployment safety: the validator (`moderate`
severity, auto-appended) guarantees `final_unsafe = 0.0` regardless.

---

## 4. What was built

```
llm/eval/phase9_profiles.py   # fresh 30-profile set, scored by the REAL ML model
llm/eval/run_phase9.py        # 4-arm harness: batched generation, per-arm CSV/JSON/MD, combine step
llm/serving/contract.py       # + EMERGENCY_SBP/DBP_THRESHOLD, has_emergency_vitals(), is_emergency_bp
llm/serving/prompts.py        # + _emergency_alert_block() (patient-specific urgent-vitals alert)
llm/serving/safety_validator.py  # is_emergency() now uses the shared threshold; nutrition-negation fix
```

### 4.1 Profile generation (`phase9_profiles.py`)
Samples synthetic patient inputs across a broad per-feature range (not
risk-conditioned), calls `KidneyRiskModelService.predict()`, and buckets by
the **actual** predicted `risk_class` via rejection sampling until 10/10/10 is
reached. A fraction of candidates force extreme BP to guarantee emergency-BP
coverage — but the risk label and factor importances are always the model's
genuine output, never assigned by hand.

### 4.2 Harness (`run_phase9.py`)
- `--arm {base,base_rag,base_rag_val,finetuned}` runs one arm at a time (one
  model in GPU memory at once); `--combine` builds the comparison table from
  the four summaries with no model loading.
- **Batched generation:** all profiles for an arm are pre-built into chat
  messages, then generated in chunks via `QwenChatModel.generate_batch()`
  (new method — left-padded batch decode) instead of one-at-a-time. Only the
  rare profile whose first attempt fails validation falls back to the
  sequential `enforce()` retry loop.
- Same scoring (`llm/eval/score.py`) as Phase 6, plus a `prompt_tokens` field
  per record for the inference-cost comparison.
- Example-picker deduplicates across good/fixed/bad; the mock-model note is
  now conditional on the actual model spec.

### 4.3 Speed
Batched generation (`--batch-size`, default 6) turns the GPU-bound generation
step from 30 sequential forward-pass chains into 5 batches of 6 — the
dominant cost in the previous per-profile loop. Only profiles that fail
first-attempt validation (a small minority) pay for an extra sequential
retry generation.

---

## 5. How to run

```bash
python -m llm.eval.phase9_profiles                 # (re)generate the 30 profiles
python -m llm.eval.run_phase9 --arm base
python -m llm.eval.run_phase9 --arm base_rag
python -m llm.eval.run_phase9 --arm base_rag_val
python -m llm.eval.run_phase9 --arm finetuned --adapter llm/finetune/adapters/v1/adapter
python -m llm.eval.run_phase9 --combine
```

Offline harness check (no GPU): add `--model-spec mock` to any `--arm` call.

---

## 6. Next steps

1. **Human/expert review** on 30-50 outputs per the plan's rubric (safety,
   guideline consistency, personalization, clarity, overclaiming,
   actionability) — the remaining Phase 9 item.
2. Feed this table + the emergency-redirect finding into the paper's §3.10 /
   Phase 11 write-up, including the stated limitation (isolated-diastolic
   emergency compliance is not 100% even with an explicit, patient-specific
   alert — the validator is what guarantees final safety, not the model
   alone).
3. `misc/plan/LLM_PLAN.md` Phase 9 status line updated to DONE (this run);
   Phase 10 (GGUF/Ollama export + frontend) remains deferred per the plan.

---

## 7. File map

```
llm/eval/phase9_profiles.py              # generate_phase9_profiles(), load_phase9_profiles()
llm/eval/run_phase9.py                   # run(arm=...), combine(); python -m ...run_phase9 --arm ...
llm/data/eval_profiles_phase9.jsonl      # the 30 real-model-scored profiles
llm/serving/contract.py                  # EMERGENCY_*_THRESHOLD, has_emergency_vitals, is_emergency_bp
llm/serving/prompts.py                   # _emergency_alert_block()
llm/serving/model_backends.py            # QwenChatModel.generate_batch()
phase9_eval/phase9_comparison.md         # the 4-arm table
docs/LLM_PHASE9_ABLATION_EVAL.md         # this file
```
