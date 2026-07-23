# LLM Phase 1 — Foundations & Data Contract

**Status:** ✅ Done — acceptance test passing
**Plan reference:** `LLM_PLAN.md` → Phase 1
**Date implemented:** June 2026

---

## 1. What this phase is (in plain language)

Before we build any AI model, we need a firm "agreement" about two things:

1. **What goes *into* the LLM** — the kidney-risk screening result from our existing ML model (risk level, the factors that drove it, the patient's basic inputs).
2. **What comes *out* of the LLM** — a fixed, predictable format of safe kidney-health education (Summary, Food habits, Rest and sleep, Daily routine, Follow-up reminder).

This "agreement" is called the **data contract**. We freeze it now so that every later piece — the retrieval system, the training data, the fine-tuned model, the safety checker, the evaluation — all speak the same language and nothing breaks when we connect them.

We also wrote the **canonical prompt**: the exact instructions and safety rules the LLM always receives. There is now exactly *one* copy of these rules, so we never have two versions drifting apart.

**Important framing:** the component does **kidney-risk education**, not "counseling," not diagnosis, not treatment. That wording is baked into the prompt and the docs on purpose (it is both a safety and a publishing decision).

---

## 2. What was built (technical)

### 2.1 Files

| File | Role |
|---|---|
| `ckd_risk_app/llm/__init__.py` | Marks the `llm` package |
| `ckd_risk_app/llm/serving/__init__.py` | Marks the `serving` sub-package |
| `ckd_risk_app/llm/serving/contract.py` | The data contract: input + output shapes, output parser, validators |
| `ckd_risk_app/llm/serving/prompts.py` | Canonical system prompt, safety rules, prompt builders |
| `ckd_risk_app/llm/demo_phase1.py` | Runnable acceptance test / demo |

All of these are **pure Python (standard library only)** — no `torch`, `transformers`, or other heavy dependencies. That means the contract can be imported and tested anywhere (data generation, fine-tuning, serving, evaluation) and the demo runs instantly.

### 2.2 The INPUT contract (`contract.py`)

The LLM input is adapted directly from the `llm_context` block that the **existing backend already produces** in `model_service._build_llm_context`. We did **not** change the backend; we wrote an adapter that reads it.

- `LLMInput` — a dataclass holding exactly what the prompt needs: risk class, probabilities, confidence, reliability, missing-input info, the top contributing factors, patient inputs, and the two cascade thresholds.
- `LLMInput.from_predict_response(resp)` — the adapter. It accepts **either** a full `/predict` response **or** just its `llm_context` sub-block, and is defensive: missing keys never crash it.
- `RiskFactor` — one contributing model factor, with a `human_phrase` property (e.g. `"Mean systolic BP 152 mmHg"`) used to describe factors in plain words inside the prompt.
- `validate_predict_response(resp)` — returns a list of contract problems (empty list = OK). Used by tests and, later, by the data-generation guards.

### 2.3 The OUTPUT contract (`contract.py`)

The model must always answer in this fixed, **non-chain-of-thought** structure:

```
Summary:
...
Food habits:
...
Rest and sleep:
...
Daily routine:
...
Follow-up reminder:
...
Grounding used internally: ["KDIGO2024_bp_012", "ADA2026_ckd_007"]
```

- `OUTPUT_SECTIONS` — the ordered list of user-facing sections (single source of truth).
- `GROUNDING_LABEL` — the internal-only line listing which guideline excerpts were used. **This is metadata for logging and groundedness evaluation; it is never shown verbatim to the user.** The frontend will show only a general line like "based on KDIGO/ADA guidelines."
- `GuidanceOutput` — the parsed result, with helpers:
  - `user_facing_sections()` — the five sections safe to display.
  - `missing_sections()` — which required sections came back empty.
  - `is_structurally_valid()` — `True` only if all sections are present.
- `parse_guidance(text)` — tolerant parser. Headers are matched case-insensitively at line starts; each section captures everything up to the next header; the grounding line is parsed as a JSON array (with a comma-separated fallback).
- `READABILITY_TARGET_GRADE = 8` — the reading-level goal, referenced by both the prompt and (later) evaluation.

### 2.4 The canonical prompt (`prompts.py`)

- `SCOPE_STATEMENT` — the one-paragraph "education, not diagnosis/treatment" statement.
- `SAFETY_RULES` — the **single authoritative list of 9 safety rules**. The Phase 4 validator will enforce these post-hoc; the model is also instructed to follow them. Highlights:
  - No diagnosis ("you have CKD/kidney failure" is forbidden).
  - No medication start/stop/change/dose.
  - **No individualized nutrition restrictions** (no protein/potassium/phosphorus/sodium targets) → defer to clinician / renal dietitian. *(This is the KDOQI-restriction rule from the plan.)*
  - Moderate/high results must include a clinician + lab follow-up reminder.
  - Urgent symptoms → advise prompt medical care.
  - Grade-8, calm, non-alarming language.
- `build_system_prompt()` — assembles scope + role + safety rules + reading level + output format into the system message.
- `build_user_prompt(inp, retrieved_chunks)` — renders the screening result, the top factors in plain words, and the retrieved guideline excerpts. Restricted (KDOQI) chunks are tagged `[RESTRICTED: mention only as 'discuss with a clinician/renal dietitian']`.
- `build_messages(inp, retrieved_chunks)` — returns `[{system}, {user}]` in chat format for Qwen3. The tokenizer's chat template is applied later (fine-tuning / serving); this keeps one builder feeding every phase.
- `render_messages_preview(messages)` — flattens messages to readable text for debugging.

### 2.5 How the pieces connect

```
/predict response (existing backend)
        │
        ▼
LLMInput.from_predict_response()      ← INPUT contract (contract.py)
        │
        ▼
build_messages(inp, retrieved_chunks) ← canonical prompt (prompts.py)
        │
        ▼   [later phases: send to Qwen3]
        ▼
raw model text
        │
        ▼
parse_guidance(text) → GuidanceOutput ← OUTPUT contract (contract.py)
        │
        ▼   [later phases: safety validator, then frontend]
```

---

## 3. How to test / run / experiment

### 3.1 Run the acceptance test

From the project root (`e:\ckd_risk_app_LLM_READY_INTERPRETABLE_FRONTEND`):

```bash
python -m ckd_risk_app.llm.demo_phase1
```

(Equivalent: `python ckd_risk_app/llm/demo_phase1.py` — the script self-adds the project root to `sys.path`.)

**Expected result:** it prints the input-contract check, the full system+user prompt preview, the output-parser round-trip, and ends with:

```
Phase 1 acceptance test: PASSED
```

> Note: on the Windows console you may see `kg/m²` rendered as `kg/m�`. That is only the terminal's display encoding (cp1252); the underlying UTF-8 strings are correct.

### 3.2 Experiment with your own scenarios

Open `ckd_risk_app/llm/demo_phase1.py` and edit `mock_predict_response()`:
- Change `screening_category` to `"low"` or `"high"` and watch the prompt's "Note" line and the required follow-up reminder change.
- Add/remove entries in `key_model_factors` to see how factor phrasing changes.
- Edit `mock_retrieved_chunks()` to flip a chunk's `restricted` flag and confirm the `[RESTRICTED]` tag appears.

### 3.3 Use the contract against the *real* backend output

Once the model package is loaded and the API runs, you can feed an actual response straight in:

```python
import requests
from ckd_risk_app.llm.serving.contract import LLMInput
from ckd_risk_app.llm.serving.prompts import build_messages, render_messages_preview

resp = requests.post("http://localhost:8000/predict", json={
    "age": 58, "sex_female": 1, "bmi": 33.0, "mean_sbp": 152,
    "hba1c": 7.8, "blood_urea_nitrogen": 22, "diabetes_yes": 1,
    "hypertension_doctor_told": 1, "taking_bp_medicine": 1,
}).json()

inp = LLMInput.from_predict_response(resp)
print(render_messages_preview(build_messages(inp, retrieved_chunks=None)))
```

This is exactly how Phase 5 (the MVP runtime) will call it — only with real retrieved chunks instead of `None`.

### 3.4 Quick parser check

```python
from ckd_risk_app.llm.serving.contract import parse_guidance
out = parse_guidance(open_your_model_output_text)
print(out.is_structurally_valid(), out.missing_sections(), out.grounding)
```

---

## 4. Design decisions worth remembering

- **No backend changes.** We adapt the existing `llm_context`; we don't modify `model_service`. This keeps the ML side stable.
- **One copy of the safety rules.** `SAFETY_RULES` in `prompts.py` is authoritative. When the Phase 4 validator is built, it imports/mirrors this list — do not fork it.
- **Grounding is internal.** Chunk ids are for logging + groundedness scoring, never shown to the user. The frontend shows a general provenance line only.
- **No chain-of-thought.** The output format deliberately excludes visible reasoning (a medical-safety choice).
- **Pure stdlib.** Keeps the contract importable in every environment, including the data-generation and evaluation scripts that won't always have a GPU stack installed.

---

## 5. Known open items / dependencies

- **Teacher-model decision (Phase 1, step 4): still pending.** Needed before Phase 7 (synthetic data generation), *not* before the MVP. Options: a hosted large model (Claude/GPT-4-class) or a larger hosted Qwen. Once chosen, record it in `LLM_PLAN.md` Phase 1 with the transparency statement (teacher used only for data generation; no external API at inference; only de-identified profiles sent).

---

## 6. Next steps

**Phase 2 — Knowledge Base Construction** (next, per the MVP-first order in `LLM_PLAN.md`):
- Collect the guideline source documents into `ckd_risk_app/llm/knowledge_base/raw/` (KDIGO 2024 / 2021 BP / 2022 Diabetes, ADA 2026 CKD chapter, KDOQI 2020 nutrition [restricted], CDC/NKF patient education; KDIGO 2012 background-only).
- Build `ingest.py` (PDF → cleaned text → section-aware chunks → JSONL with metadata including `restricted`, `priority`, `topic`).
- Produce `sources.csv` (citation + URL + access date per source) for paper reproducibility.
- Manual scoping pass to keep only prevention/lifestyle/early-stage content and exclude dialysis/transplant/medication-dosing.

This will be documented in `docs/LLM_PHASE2_KNOWLEDGE_BASE.md` when complete.

---

## 7. File map (quick reference)

```
ckd_risk_app/llm/
  __init__.py
  demo_phase1.py                  # run: python -m ckd_risk_app.llm.demo_phase1
  serving/
    __init__.py
    contract.py                   # LLMInput, GuidanceOutput, parse_guidance, validators
    prompts.py                    # SAFETY_RULES, build_system_prompt, build_messages
docs/
  LLM_PHASE1_FOUNDATIONS_AND_CONTRACT.md   # this file
```
