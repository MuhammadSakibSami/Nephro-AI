# LLM Phase 5 — MVP (Base Qwen3 + RAG + Guardrail)

**Status:** ✅ Done — runtime built + verified end-to-end (offline mock); Colab-ready for Qwen3
**Plan reference:** `LLM_PLAN.md` → Phase 5
**Date implemented:** June 2026

---

## 1. What this phase is (in plain language)

This is the first time all the LLM pieces run together as **one call**. Give it a
patient's risk result, and it:

1. builds a search query from the result,
2. pulls the most relevant guideline chunks (RAG),
3. writes the prompt,
4. asks the language model for kidney-risk education,
5. runs the safety guardrail,
6. returns a **guaranteed-safe**, well-formed answer with its sources.

It uses the **base (un-fine-tuned) Qwen3-4B** on purpose. That gives Phase 6 a
baseline to answer the key question: *do we even need to fine-tune, or is base +
RAG + guardrail already good enough?* Phase 8 later swaps in the fine-tuned model
with no other change.

Because the model itself needs a GPU, the model call runs in **Colab**. So the
runtime also ships with a **deterministic offline mock model**, which lets the
entire pipeline (retrieve → prompt → generate → validate) be tested here without
a GPU.

---

## 2. What was built (technical)

```
ckd_risk_app/llm/serving/
  model_backends.py    # QwenChatModel (prod) + MockGuidanceModel (offline) + get_model()
  llm_runtime.py       # GuidanceRuntime.generate_guidance(...) -> GuidanceResult
ckd_risk_app/llm/
  demo_phase5.py       # 5 end-to-end cases + validator-integration check
```

### 2.1 Model backends (`model_backends.py`)
One interface, `generate(messages) -> str`:
- **`QwenChatModel`** — `Qwen/Qwen3-4B-Instruct-2507` (non-thinking instruct
  variant) via transformers. Low-temperature decoding (0.3), capped tokens. Has an
  `adapter_path` hook so Phase 8 can load the LoRA adapter with no API change.
- **`MockGuidanceModel`** — reads the rendered prompt (risk level, top factors,
  retrieved chunk ids) and returns a complete, safe `GuidanceOutput`. An
  `unsafe_first=True` switch forces an unsafe first attempt to exercise the
  guardrail's retry/fallback.
- **`get_model("auto"|"qwen"|"mock")`** — `auto` uses Qwen3 if torch+transformers
  import, else the mock.

### 2.2 Runtime (`llm_runtime.py`)
- **`GuidanceRuntime(model="auto", k=4, max_retries=1, log_path=None)`** loads the
  retriever once and the chosen model.
- **`generate_guidance(predict_response) -> GuidanceResult`** runs the full chain.
  It wraps generation in `safety_validator.enforce`, so a severe first answer is
  regenerated once (with a stricter reminder appended) and otherwise replaced by a
  safe fallback. The returned `safe_text` is guaranteed to pass `scan_unsafe`.
- **`GuidanceResult`** carries: `status` (pass/sanitized/fallback), `safe_text`,
  `sections`, `provenance` (source organisations for the UI), `retrieved_chunk_ids`,
  `model_cited_grounding` (validated against what was actually retrieved),
  `violations`, `query`, `model_spec`, `raw_model_text`. `.to_dict()` is API-ready.

### 2.3 The full chain
```
/predict output
  -> LLMInput.from_predict_response            (Phase 1)
  -> Retriever.retrieve_for_llm_input          (Phase 3)
  -> prompts.build_messages                    (Phase 1)
  -> model.generate            (+1 retry)      (Phase 5)
  -> safety_validator.enforce                  (Phase 4)
  -> GuidanceResult (safe + grounded + sourced)
```

---

## 3. How to run

### 3.1 Offline (no GPU) — mock model
```bash
python -m ckd_risk_app.llm.demo_phase5
```
Expected: every case `complete=True safe=True`, statuses pass/sanitized, the
emergency case sanitized, the unsafe-first case recovered, and
`Phase 5 acceptance test: PASSED`.

### 3.2 Production — Qwen3-4B in Colab
```bash
pip install -q transformers accelerate sentence-transformers faiss-cpu
# Build the real BGE-M3 + FAISS index (Phase 3):
python -m ckd_risk_app.llm.rag.embed_index
```
```python
from ckd_risk_app.llm.serving.llm_runtime import GuidanceRuntime
rt = GuidanceRuntime(model="auto")            # downloads Qwen3-4B-Instruct-2507
result = rt.generate_guidance(predict_response)   # a /predict response dict
print(result.status)
print(result.safe_text)
print("sources:", result.provenance)
```
The only difference from offline is the model; everything else is identical.

---

## 4. Verification (offline mock)

| case | status | complete | safe | grounded |
|---|---|---|---|---|
| low_clean | pass | ✓ | ✓ | ✓ |
| moderate_bp_dm | pass | ✓ | ✓ | ✓ |
| high_emergency_bp (SBP 190) | **sanitized** (urgent line added) | ✓ | ✓ | ✓ |
| moderate_weight | pass | ✓ | ✓ | ✓ |
| low_smoking | pass | ✓ | ✓ | ✓ |
| unsafe-first model | recovered → safe | — | ✓ | — |

All returned answers pass `scan_unsafe`, cite only real retrieved chunk ids, and
carry source provenance (CDC / NKF / KDIGO).

> The mock is a **plumbing** check, not a quality measure. Real answer quality is
> measured in Phase 6 with Qwen3.

---

## 5. Design decisions worth remembering

- **MVP uses the base model on purpose** — it is the control arm for the Phase 6
  go/no-go and the Phase 9 "base vs fine-tuned" ablation.
- **Generation is wrapped by the guardrail, not the other way round.** The runtime
  passes a `generate_fn` to `enforce`; safety owns retries and the final fallback,
  so the runtime can never return unsafe text.
- **One swap to go fine-tuned.** Phase 8 only changes the model (adapter path);
  retriever, prompt, validator, and runtime are untouched.
- **Grounding is validated, not trusted.** The model's claimed grounding is
  intersected with the chunks actually retrieved, so hallucinated ids are dropped.
- **Provenance ≠ grounding.** The user sees source organisations ("based on KDIGO/
  CDC"); raw chunk ids and reasoning stay internal (Phase 1 decision).

---

## 6. Known open items

1. **Run the real Qwen3 MVP in Colab** and capture a few outputs for Phase 6.
2. **Base-model format drift** is expected; `enforce` + the retry note handle it,
   but Phase 6 should record how often base Qwen3 needs a retry/fallback — that
   rate is itself evidence for fine-tuning.
3. **LLM-judge layer** can be passed to `enforce` here once a judge model is
   available (currently rule-only).
4. **No API/frontend yet** — that is Phase 10 (`/generate-guidance` + UI).

---

## 7. Next steps

**Phase 6 — MVP Evaluation (go/no-go):** run the base Qwen3 MVP on ~30 held-out
profiles; score format adherence, safety (raw vs post-validator), factor relevance,
readability, grounding. Decide whether fine-tuning is justified (it almost
certainly is, but the gaps define the training objective).

Documented in `docs/LLM_PHASE6_MVP_EVALUATION.md` when complete.

---

## 8. File map (quick reference)

```
ckd_risk_app/llm/serving/model_backends.py   # QwenChatModel / MockGuidanceModel / get_model
ckd_risk_app/llm/serving/llm_runtime.py      # GuidanceRuntime.generate_guidance -> GuidanceResult
ckd_risk_app/llm/demo_phase5.py              # python -m ckd_risk_app.llm.demo_phase5
docs/LLM_PHASE5_MVP.md                        # this file
```
