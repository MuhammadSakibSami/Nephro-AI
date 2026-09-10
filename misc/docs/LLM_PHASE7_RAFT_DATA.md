# LLM Phase 7 — RAFT Training-Data Generation

**Status:** ✅ DONE — real dataset generated in Colab (600 examples, `ckd_risk_app/phase7_data/`).
**Plan reference:** `LLM_PLAN.md` → Phase 7 (revised after Phase 6)
**Date implemented:** June 2026 (pipeline); dataset generated 2026-07-26
**Teacher (as actually run):** **100% self-distillation** (base Qwen3-4B + RAG + safety validator).
The Gemini subset was designed and implemented but **not used** in the final run — free-tier
quota made it unreliable, and the self-distilled set passed the safety gate at 1.00. See
`phase7_data/data_card.md` (`teacher mix: {'self': 600}`, Gemini share 0.0). **Report
100% self-distillation in the paper** — it is also the stronger claim, since no external
model was involved at any point in the pipeline.

---

## 1. What this phase is (in plain language)

To fine-tune Qwen3 we need training examples: *"given this kidney-risk result and
these guideline snippets, here's the ideal safe answer."* We generate them
automatically.

Phase 6 showed the base model is already strong, so we don't need an expensive
teacher to "fix" it. Instead:
- **All answers come from the model itself** (base Qwen3 + RAG + the safety
  validator) — *self-distillation*. The model learns to directly produce what
  currently needs the validator's help.
- A **Gemini (free-tier) teacher backend** is implemented and pluggable for
  harder/emergency/diverse cases, but was **not used in the final dataset**
  (`teacher_share=0.0` as run). Keep it available for a future diversity experiment.

Every answer is checked by the safety validator before it can enter the dataset,
and each example is paired with a **short prompt** so the fine-tuned model runs
cheaper.

---

## 2. Why this shape (the reframe)

From Phase 6: base+RAG+guardrail is near-publishable; the only gap is ~10% of
outputs needing the validator (emergency redirect, nutrition deferral). So the
training **target is the post-validator answer**, and the goals are:
1. **Safety self-containment** — model internalizes the rules → raw_unsafe → ~0
   without the validator.
2. **Consistency** — remove rare misses / over-specific numbers.
3. **Prompt compression** — train on a compact prompt → cheaper inference.

This keeps the fine-tuning USP with an honest, measurable objective (not a quality
overhaul the base doesn't need).

---

## 3. What was built (technical)

```
ckd_risk_app/llm/
  serving/prompts.py        # + build_compact_system_prompt() / build_compact_messages()
  data/
    profiles.py             # refactored: build_predict_profile(), make_factor() (reusable)
    train_profiles.py       # generate_train_profiles(n, seed) — training-distribution profiles
    teacher_backends.py     # SelfDistillTeacher + GeminiTeacher + get_teacher()
    generate_dataset.py     # orchestrate -> train.jsonl / val.jsonl / data_card.md
notebooks/
  phase7_generate_dataset.ipynb   # Colab: BGE-M3 index + Qwen3 self-distill + Gemini subset
```

### 3.1 Training profiles (`train_profiles.py`)
`generate_train_profiles(n, seed)` makes `n` seeded-random `/predict`-shaped
profiles, ~1/3 per risk class, with 1–4 risk-appropriate drivers (BP, glucose,
weight, kidney labs, uric acid, etc.) and paired binary flags. ~20% of high-risk
profiles use **emergency-level vitals** (SBP ≥ 180) so the model learns to self-add
emergency redirects. Distinct from the 30 frozen eval profiles and the ML test set.

### 3.2 Teacher backends (`teacher_backends.py`)
One interface — `generate_gold(inp, chunks) -> (gold, raw, status)` — both routed
through `safety_validator.enforce` (so gold is always safe + well-formed):
- **`SelfDistillTeacher`** — base chat model (Qwen3 or offline mock) + RAG + validator.
- **`GeminiTeacher`** — Google Gemini via `google-genai` (lazy import), free-tier
  friendly (configurable `request_delay_s` for RPM). Sends only synthetic profiles.
- **`get_teacher("self"|"gemini", ...)`** — pluggable; swap in a paid teacher later
  by changing one flag.

### 3.3 Dataset generation (`generate_dataset.py`)
Per profile: retrieve **golden** chunks (top-k) + **1–3 distractors** (off-topic) →
shuffle → teacher (Gemini with probability `teacher_share`, else self) → gold →
**hard gate** → Unicode-normalize → training example.

- **RAFT distractors** teach the model to ignore irrelevant retrieved context.
- **Hard gate (drop if any fail):** `scan_unsafe == 0`, all 5 sections present,
  mentions the risk level. Drop reasons are counted in the data card.
- **Each example:** `[COMPACT system, user(+golden+distractors), assistant=gold]`.
  Gold is generated under the FULL prompt but paired with the COMPACT prompt — the
  prompt-compression objective.
- Outputs `train.jsonl` / `val.jsonl` (90/10) + `data_card.md` (counts, teacher mix,
  Gemini share, pass rate, drops, sample).

---

## 4. How to run

### 4.1 Offline (no GPU/key) — validates the whole pipeline
```bash
python -m ckd_risk_app.llm.data.train_profiles      # sanity-check profile generator
python -m ckd_risk_app.llm.data.generate_dataset --n 18 --self-model mock --teacher-share 0 --out-dir phase7_data
```

### 4.2 Production (Colab) — real dataset
Open `notebooks/phase7_generate_dataset.ipynb` on a **GPU** runtime:
install → make project importable → build BGE-M3 index → paste free Gemini key →
`generate(n=600, teacher_share=0.2, self_model="qwen", ...)` → review data card →
download `train.jsonl` / `val.jsonl` / `data_card.md`.

> Tip: run `n=60` first to verify, then scale to 600. Self-distillation bulk
> (~480 gens) runs on the GPU (~40–80 min on T4); Gemini subset (~120 calls) is
> rate-limited and stays within free-tier daily limits. If you hit limits, lower
> `teacher_share` or `n` — self-distillation needs no key.

---

## 5. Offline validation result

`--n 18 --self-model mock --teacher-share 0`:
- 18/18 kept (pass rate 1.0); train 17 / val 1; 6/6/6 class mix.
- Each example: compact system prompt (~800 chars vs full ~1800), distractors
  present in the user prompt, gold parses to all 5 sections + grounding, Unicode
  normalized (no curly quotes), data card written.

(Validates plumbing only — real targets come from Qwen3/Gemini in Colab.)

---

## 6. Design decisions worth remembering

- **Targets = post-validator answers.** Self-distillation trains the model to
  produce the already-safe output directly → guardrail-independence.
- **Generate under the full prompt, train on the compact prompt.** Best target
  quality + cheaper deployment.
- **One safety gate for all teachers.** Gemini output is validated exactly like the
  base model's; nothing unsafe can enter the dataset.
- **Teacher is pluggable + optional.** 100% self-distillation works with no key;
  Gemini (free) adds diversity; a paid teacher is a one-flag swap later.
- **Privacy:** only synthetic, de-identified profiles reach Gemini; no external API
  at inference. State this + the teacher disclosure in the paper.
- **Profile separation:** training profiles ≠ the 30 eval profiles ≠ ML test set.

---

## 7. Known open items

1. **Run the Colab notebook** to produce the real `train.jsonl` / `val.jsonl`;
   commit the `data_card.md` for the paper.
2. **Manually review ~30 pairs** from the real run (safe, grounded, formatted,
   no over-specific numbers) and note the gate pass-rate.
3. **Teacher share** is at 0.2 for free-tier limits; bump later for an experiment
   if a stronger/paid teacher becomes available.
4. Consider light **dedup / diversity checks** on the real set if many profiles
   collapse to similar answers.

---

## 8. Next steps

**Phase 8 — QLoRA fine-tuning:** train Qwen3-4B (4-bit NF4 + LoRA) on `train.jsonl`
with the compact prompt, loss masked to the assistant turn; load via the existing
`QwenChatModel(adapter_path=...)` hook. Then **Phase 9** ablation: fine-tuned vs the
frozen Phase-6 base arm (raw_unsafe ↓, prompt-length/cost ↓).

Documented in `docs/LLM_PHASE8_QLORA_FINETUNE.md` when complete.

---

## 9. File map (quick reference)

```
ckd_risk_app/llm/data/train_profiles.py       # generate_train_profiles(n, seed)
ckd_risk_app/llm/data/teacher_backends.py     # SelfDistillTeacher / GeminiTeacher / get_teacher
ckd_risk_app/llm/data/generate_dataset.py     # python -m ...data.generate_dataset --help
ckd_risk_app/llm/serving/prompts.py           # build_compact_system_prompt()
notebooks/phase7_generate_dataset.ipynb       # Colab data-gen
docs/LLM_PHASE7_RAFT_DATA.md                   # this file
```
