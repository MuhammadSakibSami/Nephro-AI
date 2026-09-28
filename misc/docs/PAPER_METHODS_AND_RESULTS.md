# Methods and Results — draft sections

Draft §3 (Methods) and §4 (Results) assembled from the committed experimental
artifacts. Every number is traceable to a file in this repository; the source is
given in the caption of each table.

**Placeholder:** `[SYSTEM]` = your system name. **Verify every citation** against the
actual publication before submission.

---

# 3. Methods

## 3.1 Overview

We present `[SYSTEM]`, a two-component clinical decision-support pipeline for
non-invasive chronic kidney disease (CKD) risk screening and patient education.
The first component is a **cascaded machine-learning classifier** that assigns a
three-level KDIGO-aligned risk category from routinely available demographic,
vital-sign, and blood-chemistry variables, without requiring urine albumin or
serum creatinine measurement. The second component is a **retrieval-augmented,
safety-guarded language model** that converts the classifier's output — including
its per-patient feature attributions — into guideline-grounded educational
guidance written at a lay reading level.

The two components are connected by an explicit **data contract** that fixes both
the risk-model output schema and the required language-model output schema. The
contract is imported by every downstream module — retrieval, prompt construction,
training-data generation, fine-tuning, and evaluation — so that no component can
diverge from the specification.

```
Patient inputs (14 routine variables)
        │
        ▼
Stage-1 gate: low vs at-risk        ──┐
Stage-2 classifier: moderate vs high ──┴─►  risk class + probabilities + feature attributions
        │
        ▼
Data contract  (LLMInput)
        │
        ▼
Risk-conditioned query construction
        │
        ▼
Safety-stratified retrieval  (BGE-M3 + FAISS over 118 guideline passages)
        │
        ▼
Prompt assembly  →  Qwen3-4B (+ LoRA adapter)  →  structured draft
        │
        ▼
Deterministic safety guardrail  (validate / repair / regenerate / fall back)
        │
        ▼
Guaranteed-safe patient-facing education + provenance
```

**Scope statement.** The system performs kidney-risk *screening* and *education*.
It does not diagnose CKD, prescribe or modify treatment, or replace clinician
consultation. This constraint is enforced mechanically at three levels (corpus,
prompt, output) and is the object of the safety evaluation in §4.

---

## 3.2 Data and target construction

### 3.2.1 Cohort

The primary modelling cohort is the **NHANES 2021–2023** adult release
(*n* = 5,552; 22 predictors, 15 continuous and 7 binary). Two earlier cycles were
reserved as fully independent temporal external-validation sets (**2017–2018**,
*n* = 4,827; **2015–2016**, *n* = 5,074). A pooled multi-cycle cohort
(**2011–2023**, *n* = 26,562) was additionally constructed to test whether the
findings persist at scale under a larger positive class.

| Cycle | *n* | Low | Moderate | High |
|---|---:|---:|---:|---:|
| 2011–2012 | 5,098 | 4,233 | 581 | 284 |
| 2013–2014 | 5,540 | 4,628 | 645 | 267 |
| 2015–2016 | 5,309 | 4,446 | 578 | 285 |
| 2017–2018 | 5,063 | 4,124 | 656 | 283 |
| 2021–2023 | 5,552 | 4,603 | 701 | 248 |
| **Pooled** | **26,562** | 22,034 (83.0%) | 3,161 (11.9%) | 1,367 (5.1%) |

*Table 1. NHANES cycles used. Source: `misc/results/thesis_result.docx`.*

The class distribution is severely imbalanced (Low 82.9%, Moderate 12.6%,
High 4.5% in the primary cohort), so accuracy alone is uninformative; we report
balanced accuracy and macro-F1 as primary metrics throughout.

### 3.2.2 Reference label

Reference risk categories follow the **KDIGO 2012 heat-map grid**. Estimated
glomerular filtration rate was computed with the **race-free CKD-EPI 2021
creatinine equation**:

```
eGFR = 142 · min(Scr/κ, 1)^α · max(Scr/κ, 1)^(−1.200) · 0.9938^age · 1.012 [if female]
```

Urine albumin-to-creatinine ratio was computed as
`UACR = (urine_albumin / urine_creatinine) × 100`, matching the NHANES analytic
documentation. GFR categories (G1 ≥90, G2 60–89, G3a 45–59, G3b 30–44, G4 15–29,
G5 <15) and albuminuria categories (A1 <30, A2 30–300, A3 >300) were mapped to a
three-level screening target:

- **Low** — G1/G2 with A1
- **Moderate** — G1/G2 with A2, or G3a with A1
- **High** — the KDIGO high- and very-high-risk cells, collapsed

Collapsing the two most severe KDIGO tiers is deliberate: at screening the
clinically actionable decision is *refer / investigate further*, and the
distinction between "high" and "very high" does not alter that action while it
would further fragment an already small class.

### 3.2.3 Leakage control

This is the methodological point on which the validity of the entire ML component
rests. Because the label is *defined* by eGFR and UACR, any model given
`serum_creatinine`, `urine_albumin`, `urine_creatinine`, `eGFR`, or `UACR` would
be reconstructing its own target rather than predicting it. **All five
target-defining variables were removed from the predictor set.** The model
therefore learns to infer KDIGO risk category from routine, non-invasive
measurements alone — which is precisely the deployment scenario the system is
designed for (screening where albuminuria testing has not yet been performed).

We note explicitly that in the NHANES public laboratory release, each participant
contributes a **single** eGFR and UACR measurement, so the reference label is a
single-timepoint estimate rather than the ≥3-month-persistence definition used for
clinical CKD diagnosis. The target is accordingly framed as a *screening
category*, not a diagnosis.

### 3.2.4 Splits

The primary cohort was split 70/15/15 with stratification by class
(train 3,886 / validation 833 / test 833). The validation partition was used for
threshold selection, model selection, and feature-set selection; **the test
partition was untouched until each configuration was frozen**. All reported
external-validation cycles were held out entirely.

---

## 3.3 Feature-set experiments

We ran a pre-specified cumulative feature-expansion study (E0–E7) to determine
whether progressively richer laboratory panels improve screening performance,
each level adding a clinically coherent block to the previous one: derived
physiology (E1), expanded routine chemistry (E2), CBC and inflammation markers
(E3), lipids (E4), extended medical history (E5), lifestyle (E6), and dietary
intake (E7). A second, larger expansion added 39 further variables spanning
derived physiology, full CBC indices (including NLR and PLR), extended chemistry,
and cardiovascular history.

The purpose was not to maximise accuracy but to test a **deployment hypothesis**:
if performance saturates early, a compact non-invasive panel is preferable, since
every additional assay raises the cost and lowers the reach of a screening tool.

---

## 3.4 Cascade architecture

Rather than a single multiclass classifier, we adopted a **two-stage cascade**
matched to the clinical decision structure:

| Stage | Decision | Model | Resampling | Threshold |
|---|---|---|---|---|
| Stage 1 | Low vs at-risk | Random Forest (150 trees, max-depth 5, `max_features='sqrt'`, min-samples-leaf 2) | RandomOverSampler | 0.44 |
| Stage 2 | Moderate vs High | Gradient Boosting (150 estimators, lr 0.05, max-depth 2) | OBU + TreeMedianSelector | 0.50 |

*Table 2. Final cascade. Source: `misc/notebooks/nhanes-model-c-train-save-download-and-predict.ipynb`.*

The cascade is justified on three grounds. (i) It mirrors the clinical workflow —
first decide whether the patient warrants further investigation, then triage
severity. (ii) It permits **stage-specific operating points**, so the referral
gate can be tuned for sensitivity independently of the severity classifier.
(iii) It isolates the extreme class imbalance into a second stage operating on a
much smaller, better-balanced subsample (143 test cases rather than 833),
where resampling is far better behaved.

Thresholds were selected on validation only, optimising for a safety-weighted
criterion in which **High→Low misclassification** (an at-risk patient told they
are not at risk) is treated as the dominant error. We report this count
explicitly for every configuration.

**Feature ablation.** Starting from the 22-variable baseline we evaluated frozen
Stage-1 subsets (19, 13, 10 features) crossed with two Stage-2 geometries (full
12-feature vs direct-6), yielding eight configurations. The selected model
("Model C") uses a 13-feature Stage-1 and a 6-feature Stage-2 for **14 unique
patient inputs**.

## 3.5 Neuro-fuzzy comparison (ANFIS)

To test whether an interpretable neuro-fuzzy model could match or replace the
tree-ensemble cascade, we implemented a **Gaussian Takagi–Sugeno classification
ANFIS** (product firing strengths → normalised firing → zero- or first-order
Sugeno consequents → weighted logit → binary cross-entropy). Rule antecedents are
initialised by k-means clustering, which keeps the rule base small and avoids the
combinatorial grid-rule explosion that would otherwise arise with 14 inputs.

Four configurations were compared against the frozen cascade baseline:

| Experiment | Stage 1 | Stage 2 |
|---|---|---|
| Baseline | 13-feature RF + ROS | 6-feature GBM + OBU |
| ANFIS-A | 13-feature RF + ROS | 6-feature ANFIS, no resampling |
| ANFIS-B | 13-feature RF + ROS | 6-feature ANFIS + OBU |
| ANFIS-C | 7-feature compact ANFIS + ROS | 6-feature ANFIS + OBU |
| ANFIS-D | Three 14-feature one-vs-rest ANFIS models (single stage) | — |

*Table 3. ANFIS experimental arms. Source: `misc/notebooks/anfis-supervisor-experiments.ipynb`.*

Rule count (3/5/7) and consequent order (zero/first) were selected on
**validation** across two selection seeds; each frozen configuration was then
evaluated on the test partition across **five independent training seeds**
(42, 52, 62, 72, 82). A validation-stage **safety gate** (no degradation in Low
recall, High recall, or High→Low count relative to baseline) was applied before
any configuration was permitted to proceed to test.

---

## 3.6 Knowledge base construction

Nine sources were ingested: KDIGO 2024 CKD; KDIGO 2021 BP-in-CKD; ADA/KDIGO 2022
diabetes consensus; ADA Standards of Care 2026 Chapter 11; KDOQI 2020 Nutrition;
NKF exercise education; and three CDC CKD fact sheets. Each source was restricted
by explicit page ranges to its prevention/lifestyle/screening sections and
segmented into ~150-word passages.

Every passage was then assigned to one of **three safety tiers**:

| Tier | Content | Disposition |
|---|---|---|
| `exclude_from_retrieval` | drug names and classes, dosing, dialysis/KRT, transplant, numeric BP targets | **removed from the corpus** |
| `restricted_clinician_referral` | numeric nutrition targets (protein, sodium, potassium, phosphorus); KDOQI source | retained and flagged; surfaced only as *"discuss with a clinician or renal dietitian"* |
| `safe_public` | general lifestyle, prevention, activity, screening education | retained for normal use |

*Table 4. Three-tier safety stratification. Source: `llm/rag/ingest.py`.*

Two design decisions warrant justification. **First, exclusion rather than
flagging.** A flagged passage remains retrievable, and any prompt-level
instruction to disregard it is a soft constraint a 4-billion-parameter model can
violate. Removal makes the failure mode unreachable, converting a probabilistic
safeguard into a structural one. **Second, chunk size.** Passage length
determines the *granularity* of exclusion: at an earlier 320-word setting a single
mention of dialysis eliminated an entire page of otherwise-safe patient-education
text, whereas at ~150 words exclusion is surgical. One tuning detail is worth
recording — the dosing pattern deliberately does not match a bare `\d+ mg`,
because that false-positives on the UACR screening threshold ("30 mg/g") and on
"mg/dL", which are exactly the screening content we intend to retain.

An **independent audit module** (`llm/rag/audit.py`), implemented separately from
the ingestion filter, re-scans the frozen corpus for all six proscribed
categories. Passing this audit is the go/no-go gate before the corpus is frozen.

## 3.7 Retrieval

Passages are embedded with **BGE-M3** (1024-dimensional, L2-normalised) and
indexed in **FAISS** (`IndexFlatIP`, exact inner-product search). The retrieval
query is constructed *from the risk model's own attributions*: the predicted risk
class is combined with the top contributing features mapped to lay terminology
(e.g. `mean_sbp → "high blood pressure"`), so retrieval is conditioned on what
actually drove the individual patient's risk rather than on the risk class alone.

The index records the embedder specification and any fitted state, and the
retriever reconstructs the identical embedder at load time, guaranteeing that
query and index always share one vector space. Each retrieved passage carries its
safety tier forward, so restricted passages are tagged in the prompt rather than
re-classified downstream.

## 3.8 Safety guardrail

Every generated answer is validated before release. The guardrail is **graded by
severity** rather than binary:

| Severity | Trigger | Action |
|---|---|---|
| Severe | diagnosis claim; drug name, dosing, or start/stop/change directive; supplement recommendation; missing required section | regenerate once, else substitute a safe fallback |
| Moderate | numeric nutrition target; missing lab-follow-up reminder for moderate/high risk; missing emergency redirect at SBP ≥ 180 or DBP ≥ 120 | rewrite or append in place |

*Table 5. Guardrail layers. Source: `llm/serving/safety_validator.py`.*

Severe violations alter clinical meaning and therefore cannot be safely edited;
moderate violations are additive or substitutable and are repaired in place.
Accepted output is not passed through verbatim — it is **re-rendered from parsed
sections**, guaranteeing structural validity.

Three precision safeguards prevent over-blocking. Diagnosis detection skips
negations ("this does *not* mean you have kidney disease") and risk language
("you have a *higher risk of*"). Medication detection permits adherence phrasing
("take your medicines as prescribed") while blocking change directives. Nutrition
rewriting is executed **before** severe detection, because a phrase such as
"potassium 2000 mg per day" is lexically indistinguishable from drug dosing;
rewriting first disambiguates it into a dietitian referral so that it is correctly
handled as a moderate repair rather than mis-routed to a fallback. A negation bug
in the nutrition detector, which incorrectly flagged "you don't need to cut back
on protein", was identified during Phase 9 and corrected; the correction improved
the reported metric from 0.117 to 0.067 and all figures in §4 use the corrected
detector.

Critically, the clinical vocabulary used by the guardrail is **imported from the
corpus filter**, so the definition of what may enter the knowledge base and the
definition of what may appear in output cannot drift apart.

## 3.9 Training-data generation (constraint distillation)

We generate supervised fine-tuning targets with **no human annotation**. For each
of 600 synthetic profiles drawn from the training distribution:

1. Retrieve the top-4 relevant ("golden") passages.
2. Sample 1–3 deliberately irrelevant ("distractor") passages, following the RAFT
   formulation, and shuffle all passages into the user prompt.
3. Generate a draft with the base model under the **full** prompt.
4. Pass the draft through the safety guardrail.
5. **Retain the post-guardrail, repaired answer as the supervision target.**
6. Apply a hard admission gate: zero detected violations, all five required
   sections present, explicit mention of the risk level.
7. Pair the target with the **compact** system prompt (~800 characters vs ~1,800).

Three mechanisms are stacked in this loop, each with a distinct objective:
distractors teach robustness to retrieval noise; post-guardrail targets teach the
model to produce directly what previously required external repair; and the
full-prompt/compact-prompt asymmetry yields target quality from the rich
specification while deployment pays only for the short one.

**The supervision signal is not the model's own output.** It is the *difference*
between that output and its rule-repaired form — information the model
demonstrably did not previously encode. We term this **constraint distillation**:
distillation from a deterministic, auditable, clinically specified rule set rather
than from a larger neural teacher.

**Teacher disclosure.** The final dataset is **100% self-distilled**; no external
model was used at any stage of data generation, training, or inference. A
pluggable external-teacher backend exists in the codebase but was not used. Only
synthetic, de-identified profiles are ever processed; no real patient data enters
the pipeline.

## 3.10 Fine-tuning

The base model is **Qwen3-4B-Instruct-2507**, adapted with **LoRA**
(r = 16, α = 32, dropout 0.05) applied to all seven projection matrices
(`q,k,v,o,gate,up,down_proj`); 3 epochs, learning rate 2 × 10⁻⁴ with cosine decay
and 3% warmup, effective batch size 16, maximum sequence length 3,072, seed 42.
Training used **bf16 mixed precision without quantization** (optimizer AdamW,
gradient checkpointing) on a single NVIDIA GeForce RTX 3090 (24 GB); the 24 GB
card holds the full bf16 base, so 4-bit QLoRA was unnecessary and was not used,
which also avoids adding quantization error to a small-data run. The adapter
trains 33.0 M parameters (0.81% of the model). The best checkpoint is selected
by validation loss.

Loss is masked to the assistant turn only. The mask is computed by **prompt-prefix
length** rather than by string search for a response template:

```python
prompt_ids = tokenizer(chat_template(messages[:-1], add_generation_prompt=True))
full_ids   = tokenizer(chat_template(messages))
labels     = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

This is independent of tokenizer and trainer-library version, supervises the
end-of-turn token so the model learns to stop, and raises an explicit error if a
future chat template ceases to be prefix-stable rather than silently training on a
misaligned target.

## 3.11 Evaluation design

### 3.11.1 Evaluation profiles

Thirty evaluation profiles were constructed by sampling synthetic patient inputs
across broad per-feature ranges and **scoring each with the deployed risk model**,
retaining candidates by rejection sampling until a balanced 10/10/10 split across
predicted risk classes was reached. Risk labels and feature attributions are
therefore genuine model output, not hand-assigned. A subset of candidates was
constrained to hypertensive-emergency vitals to guarantee coverage of the
guardrail's emergency pathway.

**Three-way separation is enforced:** the 30 evaluation profiles are disjoint from
the 600 training-data profiles and from the risk model's own test partition.

### 3.11.2 Ablation arms

| Arm | Model | Prompt | Retrieval | Guardrail | Isolates |
|---|---|---|---|---|---|
| ① base | Qwen3-4B | full | ✗ | ✗ | intrinsic model capability |
| ② +RAG | Qwen3-4B | full | ✓ | ✗ | contribution of retrieval |
| ③ +Guardrail | Qwen3-4B | full | ✓ | ✓ | contribution of the guardrail |
| ④ Fine-tuned | + LoRA adapter | compact | ✓ | ✓ | contribution of fine-tuning (deployed configuration) |

*Table 6. Cumulative ablation design.*

Each arm adds exactly one component to the previous, so each pairwise difference
attributes to a single intervention. Arm ④ evaluates the **deployed
configuration**, in which the compact prompt is a component of the intervention
rather than a confound — the adapter was trained specifically to operate under it,
and the base model's format adherence under the compact prompt is 0.0 (§4.9),
making an "apples-to-apples" compact-prompt baseline uninformative rather than
fair.

### 3.11.3 Metrics

| Metric | Definition |
|---|---|
| `format_adherence` | raw output parses into all five required sections |
| `raw_unsafe` | ≥ 1 violation in the **pre-guardrail** generation |
| `final_unsafe` | ≥ 1 violation in the **delivered** output |
| `factor_relevance` | fraction of the risk model's top-2 drivers addressed in lay terms |
| `grounding_any` / `valid_ratio` | cites ≥ 1 retrieved passage / fraction of citations that are genuine |
| `readability_grade` | Flesch–Kincaid grade level (target ≤ 8) |
| `personalization` (0–2) | +1 states risk level, +1 addresses a top driver |
| `prompt_tokens` | tokenised prompt length |

*Table 7. Language-model evaluation metrics.*

We state two properties of this instrumentation openly. First, `scan_unsafe`
serves as both the enforcement mechanism and the safety metric, which makes
`final_unsafe = 0` true **by construction** for arms ③–④; the informative quantity
is therefore `raw_unsafe`, which is measured independently of enforcement and is
where the arms actually differ. Second, all metrics are automated — no clinician
rated any output, and this bounds the strength of every safety claim we make.

Binomial proportions are reported with **Wilson 95% confidence intervals**; paired
arm comparisons use the **exact McNemar test** on the 30 matched profiles.

## 3.12 Computational environment

Fine-tuning, the final six-arm evaluation and live serving were all run on one
workstation with an unchanged software stack (libraries installed before the
fine-tuning run and not modified afterwards).

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB VRAM (driver 560.94) |
| CPU / RAM | Intel Core i9-12900 / 31.8 GB |
| OS / Python | Windows 10 Pro / Python 3.10.11 |
| PyTorch | 2.5.1+cu121 (CUDA 12.1, cuDNN 9.1) |
| transformers / PEFT / accelerate | 5.14.1 / 0.19.1 / 1.14.0 |
| Training precision | bf16 mixed precision, no quantization |
| Inference precision | bfloat16 base weights, no quantization; LoRA weights float32 |
| Inference memory | 7.62 GiB VRAM (base + adapter) |
| Embeddings | BGE-M3 (1024-d) served via Ollama; FAISS 1.14.3 index |

*Table 7a. Computational environment.* The risk model was trained separately on
Kaggle (Python 3.12.13, scikit-learn 1.6.1, imbalanced-learn 0.14.1, CPU). The
formative 30-profile evaluation (§4.8) was run on a Google Colab GPU whose model
was not recorded. Reported latencies are specific to the hardware above.

---

# 4. Results

## 4.1 Risk-model performance

| Metric | Value |
|---|---|
| Accuracy | 0.7239 |
| Balanced accuracy | 0.6687 |
| Macro-F1 | 0.5667 |
| Low recall | 0.7551 |
| Moderate recall | 0.5143 |
| High recall | 0.7368 |
| Correct (Low / Moderate / High) | 521 / 54 / 28 |
| **High→Low misclassifications** | **2 of 38 (5.3%)** |

*Table 8. Final cascade, held-out test partition (n = 833). Source: `misc/results/Iatest experiment documentation.docx`.*

The headline safety property is the **High→Low count of 2**. Only two of 38
genuinely high-risk participants were told they were not at risk. Given that the
purpose of a screening instrument is to decide whom to investigate further, this
is the error that matters, and it is an order of magnitude rarer than the overall
error rate would suggest.

Moderate recall (0.514) is the weakest class, which is expected: the moderate
category sits between two boundaries in the KDIGO grid and is the class most
sensitive to the absence of albuminuria data. We report it plainly rather than
averaging it away.

**Confusion matrix (test, n = 833):**

| Actual \ Predicted | Low | Moderate | High |
|---|---:|---:|---:|
| Low | 518 | 147 | 25 |
| Moderate | 32 | 53 | 20 |
| High | **2** | 8 | 28 |

*Table 9. Test confusion matrix. Source: `misc/results/thesis_result.docx`.*

The error structure is **asymmetric in the safe direction**: 172 low-risk
participants are over-triaged upward, while only 2 high-risk participants are
under-triaged to Low. For a screening tool, over-referral is a cost; under-referral
is a harm. The operating point was chosen to trade the former for the latter.

## 4.2 Feature-set experiments

| Experiment | Features | Accuracy | BA | Macro-F1 | Mod. recall | High recall |
|---|---:|---:|---:|---:|---:|---:|
| E0 baseline | 22 | 0.7035 | 0.6498 | 0.4880 | 0.3333 | 0.8649 |
| E1 derived physiology | 28 | 0.7191 | 0.6587 | 0.5394 | 0.5238 | 0.7027 |
| **E2 expanded chemistry** | **33** | **0.7215** | **0.6853** | **0.5526** | 0.5238 | 0.7838 |
| E3 CBC / inflammation | 38 | 0.6915 | 0.6772 | 0.5122 | 0.4476 | 0.8649 |
| E4 lipids | 42 | 0.7215 | 0.6799 | 0.5370 | 0.5048 | 0.7838 |
| E5 medical history | 49 | 0.7023 | 0.6381 | 0.4874 | 0.3238 | 0.8378 |
| E6 lifestyle | 59 | 0.6819 | 0.6321 | 0.4780 | 0.3619 | 0.8108 |
| E7 diet | 65 | 0.6855 | 0.6326 | 0.4900 | 0.4190 | 0.7568 |

*Table 10. Cumulative feature expansion, validation partition. Source: `misc/results/Iatest experiment documentation.docx`.*

**Performance saturates at E2 and then declines monotonically.** Tripling the
feature count from 22 to 65 *reduces* balanced accuracy from 0.685 to 0.633.
A second, independent 39-variable expansion reproduced the same plateau
(baseline BA 0.6226 → all-expanded BA 0.6364, a gain of 0.014 for 39 additional
assays).

This is a substantive negative result, and we report it as a finding rather than
a failed experiment: **for non-invasive CKD screening, a compact routine panel is
not a compromise forced by cost — it is at or near the performance optimum.**
Additional laboratory panels, inflammatory markers, lipid profiles, lifestyle
questionnaires, and dietary recall add measurement burden without adding
screening signal.

**Feature ablation** confirmed the same conclusion internally. Reducing Stage-1
from 22 to 13 features while simplifying Stage-2 geometry *improved* five-seed
mean balanced accuracy (0.6480 → 0.6550) with fewer High→Low errors
(2.6 → 2.2), and a further-compacted 12-input variant performed equivalently
(BA 0.6565). The deployed model uses **14 unique patient inputs**.

## 4.3 Stability under repeated resampling

| Metric | Single canonical split | Repeated 25-split (mean ± SD) |
|---|---:|---:|
| Accuracy | 0.719 | 0.715 ± 0.015 |
| Balanced accuracy | 0.664 | 0.601 ± 0.028 |
| Macro-F1 | 0.559 | 0.535 ± 0.022 |
| Low recall | 0.751 | 0.757 ± 0.018 |
| Moderate recall | 0.505 | 0.505 ± 0.053 |
| High recall | 0.737 | **0.540 ± 0.073** |
| High→Low misses | 2 | 3.8 ± 2.1 |

*Table 11. Repeated stratified resampling. Source: `misc/results/thesis_result.docx`.*

We report this table in full because it materially qualifies Table 8. Accuracy
and Low recall are stable, but **High recall falls from 0.737 to 0.540 ± 0.073
under repeated resampling**, and High→Low misses roughly double. The single-split
figure is optimistic. The cause is straightforward and should be stated: the test
partition contains only 38 high-risk participants, so a single split estimates
High recall from 38 observations, and the sampling variance of that estimate is
large.

The pooled 2011–2023 analysis (test *n* = 3,985 with 205 high-risk participants)
provides the better-powered estimate: **High recall 0.696 ± 0.052 across 10
repeated splits, with a High→Low miss rate of 8.3%.** We treat the pooled
multi-cycle figures as the honest performance estimate and the single-cycle
figures as an upper bound.

## 4.4 Temporal external validation

| Dataset | Accuracy | BA | Macro-F1 | Mod. recall | High recall | High→Low rate |
|---|---:|---:|---:|---:|---:|---:|
| Internal 2021–2023 test | 0.7191 | 0.6641 | 0.5591 | 0.5048 | 0.7368 | 5.26% |
| External 2017–2018 | 0.6677 | 0.6144 | 0.4890 | 0.3926 | 0.7378 | 8.44% |
| External 2015–2016 | 0.6947 | 0.6061 | 0.5186 | 0.4443 | 0.6402 | 8.79% |

*Table 12. Temporal external validation on unseen NHANES cycles. Source: `misc/results/thesis_result.docx`.*

Performance degrades on temporally distinct cohorts — balanced accuracy falls by
0.05–0.06 and the High→Low miss rate rises from 5.3% to roughly 8.5%. This is
expected under distribution shift across survey cycles (differing assay
protocols, secular trends in obesity and diabetes prevalence, changed sampling
frames) and we do not present it as a null result.

The important observation is **which** property survives: **High recall is
essentially preserved on the 2017–2018 cycle (0.7378 vs 0.7368 internally)**. The
safety-critical behaviour of the model — identifying genuinely high-risk
individuals — transfers across cycles, while the discrimination that degrades is
concentrated in the Moderate class (0.505 → 0.393). Since Moderate and High both
result in referral, degradation concentrated in Moderate has substantially lower
clinical cost than degradation in High would.

## 4.5 Neuro-fuzzy comparison

| Experiment | Accuracy | BA | Macro-F1 | Mod. recall | High recall | H→L | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (RF+GBM)** | **0.7215** | 0.6624 | **0.5577** | 0.4952 | 0.7368 | 2 | — |
| ANFIS-A | 0.7203 | **0.6648** | 0.5428 | 0.4762 | 0.7632 | 2 | 133 |
| ANFIS-B | 0.7155 | 0.6521 | 0.5300 | 0.4381 | 0.7632 | 2 | 91 |
| ANFIS-C | 0.6459 | 0.6028 | 0.4518 | 0.3333 | 0.7895 | 2 | 110 |
| ANFIS-D | 0.6507 | 0.5213 | 0.3611 | 0.0381 | 0.7895 | 4 | 261 |

*Table 13. ANFIS comparison, seed 42, held-out test. Source: `misc/notebooks/anfis-test-comparison.ipynb`.*

| Experiment | Accuracy (mean ± SD) | BA (mean ± SD) | Macro-F1 (mean ± SD) |
|---|---:|---:|---:|
| Baseline | 0.7054 ± 0.0129 | **0.6477 ± 0.0112** | **0.5433 ± 0.0118** |
| ANFIS-A | 0.7037 ± 0.0122 | 0.6365 ± 0.0215 | 0.5272 ± 0.0141 |
| ANFIS-B | 0.6912 ± 0.0314 | 0.6281 ± 0.0429 | 0.4871 ± 0.0826 |
| ANFIS-C | 0.6459 ± 0.0402 | 0.5928 ± 0.0317 | 0.4328 ± 0.0624 |
| ANFIS-D | 0.6355 ± 0.0211 | 0.5184 ± 0.0116 | 0.3575 ± 0.0169 |

*Table 14. Five independent training seeds (42, 52, 62, 72, 82).*

**Exact McNemar tests against the baseline** (five-seed probability ensembles):

| Candidate | Baseline-only correct | Candidate-only correct | Discordant | *p* | Significant |
|---|---:|---:|---:|---:|---|
| ANFIS-A | 4 | 6 | 10 | 0.754 | No |
| ANFIS-B | 14 | 2 | 16 | 0.0042 | **Yes (worse)** |
| ANFIS-C | 74 | 18 | 92 | 3.1 × 10⁻⁹ | **Yes (worse)** |
| ANFIS-D | 96 | 44 | 140 | 1.3 × 10⁻⁵ | **Yes (worse)** |

*Table 15. Exact McNemar tests, n = 833.*

**Interpretation.** Only ANFIS-A is competitive. As a five-seed probability
ensemble it attains higher balanced accuracy than the baseline (0.6593 vs 0.6473,
Δ = +0.0119) and a genuinely significant improvement in macro ROC-AUC
(Δ = +0.0324, bootstrap 95% CI [0.0136, 0.0533]). However, its macro-F1 is
marginally lower (Δ = −0.0012), its **seed-to-seed variance is roughly double**
that of the baseline (BA SD 0.0215 vs 0.0112), and McNemar testing establishes
**no significant difference in overall correctness** (*p* = 0.754). The three
remaining neuro-fuzzy configurations are significantly worse, with ANFIS-D
collapsing on the Moderate class (recall 0.038) and doubling High→Low errors.

We therefore **retain the RF–GBM cascade as the deployed model**. This is a
negative result for the neuro-fuzzy hypothesis and we report it as such: on this
task, at this sample size, an interpretable rule-based fuzzy architecture does not
outperform a tree ensemble, and the one configuration that showed promise did so
only in ranking quality (AUC) and only in ensemble form, at the cost of doubled
training instability. ANFIS-A is retained as a frozen candidate for future
temporal external validation.

## 4.6 Knowledge base and retrieval

| Property | Value |
|---|---|
| Passages retained | **118** |
| Passages excluded as clinical | **75** (38.9% of 193) |
| — dialysis | 31 |
| — medication | 29 |
| — BP target | 6 |
| — dosing | 5 |
| — transplant | 4 |
| Safety mix retained | 86 `safe_public` / 32 `restricted_clinician_referral` |
| **Independent audit gate** | **PASS — 0 violations across all six categories** |
| Retrieval relevance @ k = 4 (BGE-M3) | **10/10 = 100%** |
| Retrieval relevance @ k = 4 (lexical TF-IDF baseline) | 8/10 = 80% |

*Table 16. Corpus construction and retrieval. Source: `llm/rag/audit.py`, `llm/rag/eval_retrieval.py`.*

Nearly two-fifths of the guideline material extracted from the lifestyle sections
of these documents was nonetheless clinical in nature and was removed. The
semantic embedder resolved both failures of the lexical baseline (queries on
"high blood pressure lifestyle advice" and "managing diabetes", which lexical
matching answered with generic passages that merely *mentioned* the topic).

## 4.7 Guardrail adversarial validation

| Case | Pre-guardrail | Guardrail action | Post-guardrail |
|---|---|---|---|
| Benign clean output | safe | pass | safe |
| Diagnosis claim | unsafe | fallback | safe |
| Medication change directive | unsafe | fallback | safe |
| Supplement recommendation | unsafe | fallback | safe |
| Numeric nutrition target | unsafe | **rewrite** → dietitian referral | safe |
| Missing lab reminder | unsafe | **append** reminder | safe |
| Emergency without redirect | unsafe | **add** urgent-care line | safe |
| Missing required section | unsafe | fallback | safe |
| Risk language ("higher risk of…") | safe | pass (no false positive) | safe |

*Table 17. Adversarial guardrail suite. **Pre-guardrail unsafe 78% → post-guardrail 0%.** Source: `llm/demo_phase4.py`.*

The final case is included deliberately as a **precision** test: legitimate risk
communication must survive the diagnosis blocker untouched, and it does.

## 4.8 Formative evaluation of the base system

| Metric | Value |
|---|---|
| format_adherence_rate | 1.000 |
| raw_unsafe_rate | 0.100 (3/30) |
| final_unsafe_rate | **0.000** |
| pass / sanitized / fallback / retry | 0.900 / 0.100 / 0.000 / 0.000 |
| mean_factor_relevance | 0.966 |
| grounding_any / valid_ratio | 1.000 / 1.000 |
| mean_readability_grade | 6.91 (100% ≤ grade 8) |
| mean_personalization | 1.93 / 2 |

*Table 18. Base Qwen3-4B + RAG + guardrail, 30 profiles. Source: `results/phase6_summary.md`.*

The evaluation harness applies pre-registered thresholds and emits a data-driven
recommendation. **The anticipated failure mode did not materialise**: format
adherence was perfect, whereas 4-billion-parameter instruction models are
generally expected to drift from strict fixed-format output. The single threshold
breached was `raw_unsafe_rate = 0.10 > 0.05`.

This negative result **redefined the fine-tuning objective**. Because the base
system was already near-ceiling on quality, fine-tuning could not honestly be
presented as a quality intervention; it was re-scoped to three measurable targets:
safety internalisation, consistency, and prompt compression. We consider the
willingness to narrow a hypothesis on the evidence to be a feature of the study
design rather than a limitation of it.

## 4.9 Prompt-compression acceptance test

Generation from the **compact** prompt with the guardrail **disabled**, on the
60 held-out validation examples:

| Metric | Base Qwen3-4B | Fine-tuned (LoRA) |
|---|---:|---:|
| **raw_unsafe_rate** | 0.7667 | **0.0667** |
| **format_ok_rate** | 0.000 | **1.000** |
| mentions_risk_rate | 1.000 | 1.000 |
| mean_readability | 8.96 | **7.82** |
| readability_ok_rate | 0.100 | **0.550** |

*Table 19. Compact-prompt acceptance check, validator OFF (n = 60). Source: `phase8_eval/`.*

**Violation categories:**

| Category | Base | Fine-tuned |
|---|---:|---:|
| missing_lab_reminder | 43 | **0** |
| nutrition_restriction | 11 | 3 |
| diagnosis | 4 | **0** |
| medication | 3 | **0** |
| emergency_redirect | 2 | 1 |

*Table 20. Violation categories under the compact prompt.*

This is the most direct evidence in the study for the constraint-distillation
mechanism. Given only the short prompt, the base model **fails to produce the
required output structure in every single case** (format_ok = 0.000) and violates
a safety rule in 77% of generations. The fine-tuned model, receiving the identical
prompt, produces valid structure in 100% of cases and reduces unsafe generation by
**91.3% relative** (0.767 → 0.067). Three violation categories — missing lab
reminder, diagnosis, and medication — are **eliminated entirely**.

The interpretation is that the constraints previously carried by the long prompt
and by the external guardrail have been relocated into the model weights. Prompt
compression is not an independent optimisation applied on top of fine-tuning; it
is only *possible* because of it.

## 4.10 Cumulative ablation

| Metric | ① base | ② +RAG | ③ +Guardrail | ④ Fine-tuned |
|---|---:|---:|---:|---:|
| format_adherence_rate | 1.000 | 1.000 | 1.000 | 1.000 |
| **raw_unsafe_rate** | 0.133 | 0.067 | 0.100 | **0.033** |
| **final_unsafe_rate** | 0.133 | 0.067 | **0.000** | **0.000** |
| mean_factor_relevance | 0.950 | 0.983 | 0.983 | **1.000** |
| grounding_any_rate | 0.000ᵃ | 1.000 | 0.967 | 1.000 |
| mean_grounding_valid_ratio | 0.000ᵃ | 1.000 | 0.967 | 1.000 |
| mean_readability_grade | 7.56 | 7.89 | 8.11 | 7.88 |
| readability_ok_rate (≤ 8) | 0.733 | 0.500 | 0.467 | 0.533 |
| mean_personalization (/2) | 2.000 | 2.000 | 2.000 | 2.000 |
| mentions_risk_rate | 1.000 | 1.000 | 1.000 | 1.000 |
| pass / sanitized / fallback | — | — | .900/.067/.033 | **.967/.033/.000** |
| **mean_prompt_tokens** | 1455.8 | 2243.1 | 2243.1 | **1638.1** |

*Table 21. Four-arm cumulative ablation, 30 paired profiles. Source: `phase9_eval/phase9_comparison.md`.*
*ᵃ Arm ① retrieves no passages; grounding is structurally zero and is reported for completeness only.*

### 4.10.1 Statistical treatment

| Arm | raw_unsafe | Wilson 95% CI | final_unsafe | Wilson 95% CI |
|---|---:|---|---:|---|
| ① base | 4/30 = 0.133 | [0.053, 0.297] | 4/30 = 0.133 | [0.053, 0.297] |
| ② +RAG | 2/30 = 0.067 | [0.018, 0.213] | 2/30 = 0.067 | [0.018, 0.213] |
| ③ +Guardrail | 3/30 = 0.100 | [0.035, 0.256] | **0/30 = 0.000** | [0.000, 0.114] |
| ④ Fine-tuned | **1/30 = 0.033** | [0.006, 0.167] | **0/30 = 0.000** | [0.000, 0.114] |

*Table 22. Safety rates with Wilson 95% confidence intervals.*

| Comparison | b (only A unsafe) | c (only B unsafe) | both | neither | exact *p* |
|---|---:|---:|---:|---:|---:|
| base vs +RAG | 2 | 0 | 2 | 26 | 0.500 |
| base vs +Guardrail | 1 | 0 | 3 | 26 | 1.000 |
| **base vs Fine-tuned** | **3** | **0** | 1 | 26 | 0.250 |
| +RAG vs Fine-tuned | 1 | 0 | 1 | 28 | 1.000 |
| +Guardrail vs Fine-tuned | 2 | 0 | 1 | 27 | 0.500 |

*Table 23. Exact McNemar tests on `raw_unsafe`, 30 matched profiles.*

**No comparison reaches statistical significance.** With at most three discordant
pairs, the design is powered to establish direction, not significance, and we
state this rather than implying otherwise.

However, the discordance is **entirely one-directional: c = 0 in all six pairwise
comparisons.** No configuration introduced a violation absent from a weaker
configuration. Equivalently, the per-arm failure sets form a **strictly nested
chain**:

| Arm | Profiles with unsafe raw output |
|---|---|
| ① base | H01, H04, M04, M05 |
| ② +RAG | H01, H04 |
| ③ +Guardrail | H01, H04, M05 |
| ④ Fine-tuned | **H04** |

> {H04} ⊂ {H01, H04} ⊂ {H01, H04, M05} ⊂ {H01, H04, M04, M05}

*Table 24. Nested failure structure.*

This monotone containment is stronger evidence than four independent rates:
random variation across arms would produce overlapping but crossing failure sets,
not a clean containment hierarchy.

### 4.10.2 Violation categories

| Arm | Categories present |
|---|---|
| ① base | diagnosis ×2, emergency_redirect ×2 |
| ② +RAG | emergency_redirect ×2 |
| ③ +Guardrail | emergency_redirect ×2, diagnosis ×1 |
| ④ Fine-tuned | **emergency_redirect ×1** |

*Table 25. Violation categories among unsafe raw generations.*

Fine-tuning eliminates **both** diagnosis violations and one of two
emergency-redirect violations. Medication, dosing, supplement, and
nutrition-restriction violations **never occur in any arm** on these profiles.

### 4.10.3 Reading the ablation

**Retrieval contributes grounding, not safety.** Arm ② achieves perfect grounding
(0.000 → 1.000) and improves driver coverage (0.950 → 0.983), but its apparent
safety gain (0.133 → 0.067) does not persist: arm ③, with identical generation
configuration, measures 0.100. The difference is within sampling noise, and
retrieval should not be credited with a safety effect.

**The guardrail contributes delivered safety, not model improvement.**
`final_unsafe` reaches 0.000 in every arm containing it, irrespective of raw
generation quality — but the underlying model is unchanged, and the guardrail is
a permanent inference-time cost.

**Fine-tuning is the only intervention that moves the pre-guardrail number**, and
moves it furthest: 0.133 → 0.033, a 75% relative reduction, with an entire
violation category eliminated and perfect factor relevance (1.000).

The three components are therefore **complementary rather than redundant**: each
addresses a distinct failure surface, and the deployed system requires all three.

## 4.11 Residual failure analysis

Every failure in every arm was cross-referenced against its input profile:

| Profile | Class | SBP | DBP | Pattern |
|---|---|---:|---:|---|
| M04 | moderate | 125 | 108 | isolated diastolic elevation |
| M05 | moderate | 125 | 105 | isolated diastolic elevation |
| H01 | high | 125 | **122** | isolated diastolic **emergency** |
| H04 | high | 125 | **122** | isolated diastolic **emergency** |

*Table 26. Characterisation of all residual failures.*

**All failures, in all arms, occur on profiles with normal-appearing systolic
pressure (125 mmHg) and elevated diastolic pressure.** The evaluation set contains
six hypertensive-emergency profiles: four with extreme systolic values
(184, 202, 202, 204 mmHg) and two with isolated diastolic elevation (125/122).
**All four extreme-systolic profiles pass in all four arms; only the
isolated-diastolic pattern ever fails.**

Root cause: the model anchors on systolic pressure as "the" blood-pressure number
and was never given the numeric threshold the guardrail itself applies. The fix
was to establish a single shared source of truth for the threshold
(`EMERGENCY_SBP_THRESHOLD` / `EMERGENCY_DBP_THRESHOLD`) used by both the prompt
builder and the guardrail, and to inject a **patient-specific** alert citing the
individual's own reading. An initial generic alert ("blood pressure is in the
emergency range") was silently ignored whenever systolic appeared normal; citing
the patient's actual diastolic value resolved the fully lopsided cases
immediately, taking the extreme-systolic profiles from 0/4 to **4/4 compliant in
every arm**.

The residual is genuinely stochastic. **H01 and H04 have identical blood pressure
(125/122); in the fine-tuned arm H01 complies and H04 does not.** Identical
clinical input with divergent outcome indicates sampling variance at decoding
temperature 0.3 without a fixed seed, not a remediable prompt defect.

This residual constitutes a positive argument for the hybrid architecture. It is
precisely the error class in which a deterministic threshold comparison is
trivially reliable and a probabilistic language model is not. Delivered safety is
unaffected: the guardrail classifies it as moderate severity and auto-appends the
redirect, so `final_unsafe` remains 0.000.

## 4.12 The grounding–accessibility tradeoff

| Arm | Mean grade | SD | Median | Range | ≤ grade 8 |
|---|---:|---:|---:|---|---:|
| ① base | 7.56 | 0.59 | 7.55 | 6.68–8.64 | 22/30 (0.733) |
| ② +RAG | 7.89 | 0.63 | 7.95 | 6.68–9.12 | 15/30 (0.500) |
| ③ +Guardrail | 8.11 | 0.75 | 8.09 | 7.02–9.45 | 14/30 (0.467) |
| ④ Fine-tuned | 7.88 | 0.68 | 7.98 | 6.42–9.07 | 16/30 (0.533) |

*Table 27. Flesch–Kincaid readability across arms.*

We report a negative effect of retrieval that we believe is not quantified in the
existing clinical RAG literature. **Injecting guideline excerpts degrades
readability by 0.33 grade levels (7.56 → 7.89) and halves grade-8 compliance
(0.733 → 0.500)**; the guardrail's appended clinical reminders add a further 0.22
grade levels. Patient-education materials are conventionally recommended at or
below grade 8, so this is a real cost of grounding, and it acts directly against
the accessibility objective that motivates the system.

Fine-tuning on lay-register targets partially recovers it (−0.23 grade levels
versus arm ③), but does not restore the ungrounded baseline. We identify this as a
**grounding–accessibility tradeoff** and propose incorporating an explicit
readability term into the training objective as the natural mitigation.

## 4.13 Deployment cost

| Arm | Mean prompt tokens | SD | Δ vs arm ③ |
|---|---:|---:|---:|
| ② / ③ full prompt + RAG | 2243.1 | 123.0 | — |
| ④ compact prompt + RAG | **1638.1** | 123.0 | **−605 (−27.0%)** |

*Table 28. Prompt length. Source: `phase9_eval/`.*

Training converged cleanly: validation loss decreased monotonically across all
three epochs (0.1967 → 0.1611 → 0.1546) with final training loss 0.1325 and zero
truncated examples, indicating no overfitting despite the modest 540-example
training set. The adapter modifies under 1% of model parameters.

**Safety and cost moved together.** Guardrail approaches impose a permanent
inference cost and prompt-based safety instructions lengthen context; here, prompt
length fell 27.0% while raw safety improved 75% relative and driver coverage
reached 1.000. The constraint moved from the prompt into the weights.

---

# 5. Novelty statement

> We present, to our knowledge, the first end-to-end clinical pipeline in which a
> deterministic safety rule set is used not only as a runtime filter but as the
> **supervision signal for fine-tuning** — enabling a 4-billion-parameter open
> model to internalise clinical constraints it previously required external repair
> to satisfy, reducing pre-guardrail unsafe generation by 75% on the deployed
> configuration and by 91% under prompt compression, while simultaneously reducing
> prompt cost by 27%.

**Four contributions, in order of evidential strength:**

1. **Constraint distillation (guardrail-as-teacher).** One deterministic rule
   implementation serves three roles: runtime enforcement, evaluation metric, and
   target-generation oracle. Unlike Constitutional AI, which distils from a neural
   critic, the teacher here is deterministic, auditable, and clinically
   specifiable — a meaningful distinction in a regulated domain. The fine-tuned
   model surpasses the raw output of the base model that generated its own
   targets (0.033 vs 0.133), which a purely self-referential process cannot do.

2. **Safety-stratified retrieval with independent corpus audit.** Filtering at
   *ingestion* rather than generation makes 38.9% of extracted guideline content
   unreachable by construction, while a third tier permits *referral without
   statement* for nutrition content. Prior clinical RAG systems filter at
   generation time and leave the corpus unconstrained.

3. **Simultaneous safety improvement and cost reduction**, via the
   generate-on-full / train-on-compact asymmetry — inverting the usual
   safety–cost tradeoff.

4. **A screening model that is deliberately non-invasive and empirically
   saturated.** The feature-expansion study demonstrates that a 14-input routine
   panel is at or near the performance optimum, so the non-invasive design is
   justified by evidence rather than by convenience.

---

# 6. Limitations

1. **Statistical power of the language-model ablation.** n = 30 with 1–4 events
   per arm; no pairwise comparison reaches significance (minimum *p* = 0.250).
   Mitigating evidence is structural — perfectly nested failure sets, zero adverse
   discordance, consistent direction across four arms, and a much larger effect on
   the 60-example compact-prompt test. Scaling to ≥ 200 profiles is the immediate
   next step.

2. **No expert clinical review.** All safety measurement is automated by the same
   rule set that performs enforcement. A nephrologist-rated evaluation against a
   six-dimension rubric (safety, guideline consistency, personalisation, clarity,
   overclaiming, actionability) is required before any deployment claim.

3. **Single-timepoint reference label.** NHANES provides one eGFR and one UACR
   measurement per participant, so the target is a screening category rather than
   the ≥3-month-persistence clinical CKD definition.

4. **Optimism in single-split risk-model figures.** High recall falls from 0.737
   (single split) to 0.540 ± 0.073 (25 repeated splits) because only 38 high-risk
   participants are present in the test partition. The pooled 2011–2023 estimate
   (0.696 ± 0.052, n = 205 high-risk) should be regarded as the honest figure.

5. **Temporal external degradation.** Balanced accuracy falls 0.05–0.06 on unseen
   cycles and the High→Low miss rate rises to ~8.5%, concentrated in the Moderate
   class.

6. **Residual emergency non-compliance.** One profile in the deployed arm;
   characterised as isolated diastolic elevation and stochastic in nature.
   Delivered safety is unaffected.

7. **Readability regression under grounding**, quantified in §4.12.

8. **Evaluation-profile realism.** Blood pressure sits at population-default
   values (SBP 125 / DBP 78) for a substantial fraction of sampled evaluation
   profiles, so blood-pressure-driven variation is concentrated in a minority of
   cases. Broader input sampling would strengthen external validity.

---

# 7. References to verify

Verify venue, year, and pagination against the actual publication before
submission.

1. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
2. Zhang et al. "RAFT: Adapting Language Model to Domain Specific RAG." COLM 2024.
3. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
4. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
5. Chen et al. "BGE M3-Embedding." ACL Findings 2024.
6. Douze et al. "The Faiss Library." arXiv 2024.
7. Bai et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv 2022.
8. Inan et al. "Llama Guard." arXiv 2023.
9. Rebedea et al. "NeMo Guardrails." EMNLP 2023 (System Demonstrations).
10. Wang et al. "Self-Instruct." ACL 2023.
11. Hinton, Vinyals & Dean. "Distilling the Knowledge in a Neural Network." NIPS DL Workshop 2015.
12. Singhal et al. "Large Language Models Encode Clinical Knowledge." *Nature* 620, 2023.
13. Zakka et al. "Almanac — Retrieval-Augmented Language Models for Clinical Medicine." *NEJM AI* 1(2), 2024.
14. Ji et al. "Survey of Hallucination in Natural Language Generation." *ACM Computing Surveys* 55(12), 2023.
15. Kincaid et al. "Derivation of New Readability Formulas." Naval Technical Training Command, 1975.
16. Weiss. *Health Literacy and Patient Safety.* AMA Foundation, 2007.
17. Wilson. "Probable Inference, the Law of Succession, and Statistical Inference." *JASA* 22, 1927.
18. McNemar. "Note on the Sampling Error of the Difference Between Correlated Proportions." *Psychometrika* 12, 1947.
19. Inker et al. "New Creatinine- and Cystatin C-Based Equations to Estimate GFR without Race." *NEJM* 385, 2021. (CKD-EPI 2021)
20. KDIGO 2012 Clinical Practice Guideline for the Evaluation and Management of CKD. *Kidney Int Suppl* 3(1), 2013. (risk grid)
21. KDIGO 2024 Clinical Practice Guideline for CKD. *Kidney International* 105(4S), 2024.
22. American Diabetes Association. "Standards of Care in Diabetes—2026, Ch. 11." *Diabetes Care* 49(Suppl. 1), 2026.
23. KDOQI Clinical Practice Guideline for Nutrition in CKD: 2020 Update. *AJKD* 76(3 Suppl 1), 2020.
24. Jang. "ANFIS: Adaptive-Network-Based Fuzzy Inference System." *IEEE Trans. SMC* 23(3), 1993.
25. Qwen Team. "Qwen3 Technical Report." arXiv 2025.
26. CDC. Chronic Kidney Disease Basics / Preventing CKD / Risk Factors for CKD. (public domain; include CDC attribution and non-endorsement disclaimer)
