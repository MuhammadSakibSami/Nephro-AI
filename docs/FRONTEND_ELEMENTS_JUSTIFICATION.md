# Frontend Elements — Mathematical & Technical Justification

**Purpose.** Every number, badge, chart, and claim shown in the Nephro-AI frontend is traced here to its exact source: the formula that produced it, the file/line that computes it, and — where relevant — the evaluation that validates it. Written for thesis-defense preparation: an examiner asking "where does that come from?" should have a precise answer here, in the code, or in `results/q1_llm_evaluation_summary.md`.

Two views exist:
- **Patient view** (`frontend/src/components/patient/`) — what a screening user sees.
- **Admin/developer view** (`frontend/src/components/admin/`) — the full technical breakdown, useful for a live demo to a committee.

---

## 1. Risk category and probabilities

**Where shown:** `RiskCard` (patient view, `PatientResult.jsx:44-62`) and `PredictionDetailsPanel` (admin view, `PredictionPanels.jsx:8-40`).

**Source:** `backend/model_service.py::_score_df()` (lines 162–198). This is the deployed **Final Model C** two-stage cascade — a Stage-1 RandomForest gate feeding a Stage-2 GradientBoosting classifier, exactly reproducing the architecture trained in `misc/notebooks/nhanes-model-c-train-save-download-and-predict.ipynb`.

**Exact computation:**

```
p1 = P(at-risk)           — Stage-1 RandomForest, positive-class probability
p2 = P(high | at-risk)    — Stage-2 GradientBoosting, positive-class probability, computed
                             for every input regardless of the gate (see note below)

risk_class =
    "low"       if p1 <  stage1_threshold   (0.44)
    "high"      if p1 >= stage1_threshold  AND p2 >= stage2_threshold (0.50)
    "moderate"  otherwise (p1 >= 0.44 but p2 < 0.50)

Displayed probabilities (row-normalized so they sum to 1):
    p_low      = 1 - p1
    p_moderate = p1 * (1 - p2)
    p_high     = p1 * p2
    confidence = max(p_low, p_moderate, p_high)
```

Both stages have their own fitted `ColumnTransformer` preprocessor (stored inside the joblib package) and are applied to disjoint feature subsets: Stage-1 uses 13 features, Stage-2 uses 6 (both subsets are declared inside the model package itself — `unique_inputs`, `stage1_features`, `stage2_features` — and the service trusts those over its own static config, so a future config edit can't silently desync from the trained model; verified identical in the full-project audit).

**Thresholds (0.44 / 0.50)** are not arbitrary UI constants — they are read from the joblib package's own `thresholds` dict, i.e. whatever the training notebook selected and saved with the model. If you retrain and choose different thresholds, the app picks them up automatically without a code change.

**What this is *not*:** a probability of having CKD. It is the trained classifier's own confidence in a risk-screening category. The clinical framing on the card explicitly says "Screening result — not a diagnosis."

---

## 2. Reliability score

**Where shown:** `ReliabilityCard` (`PatientResult.jsx:64-87`) and the "Completeness & reliability" panel (`PredictionPanels.jsx:42-64`, admin view).

**Source:** `backend/model_service.py::_reliability()` (lines 254–283).

This is a **deterministic rule-based heuristic**, not a statistically calibrated confidence interval and not a second model. It combines three signals that are each independently meaningful for how much to trust a single prediction:

```python
confidence   = max(probabilities.values())          # from the cascade above
margin       = top1_probability - top2_probability  # gap between the winning
                                                      # class and the runner-up
p_atrisk     = Stage-1 output (p1)
near_boundary = |p1 - stage1_threshold| <= 0.06  OR  margin < 0.15

level =
    "low"       if n_missing_features >= 5  OR confidence < 0.50  OR margin < 0.08
    "moderate"  elif near_boundary  OR confidence < 0.65  OR n_missing_features >= 2
    "high"      otherwise
```

**Why these specific numbers:** they encode three independent failure modes for a screening decision —
1. **Too many missing inputs** (≥5 of 14 imputed) means the prediction leans heavily on the trained pipeline's imputation rather than the patient's actual values.
2. **Low outright confidence** (<0.50) means no class was picked with real conviction.
3. **A thin margin** between the top two classes (<0.15, or <0.08 for the "low" cutoff) means the model is nearly indifferent between two categories — a small input change could flip the result.
4. **Being near the Stage-1 decision boundary** (within 0.06 of the 0.44 threshold) is the single most influential number in the whole cascade, since it decides whether Stage-2 even runs — so proximity to it is flagged explicitly.

The accompanying **notes** (`_reliability()` lines 268–276) are generated from the same signals, so the label and its justification are always consistent — the UI never shows "High reliability" without notes that would contradict it, or vice versa.

**Honest framing for defense:** this is a *transparency heuristic*, not a machine-learned confidence estimator (e.g. not a conformal-prediction interval or a Platt-scaled probability). It is deterministic, reproducible, and its logic is fully inspectable — which is arguably the more defensible property for a screening tool aimed at general users, but it should not be described as "statistically validated uncertainty quantification."

---

## 3. Input completeness %

**Where shown:** the progress bar under "Reliability" (patient view) and "Supplied features" (`PredictionPanels.jsx:54`, admin view).

**Source:** `backend/model_service.py::predict()` lines 320–328 build `input_quality`; the frontend just formats it (`frontend/src/lib/risk.js::completenessPct()`, lines 60–67):

```
completeness % = round(n_supplied_features / n_model_features * 100)
```

`n_supplied_features` / `n_missing_features` are computed by checking, per one of the model's 14 declared input features, whether the submitted value is `NaN` after cleaning (`model_service.py:320`). This is a plain count, not an estimate — every missing feature the model actually imputed is also listed by name in the admin "Missing & out-of-range" panel (`PredictionPanels.jsx:66-73`), so the percentage is auditable against the same request's raw JSON.

---

## 4. "Main signals" / model explanation

**Where shown:** `KeySignals` card (patient view, `PatientResult.jsx:89-111`) and the full "Top model signals" table (`ModelExplanationPanel`, admin view, `PredictionPanels.jsx:92-149`).

**Source:** `backend/model_service.py::_local_explanation()` (lines 200–252). Method name reported in the API itself: `"local_perturbation"`.

**This is a real, computed explanation — not SHAP, and the code says so explicitly** (`model_service.py:118`: `"explanation_method": "model-agnostic local perturbation against reference input values"`). The technique:

```
For the current input row X, and for the model's predicted class C, with
base probability P(C | X):

For each supplied feature f that has a defined reference value r_f:
    X' = X with feature f replaced by r_f (all else unchanged)
    P(C | X') = re-run the FULL two-stage cascade on X'
    contribution(f) = P(C | X) - P(C | X')
    direction(f) = "supports_prediction" if contribution(f) >= 0
                   else "pulls_away_from_prediction"

Sort all features by |contribution(f)| descending, keep top 8 (max_features).
Frontend then keeps only the top 5 for the patient-facing card.
```

The **reference values** (`model_service.py:27-30`, `backend/feature_config.py:61-73`) are fixed clinical "typical/neutral" midpoints — e.g. age 55, systolic BP 125, HbA1c 5.6, and 0 for every binary flag (no diabetes history, not on BP medicine, etc.). These are the same constants used as default form values, so "what would this prediction look like for a typical patient with everything else held at this value's typical level" is a meaningful, reproducible question — but they are **fixed clinical reference points chosen by the developers, not the training-data mean or median**, and this should be stated plainly if asked.

**What this technique is and isn't:**
- It **is** a legitimate, widely-used class of model-agnostic explanation (leave-one-out / single-feature perturbation), applied honestly against the actual deployed cascade (each perturbed row is re-scored through the real Stage-1+Stage-2 pipeline, not a surrogate).
- It **is not** SHAP or LIME (no Shapley game-theoretic attribution, no local surrogate fitting) — the admin panel's separate "SHAP / LIME plots" card (`PredictionPanels.jsx:141-147`) is explicitly labeled `"Placeholder for the paper extension"` and shows no numbers — correctly flagged as future work, not delivered functionality.
- It **does not** capture feature interactions: changing two correlated features together could shift the prediction differently than the sum of each one's individual perturbation effect. This is a known limitation of one-feature-at-a-time perturbation and is worth stating proactively.
- Because Stage-2 only fires when Stage-1's gate is open, a feature that mainly acts through the gate (Stage-1) will show a different contribution profile than one that acts through Stage-2 — this is real cascade behavior, not a bug.

---

## 5. Generated guidance text and citations

**Where shown:** `GuidanceCard` (patient view, `PatientResult.jsx:113-172`) and the "LLM run" / "Retrieved guideline chunks" panels (admin view, `LLMPanels.jsx:4-86`).

**Pipeline** (`llm/serving/llm_runtime.py::generate_guidance()`, lines 108–158):

1. **Retrieval:** the patient's risk class, key model factors, and profile are turned into a query (`retriever.build_query_from_llm_input`), embedded with **BGE-M3** (served via Ollama — confirmed genuinely active, not the lexical-fallback embedder, in the earlier full-project audit), and matched via **FAISS** against the guideline knowledge base. Top-k chunks (default k, `llm_runtime.py:113`) are returned with a similarity `score` (`llm/rag/retriever.py:242`) — this is the "Score" column in the admin "Retrieved guideline chunks" table.
2. **Generation:** the fine-tuned Qwen3-4B-Instruct-2507 (LoRA adapter, when `CKD_LLM_ADAPTER` is set) generates the guidance text conditioned on the retrieved chunks, using the compact prompt template when the adapter is active.
3. **Validation:** the deterministic rule-based safety validator (`llm/serving/safety_validator.py`, detailed in §7 below) checks and, if necessary, rewrites the raw output before it is ever returned.
4. **Citation:** `llm/serving/citations.py::build_citations()` deterministically maps each output section to the KB topics it's allowed to draw from (`SECTION_TOPICS`, lines 30-36), attributes `[n]` markers only to sources that were **actually retrieved** for that section (never a model-invented citation — explicitly designed so a clinician reviewer can't catch the system citing something it never looked at), numbers them in first-citation order (standard IEEE style), and builds the reference list from `llm/rag/ingest.py::load_sources()` (the KB's own `sources.csv`).

**"Grounding" (`grounding_any`, `grounding_valid_ratio`):** `llm_runtime.py:133-136` computes what fraction of the model's *self-reported* grounding citations were actually among the chunks it was given (`valid_grounding = [g for g in parsed.grounding if g in retrieved_ids]`). This measures whether the model is citing real, supplied sources — not whether the guidance is medically correct. This distinction is stated in `q1_report/q1_results_summary.md:71-72` and should be repeated at defense: **grounding is a measure of source presence, not of factual correctness.**

---

## 6. Prediction / two-stage cascade panel (admin)

**Where shown:** `PredictionDetailsPanel` (`PredictionPanels.jsx:8-40`).

Every field here (`p.confidence`, `probs.low/moderate/high`, `s1.p_atrisk`, `s1.threshold`, `s2.p_high_given_atrisk`, `s2.threshold`, `mi.model_name`) is read verbatim from the same `score` dict computed in §1 — nothing here is recomputed or approximated for display; the admin view is a raw, unrounded (to 3 decimals) look at the exact numbers the cascade produced for this request.

---

## 7. Safety panel (admin)

**Where shown:** `SafetyPanel` (`LLMPanels.jsx:88-146`).

**Source:** the deployed `llm/serving/safety_validator.py`. Each row/badge maps directly to one of its detection layers:

| UI check | Validator layer | Severity | Effect if triggered |
|---|---|---|---|
| Emergency redirect added | Emergency-symptom / vitals detector | moderate | Auto-appends an urgent-care line |
| Lab-confirmation reminder appended | Lab-confirmation reminder | moderate | Auto-appends a reminder to confirm with eGFR/UACR testing |
| Nutrition restriction rewritten | Nutrition-restriction rule | moderate | Rewrites a numeric/clinical diet target into a dietitian referral |
| Medication / diagnosis blocked | Diagnosis / medication / supplement blockers | **severe** | Text is replaced with the safe fallback template |

The **status** badge (`pass` / `sanitized` / `fallback`) is the validator's own decision: `pass` = returned as-is, `sanitized` = moderate issues auto-fixed and the fixed text returned, `fallback` = severe or structurally broken output replaced entirely with a safe generic template. This is the exact mechanism measured as `delivered_critical_unsafe` in the Q1 evaluation (§9).

**Known scope limitation (documented, not hidden):** the validator's emergency-symptom vocabulary (chest pain, breathlessness, severe swelling, etc.) is real and tested, but the live app's current input schema has no free-text symptom field — so in production today, emergency detection only fires on the numeric vitals path (systolic ≥180 or diastolic ≥120 mmHg). The symptom vocabulary is fully exercised by the Q1 evaluation harness (which does inject symptom text) but not by any current production request. This is recorded as a limitation in `results/q1_llm_evaluation_summary.md` §10.

---

## 8. System health tiles

**Where shown:** `SystemHealth.jsx`.

Calls the real `GET /health` and `GET /generate-guidance/health` endpoints via `Promise.allSettled` and renders whatever they report — model-loaded status, LLM mode (base vs. adapter-active), and DB connectivity. If an endpoint is unreachable, the tile shows a genuine offline/error state (`SystemHealth.jsx:44-51`), not a static "all green." This was directly verified in the full-project audit by starting the real backend and confirming `/db/health` and `/generate-guidance/health` reflect true live state (adapter active, Postgres connected).

---

## 9. Evaluation tab — ML model metrics

**Where shown:** `EvaluationPanel.jsx`, "ML model — test metrics" card, values from `frontend/src/data/evaluationData.js::ML_METRICS`.

**Source and verification:** these are **static, but real and independently re-verified** figures — not placeholders. They were checked against the actual training notebook (`misc/notebooks/nhanes-model-c-train-save-download-and-predict.ipynb`) and match, to 4-6 significant figures, the **"Full-development deployment"** row — the exact configuration that produced the currently-deployed `FINAL_MODEL_C_FULL_DEVELOPMENT.joblib`:

| Metric | Displayed | Notebook value (full-development deployment) |
|---|---:|---:|
| Accuracy | 0.7095 | 0.709484 |
| Balanced accuracy | 0.6602 | 0.660245 |
| Macro F1 | 0.5561 | 0.556058 |
| Low recall | 0.7391 | 0.739130 |
| Moderate recall | 0.5048 | 0.504762 |
| High recall | 0.7368 | 0.736842 |
| High → Low miss count | 2 | 2 |

**Definitions** (standard multiclass classification metrics, computed on a held-out NHANES 2021-23 test split, n=833 in this run):
- **Accuracy** = correct predictions / total predictions.
- **Balanced accuracy** = mean of per-class recall (low, moderate, high) — corrects for the class imbalance (far more "low" than "high" cases), so it's a fairer headline number than raw accuracy for this dataset.
- **Macro F1** = unweighted mean of each class's F1 score (harmonic mean of precision and recall for that class).
- **Recall (per class)** = of all patients who were truly that class, the fraction the model correctly identified as that class.
- **High → Low miss count** = the number of true "high" cases the model called "low" — the single most clinically important error type to keep near zero, since it means a genuinely at-risk patient is told they're not. This model produced 2 such misses out of the held-out set.

This card is honestly labeled "frozen" (a fixed snapshot of one trained model version) and should stay that way — it's real evaluation evidence, not a live metric that updates per request.

---

## 10. Evaluation tab — LLM metrics (Q1 final evaluation)

**Where shown:** `EvaluationPanel.jsx`, "LLM — Q1 final evaluation (frozen)" card, `evaluationData.js::LLM_METRICS`.

**Source:** Arm F ("ft_compact_val" — fine-tuned model + full RAG + safety validator + compact prompt) of the six-arm Q1 ablation, which is **the exact configuration this app deploys in production**. Full methodology and every number is in `results/q1_llm_evaluation_summary.md` and `q1_report/`.

**How the underlying evaluation works (for defense, in case an examiner asks "how do you know it's actually safe"):**
- A **300-profile frozen benchmark** (`q1_benchmark/q1_profiles_300.jsonl`) was built via rejection sampling *against the real deployed cascade* (§1) — so every "low"/"moderate"/"high" profile in the benchmark is one the actual production model would classify that way, not a hand-picked or synthetic label. 6 subgroups × 3 risk classes × 50 profiles, including a 60-profile emergency subgroup (red-flag symptoms/vitals) and a 30-profile adversarial subgroup (explicit medication-request attacks).
- **Six arms (A–F)** isolate each component's contribution by construction: A = base model + compact prompt, no RAG, no validator; B = base + full prompt + RAG; C = B + validator; D = fine-tuned + full + RAG; E = D + validator; F = fine-tuned + compact + RAG + validator (the deployed configuration).
- **Safety detectors** (`llm/eval/q1/detectors.py`) are rule-based (regex, sentence-scoped, negation-aware) — the same class of technique as the production validator, applied independently to *score* every arm's output for unsafe content (diagnosis claims, medication/dosage advice, dangerous fluid targets, false reassurance, unsupported numeric nutrition targets, failure to redirect emergencies, risk-category alteration, adversarial-attack compliance).
- **Statistics:** primary comparisons (e.g. "does the validator reduce unsafe output?") use **McNemar's test** (correct for paired binary outcomes measured on the *same* 300 profiles across two arms) with **Holm correction** applied within each pre-declared metric family (not globally, which would be a methodology error) to control for multiple comparisons. Proportions are reported with **Wilson confidence intervals** (more accurate than a normal approximation near 0%/100%, which is exactly where several of these arms land).

**The reported numbers for the deployed configuration (Arm F, n=300):**

| Metric | Value | What it means |
|---|---:|---|
| Format adherence | 100% | Every output parsed into the expected structured sections. |
| Delivered critical-unsafe | **0.0%** | Zero outputs reaching the (simulated) patient contained a critical safety violation, after the validator. |
| Emergency-redirect recall | **100%** | Every one of the 60 emergency-subgroup profiles received an urgent-care redirect. |
| Raw critical-unsafe (pre-validator) | 6.33% | Without the validator, ~6% of the fine-tuned model's raw outputs would have been unsafe — this is the problem the validator exists to solve, and the validator closes 100% of the gap on this benchmark. |
| Factor relevance | 0.843 | Fraction of retrieved guideline content judged topically relevant to the patient's actual risk factors (KB-topic-matching proxy — see limitation below). |
| Personalization | 1.99 / 2 | A 2-point rubric score for whether the guidance referenced the patient's specific factors rather than generic text. |
| Grounding (cited ≥1 source) | 99.7% | See §5 — source presence, not correctness. |
| Readability ≤ grade 8 | 58.7% | Fraction of outputs meeting a Flesch-Kincaid grade-8 readability target. |

**This is a genuinely strong, methodologically defended result, but three honest caveats belong in any defense answer:**
1. **Profiles are synthetic**, scored by the deployed cascade rather than drawn from real patients — this evaluates internal consistency and safety-mechanism effectiveness, not real-world outcome validity.
2. **Safety detection is rule-based**, both in Q1's scoring and the production validator — it is a reproducible screen, not a clinical adjudication, and could in principle miss an unseen paraphrasing.
3. **No clinician review** of the generated guidance was performed in this phase.

---

## 11. Demo cases

**Where shown:** `DemoCases.jsx`, backed by `frontend/src/data/demoCases.js`.

These are **real, realistic input payloads** — not pre-baked fake results. `demoCases.js` states this in its own header comment, and it was independently confirmed: selecting a demo case populates the form and calls the same `api.generateGuidance()` → live `/generate-guidance` endpoint as a manually filled-in form (`App.jsx::onRunDemo`). There is no pre-computed result JSON anywhere in the codebase for these cases — every demo run is a genuine live inference through the real model and LLM pipeline. They exist purely to let a demo audience (e.g. a thesis committee) see a low/moderate/high/emergency case without manually typing 14 fields each time.

---

## 12. Export buttons

**Where shown:** `ExportButtons.jsx`.

Serializes whatever `guidance` object is currently in memory from the most recent live run (`downloadJSON`, `downloadText`, `copyText`) — disabled until a real result exists (`ExportButtons.jsx:9`). Nothing hardcoded; this is a straight dump of the actual API response for that request, useful for including a raw example run in an appendix.

---

## 13. Summary verdict table

| Element | Verdict | Basis |
|---|---|---|
| Risk category & probabilities | Real, computed | §1 — two-stage cascade, thresholds from the trained model package |
| Reliability score | Real, deterministic heuristic | §2 — rule-based on confidence/margin/missingness, not a placeholder |
| Input completeness % | Real, exact count | §3 |
| Main signals / explanation | Real, model-agnostic perturbation | §4 — genuinely re-scores the real cascade; not SHAP, honestly labeled as such |
| Guidance text & citations | Real, live RAG + validator + citation pipeline | §5 — BGE-M3/FAISS retrieval, genuine no-invented-citation guarantee |
| Prediction/cascade panel | Real, raw values | §6 |
| Safety panel | Real, live validator output | §7 — one documented scope limitation (emergency-symptom field) |
| System health | Real, live endpoint calls | §8 |
| ML model metrics | Real, static, independently re-verified | §9 — matches the actual training notebook for the deployed model |
| LLM Q1 metrics | Real, static, from a documented 300-profile evaluation | §10 — methodology fully reproducible from `llm/eval/q1/` |
| Demo cases | Real inputs, real live inference | §11 |
| Export buttons | Real, dumps the actual response | §12 |
| SHAP/LIME plots card | Explicitly labeled placeholder, no numbers shown | §4 — correctly not claimed as delivered |

No element currently displays a fabricated or unlabeled placeholder number. The only historical issue (stale Phase-6 LLM figures presented without noting that fine-tuning and the full Q1 evaluation had since superseded them) was found and corrected as part of this audit.
