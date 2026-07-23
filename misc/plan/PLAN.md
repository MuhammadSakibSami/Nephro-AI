# CKD Risk Detection — Project Assessment & Improvement Plan

**Date:** June 2026  
**Status:** Pre-publication — active development  
**Goal:** Publishable journal paper on non-invasive CKD risk screening with ML + LLM guidance

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current State](#2-current-state)
3. [What You Did Well — Genuine Strengths](#3-what-you-did-well--genuine-strengths)
4. [Critical Issues — Must Fix Before Publication](#4-critical-issues--must-fix-before-publication)
5. [Important Issues — Significant Impact on Quality](#5-important-issues--significant-impact-on-quality)
6. [Minor Issues — Polish and Completeness](#6-minor-issues--polish-and-completeness)
7. [Improvement Plan](#7-improvement-plan)
8. [Publication Strategy](#8-publication-strategy)
9. [Immediate Next Steps](#9-immediate-next-steps)

---

## 1. Project Overview

**Core thesis idea:** Predict kidney disease risk using only routine, non-specialized clinical data (vitals, standard blood work, medical history) — without requiring a creatinine-based eGFR or urine albumin test as model inputs — and use an LLM to turn that risk output into personalized, non-medical lifestyle guidance for the patient.

**What exists right now:**
- A trained two-stage ML cascade model with extensive ablation history
- Trained on NHANES 2021–23 data, 14 input features (deployed Final Model C; an earlier iteration used 22 candidate features before selection), 3 output classes (low / moderate / high)
- Target variable defined by KDIGO 2012 eGFR + UACR grid (gold-standard clinical definition)
- A FastAPI backend serving predictions with explanations and LLM-ready context
- A Streamlit frontend for testing
- A Jupyter notebook documenting the full training pipeline
- Multiple experiment notebooks documenting what was tried and rejected

**What does NOT yet exist:**
- The LLM component
- A publication manuscript
- Cross-validation and bootstrap confidence intervals
- SHAP-based interpretability analysis
- A formal flat-classifier baseline comparison (Stage-1+Stage-2 cascade vs a single 3-class model)

---

## 2. Current State

### 2.1 Target Variable Definition (KDIGO 2012 Risk Grid)

This is now fully documented. Target was derived from NHANES 2021–23 lab data as follows:

**eGFR**: Calculated using CKD-EPI 2021 creatinine equation from serum creatinine, age, and sex.

**UACR**: Calculated as (urine albumin / urine creatinine) × 100 mg/g.

**GFR Categories**: G1 (≥90), G2 (60–89), G3a (45–59), G3b (30–44), G4 (15–29), G5 (<15)

**Albuminuria Categories**: A1 (<30 mg/g), A2 (30–300 mg/g), A3 (>300 mg/g)

**3-Class Risk Assignment:**
- **Low**: eGFR G1 or G2, AND albumin A1 (normal/mildly reduced GFR + normal albumin)
- **Moderate**: (G1 or G2 + A2) OR (G3a + A1) — one risk factor present
- **High**: All other combinations (G3b/G4/G5 regardless of albumin; or A3 regardless of GFR; or G3a + A2/A3)

**Critical point**: Serum creatinine, urine albumin, and urine creatinine are used **only to define the target** — they are explicitly excluded from model input features. This means the model predicts KDIGO-graded CKD risk from routine blood work and vitals, without the patient needing kidney-specific tests.

### 2.2 Model Architecture

The deployed **Final Model C** two-stage cascade (a plain-dict `.joblib` package of two
fitted sklearn stages — no custom pickle classes):

| Stage | Task | Algorithm | Features Used |
|---|---|---|---|
| Stage 1 | Low vs At-Risk gate | RandomForest (150 trees, depth 5) + RandomOverSampler | 13 features |
| Stage 2 | Moderate vs High | GradientBoosting (150 est, lr 0.05, depth 2) + OBU resampler | direct-six (6 features) |

**Stage-1 features (13):** age, blood_urea_nitrogen, hba1c, taking_bp_medicine, mean_sbp, glucose, hypertension_doctor_told, uric_acid, diabetes_yes, income_poverty_ratio, mean_dbp, hemoglobin, waist

**Stage-2 final features (6):** blood_urea_nitrogen, uric_acid, hemoglobin, serum_albumin, mean_sbp, age

**Unique model inputs: 14** (union of the two stages).

**Decision logic:**
- If Stage-1 P(at-risk) < 0.44 → predict **Low**
- Else if Stage-2 P(high) ≥ 0.50 → predict **High**
- Else → predict **Moderate**

### 2.3 Final Test Results

Deployed **Final Model C — full-development fit** (Train + Validation, N = 4,719),
evaluated on the held-out historical test set (N = 833):

| Metric | Value |
|---|---|
| Accuracy | 0.7095 |
| Balanced Accuracy | 0.6602 |
| Macro F1 | 0.5561 |
| Low Recall | 0.7391 |
| Moderate Recall | 0.5048 |
| High Recall | 0.7368 |
| High→Low Misses | 2 |
| Stage-1 threshold | 0.44 |
| Stage-2 threshold | 0.50 |

> The train-only audit fit (N = 3,886) scores slightly higher on this already-inspected
> test (accuracy 0.7239), but the full-development fit is the deployed artifact because
> it uses the additional 833 validation participants. See the Model C notebook for the
> exact reproduction audit.

### 2.4 Experiment History — What Has Already Been Tried

This is critical context: you have done far more than a first-pass model. Multiple experiment notebooks document an exhaustive search. Below is a complete record.

**Notebook 1 — OBU/Calibration/Ensemble/Alternatives:**

| Experiment | Best Result | Decision |
|---|---|---|
| Fine OBU search (k_neighbors, majority_remove_frac, target_ratio grid) | Validation good (moderate recall 0.64) but test high recall dropped to 0.63 | Rejected — not robust |
| Sigmoid calibration | Test balanced acc 0.6417, high recall 0.6316 | Rejected — worse than baseline |
| Isotonic calibration | Same as sigmoid | Rejected |
| Stage-2 LR | Moderate recall 0.4952, high recall 0.6316 | Rejected |
| Stage-2 RF | Moderate recall 0.4571, high recall 0.7368 | Rejected |
| Stage-2 ExtraTrees | Moderate recall 0.4381, high recall 0.6842 | Rejected |
| Stage-2 HistGradientBoosting | Moderate recall 0.4476, high recall 0.6053 | Rejected |
| Stage-2 XGBoost | Moderate recall 0.5048, high recall 0.6316 | Rejected — worse high recall |
| Ensemble (mean all) | Moderate recall 0.4762, high recall 0.7105 | Rejected |
| Cost-sensitive LR (high weight 4.0) | Moderate recall 0.2952, high recall 0.8158 | Rejected — moderate collapsed |
| Cost-sensitive ExtraTrees (high weight 4.0) | Moderate recall 0.1143, high recall 0.8947 | Rejected — moderate destroyed |

**Notebook 2 — Constrained Moderate Recall Experiments (A–E):**

| Experiment | Result | Decision |
|---|---|---|
| A: Joint Stage-1 + Stage-2 threshold search | No new stable candidate found | Same as baseline |
| B: Moderate rescue rule | 0 patients rescued | No improvement |
| C: Three-zone Stage-1 gate (uncertain zone → moderate) | Moderate recall 0.5238, but high recall dropped to 0.7105 | Rejected — high recall loss |
| D: Moderate-focused cost-sensitive Stage-2 | High recall 0.0789 on test | Rejected — high recall destroyed |
| E: Calibration + constrained threshold | High recall 0.1053 on test | Rejected — unsafe |

**Notebook 3 — Advanced Rescue/Ordinal Experiments (4-parameter search):**

| Experiment | Result | Decision |
|---|---|---|
| 1: 4-parameter threshold search (6480 configs, 3 search objectives) | No strict safe improvement found | Reject, keep as ablation |
| 2: High rescue head (LR/RF/ExtraTrees) | High recall unchanged, moderate slightly worse | Rejected |
| 3: Moderate rescue head | Moderate recall unchanged, high recall dropped to 0.7105 | Rejected |
| 4: Ordinal cascade (P(≥mod) + P(≥high) models) | Accuracy 0.7491, but high recall 0.6053, high→low miss rose to 5 | Rejected — unsafe |

**Overall summary from all experiments:**

No alternative approach improved both moderate recall AND preserved high recall ≥ 0.7368 AND kept high→low misses ≤ 2. The baseline cascade is the Pareto-optimal model under safety constraints.

### 2.5 Root Cause of Moderate Recall Limitation

The error analysis reveals the source of moderate recall loss clearly:

**Stage-1 leakage:** 32 of 105 true-moderate patients are predicted as low at Stage-1 and never reach Stage-2. Stage-2 cannot recover them regardless of how well it performs.

**Stage-2 confusion:** Of the 73 moderate patients reaching Stage-2, 20 are predicted as high (moderate→high errors). These are true moderate patients whose feature profiles resemble high-risk patients.

This is not a model deficiency — it is an inherent property of the KDIGO moderate class:
- G1/G2 + A2 patients: normal/near-normal GFR but mildly elevated albumin → looks like low risk on labs
- G3a + A1 patients: slightly reduced GFR but normal albumin → also looks like borderline low
- Both sub-groups share feature distributions with both low and high risk groups

Every experiment confirms this: any approach that forces moderate recall up inevitably either collapses high recall (dangerous for a screening tool) or creates excessive high→low misses (clinically unacceptable).

---

## 3. What You Did Well — Genuine Strengths

### 3.1 Target Variable Is Gold-Standard and Fully Documented
You derived the three risk classes from eGFR (CKD-EPI 2021) and UACR using the KDIGO 2012 risk grid — the internationally accepted clinical standard for CKD staging and risk stratification. This is exactly what journal reviewers would want. The derivation is reproducible from NHANES raw files.

### 3.2 Leakage Prevention Is Correct and Documented
Feature ranking (RF importance) is computed **only on the training set** and applied to select features for all downstream steps. The modeling dataset explicitly excludes serum creatinine, urine albumin, urine creatinine, eGFR, and UACR — the very quantities used to construct the target. This is methodologically sound and many published papers get this wrong.

### 3.3 Extensive Ablation Study — A Major Publication Asset
You have run more experiments than most published CKD-ML papers:
- Multiple Stage-2 algorithms (LR, RF, ExtraTrees, HistGBM, XGBoost, GBM)
- Calibration methods (sigmoid, isotonic)
- Multiple ensembles
- Cost-sensitive training with multiple weight configurations
- 6480+ threshold configurations in a constrained 4-parameter search
- Rescue rules and ordinal cascade architectures

This constitutes a comprehensive ablation study. In the paper, all of this becomes a "Model Selection and Ablation" section that demonstrates the final model was selected through rigorous experimentation, not luck.

### 3.4 Safety-First Clinical Design
Tracking high→low misses as a hard safety constraint is exactly the right clinical reasoning — this is the error that could cause a high-risk patient to believe they are fine and skip follow-up care. Having only 2 such errors on test while maintaining this as a non-negotiable constraint across all experiments shows real clinical awareness. The decision to reject approaches that improve aggregate metrics but violate this constraint is sound and publishable.

### 3.5 Two-Stage Cascade Architecture Is Clinically Motivated
The architecture directly reflects how kidney screening would work in practice: first identify who is at-risk (Stage-1), then distinguish moderate from high among at-risk patients (Stage-2). This is not just a modeling trick — it is a design decision that mirrors clinical workflow and enables the safety constraint to be enforced at Stage-1.

### 3.6 NHANES 2021–23 Is a Strong, Recent Dataset
NHANES is nationally representative and the 2021–23 cycle is the most recent available. Very few published ML papers use this cycle. This is a publication advantage.

### 3.7 Comprehensive Metric Evaluation
You compute accuracy, balanced accuracy, macro/weighted F1, per-class recall/precision, MCC, Cohen's Kappa, ROC-AUC, PR-AUC, Brier score, ECE, per-class sensitivity/specificity/NPV, and error transition counts. This is more thorough than most published papers and should be reported in full.

### 3.8 LLM Integration Architecture
The `llm_context` block in the API response is a thoughtful design. The ML model outputs structured risk context; the LLM converts it to patient-friendly lifestyle guidance. This combined system is novel in nephrology and a key differentiator for publication.

### 3.9 Data Cleaning Audit
You documented and systematically corrected structural missingness (BP medicine when no hypertension diagnosed), diabetes coding errors, and education/sex encoding issues. This level of data hygiene documentation is expected by journal reviewers.

---

## 4. Critical Issues — Must Fix Before Publication

### 4.1 No Cross-Validation

A single 70/15/15 stratified split means all metrics could vary depending on which patients happened to end up in the test set. Journal reviewers will not accept performance claims from a single split, regardless of how many experiments you ran within that split.

**What to do:** Implement **5-fold stratified cross-validation** wrapping the entire pipeline. Each fold should re-run: feature ranking on the training portion (on training fold only), Stage-1 training, Stage-2 training, threshold selection on validation fold, evaluation on held-out test fold. Report mean ± standard deviation for all key metrics across 5 folds.

**Note on threshold tuning inside CV:** For each fold, tune Stage-2 threshold on the inner validation fold, then evaluate on the outer test fold. This prevents threshold leakage.

**Important:** Your current test results (all the experiments in section 2.4) were all evaluated on the same held-out 15% test set. Once you implement CV, you can use the full dataset more efficiently and confirm whether your performance estimates are stable.

---

### 4.2 No Bootstrap Confidence Intervals

Point estimates without uncertainty bounds are not acceptable in clinical ML papers. A reviewer seeing "balanced accuracy = 0.6641" will immediately ask: what is the 95% confidence interval?

**What to do:** After fitting the final model on the full training set and evaluating on the held-out test set, bootstrap the test-set predictions 1000 times. Compute 95% CI for: accuracy, balanced accuracy, macro F1, per-class recalls, AUC. This can be done with `sklearn.utils.resample`.

If you implement CV (4.1), also report: mean ± std across CV folds as a second form of uncertainty quantification.

---

### 4.3 No SHAP Analysis

Your current explainability method (local perturbation: swap each feature with a population median and measure probability change) is non-standard. SHAP (SHapley Additive exPlanations) is the published field standard and reviewers will ask why it was not used.

**What to do:**
```
pip install shap
```
- Use `shap.TreeExplainer` for both Stage-1 (RandomForest) and Stage-2 (GradientBoosting) — both are tree-based models and SHAP is exact for them
- Generate:
  - **Global SHAP summary plot**: feature importance across all test-set patients (for Stage-1 and Stage-2 separately)
  - **SHAP beeswarm plot**: feature value vs SHAP value distribution
  - **3–5 case study waterfall plots**: individual patients with different outcomes
- Keep your local perturbation method in the API for real-time use, but use SHAP for the paper's analysis section
- Verify: do SHAP importances agree with your RF-importance-based feature rankings? If they diverge significantly, investigate and report

---

### 4.4 No Flat Baseline Comparison

**This is different from what you've already done.** Your experiment history compared alternative Stage-2 models — but that is not a baseline comparison. A proper baseline is a **single 3-class classifier trained directly on the same 14 Model C input features, without any cascade**. This is needed to show that your two-stage design adds value.

**Required comparisons:**

| Baseline | Status | Why needed |
|---|---|---|
| Logistic Regression (direct 3-class) | Not done | Simplest interpretable baseline; can LR do this? |
| Random Forest (direct 3-class, class_weight='balanced') | Not done | Same algorithm as Stage-1 but used directly |
| GradientBoosting (direct 3-class) | Not done | Same family as Stage-2 but used directly |
| LightGBM (direct 3-class) | Not done | State-of-the-art tabular ML |
| Your two-stage cascade | Done — final model | The proposed architecture |

**Note:** These flat baselines are easy to train (1–2 hours of work) and create the comparison table that demonstrates the cascade outperforms simpler alternatives — or at least achieves the same performance with better safety properties.

---

### 4.5 Moderate Recall of 0.50 — How to Handle This for Publication

**The situation:** You have conclusively demonstrated (through >15 distinct experiments) that moderate recall cannot be improved beyond ~0.50–0.52 without violating the high-recall safety constraint. This is not a failure — it is a finding. The moderate class is intrinsically difficult due to KDIGO category structure and feature overlap.

**You must not say "the model has 50% moderate recall" without explanation. Instead, the paper must make this argument explicitly:**

1. **Clinical justification:** KDIGO moderate-risk patients (G3a+A1 or G1/G2+A2) by definition sit at the boundary between low and high risk. They have either mildly reduced GFR OR mild proteinuria — a single abnormal marker — which looks similar to both low-risk and high-risk patients in routine lab profiles. This is a known clinical challenge even for experienced nephrologists.

2. **Stage-1 leakage is the root cause:** 32 of 105 moderate patients (30.5%) are excluded at Stage-1. Even perfect Stage-2 performance would cap moderate recall at ~0.695 with the current Stage-1 threshold.

3. **Ablation evidence:** You tried 6480+ threshold configurations, cost-sensitive training, ordinal cascades, rescue rules, and ensemble methods. None improved moderate recall without safety regression.

4. **Screening framing:** For a population-level screening tool, the primary goal is to not miss high-risk patients (high recall = 0.74, high→low misses = 2). Moderate patients flagged as low-risk will still likely receive follow-up through standard care pathways. The model's reliability indicator already flags low-confidence predictions for clinical review.

5. **Future work path:** This limitation motivates two future directions (to mention in Discussion): (a) additional features that specifically distinguish moderate risk, such as serum cystatin C, dietary protein intake, or physical activity — features not in NHANES routine; (b) a prospective study with richer biomarker panels.

---

## 5. Important Issues — Significant Impact on Quality

### 5.1 "Non-Invasive" Framing Needs Precision

Your paper's premise is predicting CKD risk "without invasive tests." This framing requires careful wording because:

- **Blood draws** are technically invasive (needle puncture). What you mean is: without the specific tests used to diagnose and stage CKD — serum creatinine-based eGFR and urine albumin-to-creatinine ratio (UACR).
- **BUN** in your features is a kidney-adjacent marker found in standard metabolic panels — not a CKD-specific test. Same for serum albumin and uric acid.
- **All 14 input features** come from standard annual checkup components: vital signs, basic metabolic panel, HbA1c, CBC, and self-reported history.

**Recommended framing for the paper:**

> "Our model predicts KDIGO-stratified CKD risk using only measurements obtainable from standard annual health assessments — general metabolic panel, CBC, vital signs, and patient-reported history — without requiring the kidney-specific tests currently used for CKD diagnosis: serum creatinine-based eGFR estimation and urine albumin-to-creatinine ratio (UACR) testing."

This framing is precise, clinically accurate, and clearly states the value proposition.

---

### 5.2 Calibration Finding Should Be Reported, Not Hidden

You tested sigmoid and isotonic calibration and found it did not improve classification decisions. This is a valid finding that belongs in the paper, not just in an experiment notebook.

**What to include in the paper:**
- Report calibration curves (reliability diagrams) for the final model's class probabilities on the test set
- Report ECE values
- Report that post-hoc calibration was attempted but did not improve classification metrics, consistent with the observation that the GradientBoosting model's probability estimates are already well-conditioned for the selected threshold

**Why this matters:** Reviewers may ask about calibration. Having the answer ready — "we tested it; here are the curves; here is the ECE; here is why it didn't change decisions" — is stronger than not having tested it.

---

### 5.3 Income-to-Poverty Ratio Raises Fairness Questions

`income_poverty_ratio` is used as a model feature. In the US context, income correlates with race and ethnicity. Even though race was correctly excluded, income may serve as a proxy. A journal in the digital health or AI medicine space will likely ask about this.

**What to do:**
- Conduct a **subgroup performance analysis**: compute key metrics separately by sex, by age group (18–44, 45–64, 65+), and by income quartile
- Report whether recall rates are substantially different across groups
- In the paper's Limitations section, acknowledge this concern and frame it as a tradeoff: income is a genuine social determinant of health (affects dietary quality, physical activity, stress, healthcare access), so excluding it would harm predictive power; including it may encode proxy bias

The subgroup threshold experiment you already ran (by sex, age, BMI, diabetes, hypertension, BP medicine) shows you have thought about this — that analysis should be referenced in the paper.

---

### 5.4 Three-Zone Gate Deserves a Closer Look

Experiment C (three-zone Stage-1 gate) on the test set showed:
- Moderate recall: 0.5048 → **0.5238** (+1.9 pp)
- Macro F1: 0.5591 → **0.5684** (+0.93 pp)
- High recall: 0.7368 → 0.7105 (−2.63 pp)

You rejected this because high recall dropped. But consider: **the high→low miss count was not reported for this experiment.** If those high recall losses are high→moderate errors (not high→low), they are substantially less dangerous. A patient predicted as "moderate" still gets flagged as at-risk and referred for further testing; a patient predicted as "low" does not.

**What to investigate:**
- Run the three-zone gate experiment and record the full error decomposition: how many high→low misses vs high→moderate misses?
- If high→low misses remain at 2 (same as baseline) and only high→moderate increases, this is clinically more acceptable
- If so, the three-zone gate might be worth reconsidering as an alternative or a sensitivity analysis in the paper

This is low-cost work (one notebook run) and could yield a meaningful finding.

---

### 5.5 Sample Size Must Be Reported Prominently

The paper must report the complete data flow:
- Total NHANES 2021–23 adult participants
- Excluded because of missing serum creatinine or urine data (no eGFR/UACR possible → no target)
- Final N with target available
- N per class (low / moderate / high)
- N in each split (train / validation / test)
- Train/validation/test class counts separately

This is standard for any clinical ML paper and reviewers will look for it immediately.

---

### 5.6 TreeMedianSelector Needs Justification

The `TreeMedianSelector` (keep features with ExtraTrees importance ≥ median) is a heuristic with an arbitrary cutoff. The paper needs to explain why median was chosen.

**Approach for the paper:** Report it as a regularization step to prevent overfitting in Stage-2 (which has a small training set). Mention that the 6 selected features (BUN, uric acid, hemoglobin, albumin, SBP, age) all have established clinical relevance to kidney disease severity, which provides face validity for the selection. An ablation comparing 6 features vs all 12 features in Stage-2 would formalize this justification.

---

## 6. Minor Issues — Polish and Completeness

### 6.1 — ROC and PR Curves Missing
Compute and plot one-vs-rest ROC curves and PR curves for all three classes on the test set. These are standard visualizations for any multi-class clinical ML paper.

### 6.2 — Normalized Confusion Matrix
Add row-normalized confusion matrix (proportion form) alongside the count matrix. The normalized version directly shows per-class recall as percentages and is the standard visualization.

### 6.3 — Learning Curves
A plot of performance vs training set size shows whether more data would help and confirms the model is not severely underfitting/overfitting. This can be generated with `sklearn.model_selection.learning_curve` and takes minutes.

### 6.4 — Local Perturbation Reference Values
The API explanation uses population default reference values (age=55, BMI=28, glucose=100). The paper should justify these: are they training-set medians? Population healthy norms? Clarify and document the choice.

### 6.5 — Feature Missing Rate Report
The paper should report missingness rates per feature in the NHANES dataset, and confirm that median/mode imputation did not introduce bias. The EDA script already computes this (Table 07 in `outputs_nhanes_eda/tables/`) — it just needs to be included in the paper.

### 6.6 — Reproducibility Statement
Include: Python version, all library versions (requirements.txt exists — good), random seed (42 — document it explicitly), and the specific NHANES 2021–23 XPT filenames (DEMO_L.XPT, BMX_L.XPT, BPXO_L.XPT, BIOPRO_L.XPT, CBC_L.XPT, GHB_L.XPT, ALB_CR_L.XPT, DIQ_L.XPT, BPQ_L.XPT).

---

## 7. Improvement Plan

Given the extensive experiment history, the focus now shifts from model improvement to publication preparation. The core model is stable and well-validated through ablation.

### Phase 1 — Remaining Methodology Work (~1–2 weeks)

**1.1 — Implement 5-fold cross-validation**

Wrap the full pipeline in a `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` loop. In each fold:
1. Compute RF importance on training fold only
2. Select top-19 Stage-1 features and top-12 Stage-2 features
3. Train Stage-1 RF + ROS
4. Train Stage-2 OBU + TreeMedianSelector + GBM
5. Tune Stage-2 threshold on validation fold (search 0.25–0.70)
6. Evaluate on held-out test fold

Report: mean ± std for accuracy, balanced accuracy, macro F1, per-class recalls, AUC, and high→low miss count. If these are stable (small std), your current single-split results are confirmed. If they vary a lot, the model is less robust than expected.

**1.2 — Bootstrap confidence intervals**

After CV, fit the final model on the full training set (or use the CV predictions pooled) and bootstrap 1000 iterations on the test set to get 95% CI for all key metrics. Report as: "Balanced accuracy: 0.664 (95% CI: 0.XXX–0.XXX)".

**1.3 — Train flat baselines**

Train and evaluate 4 flat 3-class classifiers using the same 14 Model C input features, same train/val/test split, and same CV:
- Logistic Regression (`class_weight='balanced'`)
- RandomForest (`class_weight='balanced'`, 150 estimators)
- GradientBoosting (150 estimators, learning_rate=0.05, max_depth=2, analogous to Stage-2)
- LightGBM (`class_weight='balanced'`)

Create a single comparison table: flat baselines vs your two-stage cascade, showing all key metrics including high→low miss count. The cascade should show superior safety metrics even if aggregate accuracy is similar.

**1.4 — Add SHAP analysis**

```python
import shap
explainer_s1 = shap.TreeExplainer(cascade.stage1_model.steps[-1][1])  # RF
explainer_s2 = shap.TreeExplainer(cascade.stage2_pipeline.steps[-1][1])  # GBM
```

Generate:
- Global beeswarm plots for Stage-1 and Stage-2 separately
- 3–5 waterfall plots for individual patients (one low, one moderate, one high, and one near-boundary case)
- Compare SHAP global importance ranking vs your RF-importance ranking — do they agree?

**1.5 — Investigate three-zone gate with full error decomposition**

Rerun Experiment C (three-zone Stage-1 gate, t_low=0.44, t_high=0.54, Stage-2 threshold=0.50) and record the full error table: specifically, how many high→low misses vs high→moderate misses? If high→low misses remain 2, reconsider whether the tradeoff (moderate recall 0.52, high recall 0.71, macro F1 0.57) is acceptable and worth reporting as an alternative configuration in the paper.

---

### Phase 2 — Analysis for Paper (~1 week)

**2.1 — Formal subgroup fairness analysis**

Compute balanced accuracy, high recall, and moderate recall separately for:
- Males vs Females
- Age 18–44 vs 45–64 vs 65+
- Income quartile 1 (lowest) vs 4 (highest)

Report as a table in the paper. This directly answers reviewer concerns about fairness.

**2.2 — Calibration reporting**

Plot reliability diagrams for the final model's class probabilities. Report ECE. Note that post-hoc calibration was tested and did not improve classification (you have this data from Notebook 1). Include a supplementary table with calibration results.

**2.3 — Stage-1 gate analysis for paper**

Report the Stage-1 gate summary formally: what proportion of each true class is passed to Stage-2 vs kept at Stage-1? This shows the cascade design working as intended (518/690 low patients correctly gated out early) and explains the moderate recall ceiling (32 moderate patients lost at Stage-1).

**2.4 — Feature importance analysis**

Using the RF importance rankings from training and SHAP values from 1.4:
- Stage-1 top features (age, BUN, HbA1c, taking_bp_medicine, mean_sbp) align with known CKD risk factors
- Stage-2 top features (BUN, uric acid, hemoglobin, albumin, mean_sbp, age) are all clinically established markers of CKD severity
- Write a paragraph linking each feature to kidney disease physiology

---

### Phase 3 — LLM Component (~3–4 weeks, can run in parallel)

This is your most novel contribution and the strongest differentiator for high-impact journal submission.

**3.1 — Architecture decision**

Start with the Claude API (claude-sonnet-4-6) with RAG using kidney disease guidelines. This is achievable in 1–2 weeks and will produce high-quality outputs.

**Knowledge base for RAG:**
- KDIGO 2012 CKD clinical guidelines (lifestyle management sections)
- NKF KDOQI nutrition guidelines for CKD (protein, potassium, phosphorus)
- ADA guidelines on diabetes management for CKD patients
- JNC 8 hypertension management guidelines
- Exercise recommendations for CKD (Bulletin of the NKF)

**3.2 — System prompt design**

The `llm_context` block from your API is already a solid foundation. The system prompt should:
- Accept the structured ML output (risk class, key contributing features, patient inputs)
- Generate lifestyle suggestions tied to the specific risk factors that drove the prediction
- Constrain to wellness/lifestyle advice only: diet, hydration, physical activity, sleep, stress management
- Explicitly forbid: medication changes, diagnosis, referral instructions, supplement recommendations
- Format as patient-friendly language (Flesch-Kincaid grade 8 or below)

**3.3 — LLM evaluation (required for paper)**

To publish with the LLM component, you need evaluation:
- **Factual accuracy** (20–30 generated suggestions reviewed by a nephrologist or using KDIGO guidelines as reference): are suggestions consistent with guidelines?
- **Safety check** (adversarial testing with edge-case inputs): does the LLM ever give medical advice?
- **Relevance** (human rater scoring 1–5): does the advice match the patient's specific risk factors?
- **Readability**: compute Flesch-Kincaid readability score on 30 generated outputs

---

### Phase 4 — Publication (~3–4 weeks, after Phases 1–3)

**4.1 — Paper structure**

```
Title: Non-Invasive Kidney Disease Risk Screening Using a Two-Stage Machine
       Learning Cascade with LLM-Powered Lifestyle Guidance: A NHANES 2021–23 Study

1. Introduction
   - CKD burden (prevalence, progression, cost)
   - Barriers to diagnosis (eGFR/UACR access in low-resource/community settings)
   - ML for CKD risk prediction: prior work and gaps
   - LLM for patient communication: novel angle
   - Our contributions (bullet list)

2. Methods
   2.1 Dataset: NHANES 2021–23, inclusion criteria, exact N and attrition
   2.2 Target variable: KDIGO 2012 risk grid using CKD-EPI 2021 eGFR + UACR
   2.3 Feature set: 14 input features (Final Model C), explicit exclusion of creatinine/urine markers
   2.4 Preprocessing: imputation, coding fixes, structural missingness handling
   2.5 Model architecture: two-stage cascade design rationale
   2.6 Stage-1: RF + ROS, feature selection, threshold
   2.7 Stage-2: OBU resampling, TreeMedianSelector, GBM, threshold tuning
   2.8 Evaluation: CV, metrics, bootstrap CI, safety metrics
   2.9 Baselines: flat classifiers for comparison
   2.10 Ablation study: summary of all experiments tried
   2.11 Interpretability: SHAP
   2.12 LLM integration: architecture, RAG, evaluation

3. Results
   3.1 Dataset characteristics (N, class distribution, attrition)
   3.2 Baseline comparison (Table: flat models vs cascade)
   3.3 Ablation study results (Table: all experiments tried)
   3.4 Final model performance with CV mean±std and bootstrap CI
   3.5 Stage-1 gate analysis
   3.6 Error decomposition
   3.7 Feature importance (SHAP figures)
   3.8 Calibration analysis
   3.9 Subgroup fairness analysis
   3.10 LLM output evaluation

4. Discussion
   4.1 Clinical interpretation: what the performance means for screening
   4.2 Moderate recall limitation: root cause and clinical justification
   4.3 Safety analysis: why high→low miss rate is the primary metric
   4.4 Feature relevance: clinical interpretation of top features
   4.5 Comparison with KFRE and existing risk scores (literature only — you don't have their data)
   4.6 LLM integration: implications for patient engagement
   4.7 Limitations: single US cohort, no prospective validation, no external validation
   4.8 Future work: cystatin C, prospective study, external cohort validation

5. Conclusion
```

---

## 8. Publication Strategy

### 8.1 — Unique Contributions

1. **KDIGO-grounded target from routine labs** — eGFR+UACR target predicted from standard blood work, no kidney-specific tests required
2. **Two-stage cascade with safety-first design** — architecture explicitly enforces the high→low miss constraint
3. **Exhaustive ablation study** — 15+ architectures tested with consistent safety constraints; most ML papers test 2–3
4. **NHANES 2021–23** — most recent cycle; few published papers use it
5. **ML + LLM integration** — risk stratification feeding personalized lifestyle guidance; novel in nephrology
6. **Transparent limitations with clinical reasoning** — moderate recall limitation explained mechanistically, not swept under rug

### 8.2 — Target Journals

| Journal | Impact | Fit | Notes |
|---|---|---|---|
| **npj Digital Medicine** | ~12 IF | Very high | Digital health + AI; values ML+LLM novel combination |
| **Journal of Medical Internet Research (JMIR)** | ~7 IF | High | Broad digital health; accepts technology-forward papers |
| **Artificial Intelligence in Medicine** | ~6 IF | Very high | Focused ML for medicine; good fit for ablation-heavy paper |
| **BMC Nephrology** | ~3 IF | High | Domain-specific; lower bar but lower visibility |
| **PLOS ONE** | ~3 IF | High | Open access; accepts methods-focused papers |

**Recommended:** Submit to npj Digital Medicine or JMIR first (higher impact). Use BMC Nephrology or Artificial Intelligence in Medicine as fallbacks.

### 8.3 — What Separates Publishable from Current State

| Current (not publishable) | Required (publishable) |
|---|---|
| Single train/val/test split | 5-fold CV + bootstrap CI on test set |
| No flat 3-class baselines | 4+ flat baselines in comparison table |
| No SHAP | Global + local SHAP with figures |
| Local perturbation explanation only | SHAP for paper, perturbation for API |
| Ablation in notebooks, not paper | Ablation table in Methods/Results |
| LLM not built | RAG-based LLM with quality evaluation |
| Calibration tested but not reported | Calibration curves in paper |
| Subgroup threshold experiments in notebooks | Formal subgroup table in paper |
| Moderate recall limitation unexplained | Mechanistic explanation in Discussion |

---

## 9. Immediate Next Steps

Work in this order. Each step is independent unless noted.

**Step 1 (This week) — Cross-validation**
Implement 5-fold CV. This is the single highest-priority task. It either confirms your results (good) or reveals instability (important to know). Estimated time: 1 day of coding, a few hours of compute.

**Step 2 (This week, parallel with Step 1) — Flat baselines**
Train Logistic Regression, flat RF, flat GBM, and LightGBM as direct 3-class classifiers. This is a few hours of work. Create the comparison table.

**Step 3 (Next week) — SHAP analysis**
Install SHAP and run TreeExplainer on Stage-1 and Stage-2. Generate global and local plots. Compare to your existing RF importance rankings.

**Step 4 (Next week) — Three-zone gate error decomposition**
Rerun Experiment C with full error decomposition. Record specifically: how many high→low misses? If still 2, this becomes a viable alternative to present.

**Step 5 (Parallel, 2–3 weeks) — LLM prototype**
Build the first LLM integration using Claude API + your existing `llm_context` block. Get it working end-to-end before building the RAG layer.

**Step 6 (After Steps 1–4) — Paper writing**
The results section writes itself once you have: CV table, baseline comparison table, ablation summary table, SHAP figures, and calibration curves.

---

## Appendix A: Experiment Log Summary

All experiments tried and rejected before finalizing the current model:

| Notebook | Experiments | Outcome |
|---|---|---|
| Stage2_OBU_Calibration_Ensemble | Fine OBU search, sigmoid/isotonic calibration, 5 alternative Stage-2 algorithms, ensemble, cost-sensitive training | All rejected; baseline cascade best |
| Constrained_Moderate_Recall | Joint threshold search (A), moderate rescue rule (B), three-zone gate (C), moderate-focused cost-sensitive (D), calibration+threshold (E) | C marginally promising; others rejected |
| Advanced_Rescue_Head_Ordinal_Cascade | 4-parameter search (6480 configs), high rescue head, moderate rescue head, ordinal cascade | All rejected on safety grounds |

**Key finding:** The baseline cascade (Stage-1 RF+ROS, threshold 0.44; Stage-2 OBU+TreeMedianSelector+GBM, threshold 0.50) is the Pareto-optimal model under the safety constraint (high→low miss ≤ 2, high recall ≥ 0.73).

---

## Appendix B: File Reference

| File | Purpose |
|---|---|
| [nhanes_2021–2023_eda_+_cleaned_csv_creation.py](ckd_risk_app/model_package/nhanes_2021–2023_eda_+_cleaned_csv_creation.py) | Target variable construction, EDA, leakage-aware dataset creation |
| [misc/notebooks/nhanes-model-c-train-save-download-and-predict.ipynb](ckd_risk_app/misc/notebooks/nhanes-model-c-train-save-download-and-predict.ipynb) | Deployed Final Model C train/save/verify notebook |
| [final model pawar steps.pdf](ckd_risk_app/model_package/final%20model%20pawar%20steps.pdf) | Experiment history: all tried approaches and outcomes |
| [backend/model_service.py](ckd_risk_app/backend/model_service.py) | Model C loading (plain-dict package), scoring, explanation, LLM context builder |
| [backend/feature_config.py](ckd_risk_app/backend/feature_config.py) | 14-feature metadata, default values, units |
| [backend/schemas.py](ckd_risk_app/backend/schemas.py) | Request/response Pydantic models |
| [backend/main.py](ckd_risk_app/backend/main.py) | FastAPI endpoints |
| [model_package/FINAL_MODEL_C_FULL_DEVELOPMENT.joblib](ckd_risk_app/model_package/FINAL_MODEL_C_FULL_DEVELOPMENT.joblib) | Deployed trained model artifact (14 inputs, no custom pickle classes) |

---

*This plan was generated after reading: the full training notebook, the EDA and target construction script (revealing the KDIGO-based target definition), the experiment history PDF (documenting all tried and rejected approaches), and all backend/frontend code. It reflects the actual state of the project as of June 2026.*
