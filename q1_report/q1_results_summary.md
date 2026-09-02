# Q1 LLM Evaluation - Results

- Generated: 2026-09-02T10:33:29.961732+00:00
- Benchmark: `q1_benchmark\q1_profiles_300.jsonl`
- Primary analysis seed: 42 (deterministic decoding)
- Statistical unit: **profile** (n=300)

## Arm definitions

| Arm | Prompt | RAG | Validator | Model |
|---|---|---|---|---|
| A (base_compact) | compact | no | no | base |
| B (base_rag) | full | yes | no | base |
| C (base_rag_val) | full | yes | yes | base |
| D (ft_rag) | full | yes | no | fine-tuned |
| E (ft_rag_val) | full | yes | yes | fine-tuned |
| F (ft_compact_val) | compact | yes | yes | fine-tuned |

## Primary safety outcomes

| arm | arm_name | n | raw_critical_unsafe_pct | delivered_critical_unsafe_pct | emergency_redirect_recall_pct | inappropriate_emergency_redirect_pct | adversarial_attack_success_pct |
|---|---|---|---|---|---|---|---|
| A | base_compact | 300 | 9.0 | 9.0 | 65.0 | 0.333 | 6.667 |
| B | base_rag | 300 | 12.333 | 12.333 | 66.667 | 0.0 | 0.0 |
| C | base_rag_val | 300 | 12.333 | 6.0 | 70.0 | 0.0 | 0.0 |
| D | ft_rag | 300 | 6.333 | 6.333 | 76.667 | 0.0 | 0.0 |
| E | ft_rag_val | 300 | 6.333 | 5.0 | 85.0 | 0.0 | 0.0 |
| F | ft_compact_val | 300 | 7.667 | 4.333 | 80.0 | 0.0 | 0.0 |

### Pre-specified comparisons (Holm-corrected within family)

| comparison | contrast | metric | risk_difference_pct | paired_ci_lo_pct | paired_ci_hi_pct | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|
| B vs A | RAG contribution | raw_critical_unsafe | 3.333 | -0.667 | 7.333 | 0.1538599441628321 | 1.0 |
| B vs A | RAG contribution | delivered_critical_unsafe | 3.333 | -0.667 | 7.333 | 0.1538599441628321 | 1.0 |
| B vs A | RAG contribution | emergency_redirect_recall | 1.667 | -13.333 | 16.667 | 1.0 | 1.0 |
| B vs A | RAG contribution | inappropriate_emergency_redirect | -0.333 | -1.0 | 0.0 | 1.0 | 1.0 |
| B vs A | RAG contribution | adversarial_attack_success | -6.667 | -16.667 | 0.0 | 0.5 | 1.0 |
| C vs B | Validator contribution (base) | raw_critical_unsafe | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| C vs B | Validator contribution (base) | delivered_critical_unsafe | -6.333 | -9.333 | -3.667 | 3.814697265625e-06 | 9.5e-05 |
| C vs B | Validator contribution (base) | emergency_redirect_recall | 3.333 | 0.0 | 8.333 | 0.5 | 1.0 |
| C vs B | Validator contribution (base) | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| C vs B | Validator contribution (base) | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | raw_critical_unsafe | -6.0 | -10.0 | -2.0 | 0.0050976432976312935 | 0.122343 |
| D vs B | LoRA contribution (same full-RAG condition) | delivered_critical_unsafe | -6.0 | -10.0 | -2.0 | 0.0050976432976312935 | 0.122343 |
| D vs B | LoRA contribution (same full-RAG condition) | emergency_redirect_recall | 10.0 | -5.0 | 25.0 | 0.26317596435546875 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | raw_critical_unsafe | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | delivered_critical_unsafe | -1.333 | -2.667 | -0.333 | 0.125 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | emergency_redirect_recall | 8.333 | 1.667 | 15.0 | 0.0625 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| F vs E | Compact-prompt trade-off | raw_critical_unsafe | 1.333 | -1.333 | 4.333 | 0.480682373046875 | 1.0 |
| F vs E | Compact-prompt trade-off | delivered_critical_unsafe | -0.667 | -3.0 | 1.667 | 0.79052734375 | 1.0 |
| F vs E | Compact-prompt trade-off | emergency_redirect_recall | -5.0 | -16.667 | 5.0 | 0.548828125 | 1.0 |
| F vs E | Compact-prompt trade-off | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| F vs E | Compact-prompt trade-off | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |

## Response quality and citation

| arm | format_ok_pct | mentions_risk_pct | readability_ok_pct | followup_appropriate_pct | grounding_any_pct | factor_relevance_mean | personalization_mean |
|---|---|---|---|---|---|---|---|
| A | 0.0 | 100.0 | 23.0 | 88.667 | 0.0 | 0.8144 | 1.9933 |
| B | 100.0 | 100.0 | 37.333 | 91.333 | 100.0 | 0.8679 | 1.9967 |
| C | 100.0 | 100.0 | 34.333 | 90.667 | 97.333 | 0.8645 | 1.9967 |
| D | 100.0 | 100.0 | 75.0 | 95.333 | 84.0 | 0.8696 | 1.9967 |
| E | 100.0 | 100.0 | 75.333 | 95.0 | 83.667 | 0.8696 | 1.9967 |
| F | 100.0 | 100.0 | 59.667 | 93.333 | 99.333 | 0.8746 | 2.0 |

> `grounding_any` means the answer cited at least one retrieved source id.
> It is a measure of source presence, not of factual correctness.

## Efficiency

| arm | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | latency_s_mean |
|---|---|---|---|---|
| A | 878.8333 | 365.7467 | 1244.58 | 12.3248 |
| B | 2272.2667 | 442.7133 | 2714.98 | 18.4568 |
| C | 2272.2667 | 442.7133 | 2714.98 | 18.4568 |
| D | 2272.2667 | 400.0333 | 2672.3 | 23.1697 |
| E | 2272.2667 | 400.0333 | 2672.3 | 23.1697 |
| F | 1667.2667 | 419.9667 | 2087.2333 | 23.149 |

## Limitations

- Profiles are synthetic and model-scored; no real participant data is used.
- Safety detection is rule-based and deterministic; it will miss paraphrases
  no rule covers, and may over-trigger on unusual phrasing. It is a screen,
  not an adjudication.
- Q1 medication/dosage scoring is sentence-scoped and skips explicitly negated
  guidance (e.g. "do not stop your medicine without talking to your doctor"),
  which the deployed validator does flag. Q1 scoring is therefore marginally
  more permissive than the guardrail by design: the metric measures harmful
  content, not guardrail triggering. It also adds named-drug and spaced-dosage
  patterns the deployed vocabulary does not cover.
- Relevance judgments for Recall@4/Precision@4/MRR/nDCG@4 use KB topic
  matching as a proxy; no human relevance labelling was performed.
- No clinician review of the generated guidance was performed in this phase.
- Results characterise this model, adapter, index and validator version only;
  they do not establish clinical validity or generalisation to deployment.