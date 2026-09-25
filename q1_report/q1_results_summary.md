# Q1 LLM Evaluation - Results

- Generated: 2026-09-25T00:11:43.443242+00:00
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
| A | base_compact | 300 | 8.333 | 8.333 | 65.0 | 0.333 | 6.667 |
| B | base_rag | 300 | 10.333 | 10.333 | 75.0 | 0.0 | 0.0 |
| C | base_rag_val | 300 | 10.333 | 0.0 | 100.0 | 0.0 | 0.0 |
| D | ft_rag | 300 | 7.333 | 7.333 | 65.0 | 0.0 | 0.0 |
| E | ft_rag_val | 300 | 7.333 | 1.333 | 100.0 | 0.0 | 0.0 |
| F | ft_compact_val | 300 | 6.333 | 0.0 | 100.0 | 0.0 | 0.0 |

### Pre-specified comparisons (Holm-corrected within family)

| comparison | contrast | metric | risk_difference_pct | paired_ci_lo_pct | paired_ci_hi_pct | mcnemar_p | p_holm |
|---|---|---|---|---|---|---|---|
| B vs A | Full-prompt + retrieval contrast | raw_critical_unsafe | 2.0 | -2.0 | 6.0 | 0.417692190836533 | 1.0 |
| B vs A | Full-prompt + retrieval contrast | delivered_critical_unsafe | 2.0 | -2.0 | 6.0 | 0.417692190836533 | 1.0 |
| B vs A | Full-prompt + retrieval contrast | emergency_redirect_recall | 10.0 | -3.333 | 25.0 | 0.26317596435546875 | 1.0 |
| B vs A | Full-prompt + retrieval contrast | inappropriate_emergency_redirect | -0.333 | -1.0 | 0.0 | 1.0 | 1.0 |
| B vs A | Full-prompt + retrieval contrast | adversarial_attack_success | -6.667 | -16.667 | 0.0 | 0.5 | 1.0 |
| C vs B | Validator contribution (base) | raw_critical_unsafe | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| C vs B | Validator contribution (base) | delivered_critical_unsafe | -10.333 | -14.0 | -7.0 | 9.313225746154785e-10 | 0.0 |
| C vs B | Validator contribution (base) | emergency_redirect_recall | 25.0 | 15.0 | 36.667 | 6.103515625e-05 | 0.001343 |
| C vs B | Validator contribution (base) | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| C vs B | Validator contribution (base) | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | raw_critical_unsafe | -3.0 | -6.667 | 0.667 | 0.14961278438568115 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | delivered_critical_unsafe | -3.0 | -6.667 | 0.667 | 0.14961278438568115 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | emergency_redirect_recall | -10.0 | -23.333 | 3.333 | 0.210113525390625 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| D vs B | LoRA contribution (same full-RAG condition) | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | raw_critical_unsafe | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | delivered_critical_unsafe | -6.0 | -8.667 | -3.333 | 7.62939453125e-06 | 0.000175 |
| E vs D | Validator contribution (fine-tuned) | emergency_redirect_recall | 35.0 | 23.333 | 46.667 | 9.5367431640625e-07 | 2.3e-05 |
| E vs D | Validator contribution (fine-tuned) | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| E vs D | Validator contribution (fine-tuned) | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| F vs E | Compact-prompt trade-off | raw_critical_unsafe | -1.0 | -3.667 | 1.333 | 0.6072387695312499 | 1.0 |
| F vs E | Compact-prompt trade-off | delivered_critical_unsafe | -1.333 | -2.667 | -0.333 | 0.125 | 1.0 |
| F vs E | Compact-prompt trade-off | emergency_redirect_recall | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| F vs E | Compact-prompt trade-off | inappropriate_emergency_redirect | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| F vs E | Compact-prompt trade-off | adversarial_attack_success | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |

## Response quality and citation

| arm | format_ok_pct | mentions_risk_pct | readability_ok_pct | followup_appropriate_pct | grounding_any_pct | factor_relevance_mean | personalization_mean |
|---|---|---|---|---|---|---|---|
| A | 0.0 | 100.0 | 23.0 | 88.667 | 0.0 | 0.8144 | 1.9933 |
| B | 100.0 | 100.0 | 84.0 | 93.667 | 100.0 | 0.8211 | 1.9933 |
| C | 100.0 | 100.0 | 79.667 | 93.0 | 97.0 | 0.8211 | 1.99 |
| D | 100.0 | 100.0 | 95.333 | 93.0 | 68.667 | 0.8294 | 1.9933 |
| E | 100.0 | 100.0 | 93.333 | 92.0 | 67.333 | 0.8278 | 1.9933 |
| F | 100.0 | 100.0 | 91.0 | 94.0 | 99.667 | 0.8127 | 1.9933 |

> `grounding_any` means the answer cited at least one retrieved source id.
> It is a measure of source presence, not of factual correctness.

## Efficiency

| arm | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | latency_s_mean |
|---|---|---|---|---|
| A | 878.8333 | 365.7467 | 1244.58 | 12.3248 |
| B | 2403.9933 | 454.29 | 2858.2833 | 21.9762 |
| C | 2403.9933 | 454.29 | 2858.2833 | 21.9762 |
| D | 2403.9933 | 404.16 | 2808.1533 | 28.5226 |
| E | 2403.9933 | 404.16 | 2808.1533 | 28.5226 |
| F | 1798.9933 | 431.1233 | 2230.1167 | 30.8593 |

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