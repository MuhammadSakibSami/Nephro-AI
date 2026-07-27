# Phase 9 — Ablation Comparison

| Metric | base | base_rag | base_rag_val | finetuned |
|---|---|---|---|---|
| format_adherence_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| raw_unsafe_rate | 0.133 | 0.067 | 0.1 | 0.033 |
| final_unsafe_rate | 0.133 | 0.067 | 0.0 | 0.0 |
| mean_factor_relevance | 0.95 | 0.983 | 0.983 | 1.0 |
| grounding_any_rate | 0.0 | 1.0 | 0.967 | 1.0 |
| mean_readability_grade | 7.556 | 7.89 | 8.114 | 7.881 |
| readability_ok_rate | 0.733 | 0.5 | 0.467 | 0.533 |
| mean_personalization | 2.0 | 2.0 | 2.0 | 2.0 |
| mentions_risk_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| mean_prompt_tokens | 1455.8 | 2243.1 | 2243.1 | 1638.1 |