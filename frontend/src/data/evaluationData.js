// Frozen thesis evaluation evidence (static). Sourced from the project's
// results: the deployed Final Model C two-stage cascade (full-development fit)
// test metrics, and the final Q1 six-arm ablation of the deployed LLM
// configuration (fine-tuned Qwen3 + BGE-M3 RAG + safety validator + compact
// prompt), evaluated on a 300-profile held-out benchmark. See
// results/q1_llm_evaluation_summary.md for the full evaluation.

export const ML_METRICS = [
  { label: "Accuracy", value: 0.7095, fmt: "ratio" },
  { label: "Balanced accuracy", value: 0.6602, fmt: "ratio" },
  { label: "Macro F1", value: 0.5561, fmt: "ratio" },
  { label: "Low recall", value: 0.7391, fmt: "ratio" },
  { label: "Moderate recall", value: 0.5048, fmt: "ratio" },
  { label: "High recall", value: 0.7368, fmt: "ratio" },
  { label: "High → Low miss count", value: 2, fmt: "int", good: "low" },
];

// Arm F ("ft_compact_val") from the Q1 six-arm ablation — fine-tuned model +
// full RAG + safety validator + compact prompt, i.e. the exact configuration
// this app deploys. n=300 profiles, deterministic decoding (seed 42).
export const LLM_METRICS = [
  { label: "Format adherence", value: 1.0, fmt: "ratio", target: "1.00" },
  { label: "Delivered critical-unsafe", value: 0.0, fmt: "ratio", target: "0.00", good: "low" },
  { label: "Emergency-redirect recall", value: 1.0, fmt: "ratio", target: "1.00" },
  { label: "Raw critical-unsafe (pre-validator)", value: 0.06333, fmt: "ratio",
    note: "fully corrected by the safety validator before delivery", good: "low" },
  { label: "Factor relevance", value: 0.8428, fmt: "ratio" },
  { label: "Personalization", value: 1.9933, fmt: "score2", note: "/ 2" },
  { label: "Grounding (cited ≥1 source)", value: 0.99667, fmt: "ratio" },
  { label: "Readability ≤ grade 8", value: 0.58667, fmt: "ratio",
    note: "share of outputs meeting the readability target" },
];

export const EVAL_META = {
  mlSource: "NHANES 2021–23 · Final Model C two-stage cascade (full-development fit) · held-out test set",
  llmSource: "Q1 final evaluation · fine-tuned Qwen3-4B-Instruct-2507 (LoRA) + BGE-M3 RAG + safety validator, compact prompt (deployed configuration) · 300-profile six-arm ablation, held-out benchmark",
  deferNote:
    "Fine-tuning (Phase 8, LoRA) and a full 300-profile six-arm ablation (Q1) isolating the contribution of RAG, the safety validator, fine-tuning, and the compact prompt are complete; the figures below are the deployed configuration's results from that evaluation.",
  safetyClaim:
    "In the final Q1 evaluation, the deployed configuration delivered 0% critical-unsafe output and 100% emergency-redirect recall across 300 held-out profiles.",
};

export function fmtMetric(m) {
  if (m.fmt === "int") return String(m.value);
  if (m.fmt === "grade") return m.value.toFixed(1);
  if (m.fmt === "score2") return m.value.toFixed(2);
  return m.value.toFixed(m.value === 1 || m.value === 0 ? 2 : 3);
}
