// Frozen thesis evaluation evidence (static). Sourced from the project's
// results: the two-stage ML cascade test metrics and the Phase-6 real-Qwen3
// LLM evaluation (30 held-out profiles).

export const ML_METRICS = [
  { label: "Accuracy", value: 0.7191, fmt: "ratio" },
  { label: "Balanced accuracy", value: 0.6641, fmt: "ratio" },
  { label: "Macro F1", value: 0.5591, fmt: "ratio" },
  { label: "Low recall", value: 0.7507, fmt: "ratio" },
  { label: "Moderate recall", value: 0.5048, fmt: "ratio" },
  { label: "High recall", value: 0.7368, fmt: "ratio" },
  { label: "High → Low miss count", value: 2, fmt: "int", good: "low" },
];

export const LLM_METRICS = [
  { label: "Format adherence", value: 1.0, fmt: "ratio", target: "1.00" },
  { label: "Final unsafe rate", value: 0.0, fmt: "ratio", target: "0.00", good: "low" },
  { label: "Fallback rate", value: 0.0, fmt: "ratio", good: "low" },
  { label: "Retry rate", value: 0.0, fmt: "ratio", good: "low" },
  { label: "Factor relevance", value: 0.966, fmt: "ratio" },
  { label: "Personalization", value: 1.93, fmt: "score2", note: "/ 2" },
  { label: "Grounding", value: 1.0, fmt: "ratio" },
  { label: "Readability (FK grade)", value: 6.9, fmt: "grade", note: "≤ 8 target" },
  { label: "Raw unsafe (pre-validator)", value: 0.1, fmt: "ratio",
    note: "all validator-handled; 0 diagnoses/medications", good: "low" },
];

export const EVAL_META = {
  mlSource: "NHANES 2021–23 · two-stage cascade · held-out test set",
  llmSource: "Phase 6 · real Qwen3-4B-Instruct-2507 + BGE-M3 RAG + safety validator · 30 held-out profiles (10/10/10)",
  deferNote:
    "Fine-tuning is deferred as a paper extension. The current thesis MVP uses an evaluated base Qwen3 + guideline RAG + safety validator.",
  safetyClaim:
    "In our Phase 6 evaluation, the final unsafe output rate was 0% after safety validation.",
};

export function fmtMetric(m) {
  if (m.fmt === "int") return String(m.value);
  if (m.fmt === "grade") return m.value.toFixed(1);
  if (m.fmt === "score2") return m.value.toFixed(2);
  return m.value.toFixed(m.value === 1 || m.value === 0 ? 2 : 3);
}
