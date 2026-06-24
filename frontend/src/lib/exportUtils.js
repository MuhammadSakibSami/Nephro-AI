// Download / copy helpers for export tools.

export function downloadJSON(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  triggerDownload(filename, blob);
}

export function downloadText(filename, text) {
  triggerDownload(filename, new Blob([text], { type: "text/plain" }));
}

function triggerDownload(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function patientSummaryText(guidance) {
  if (!guidance) return "";
  const g = guidance.guidance || {};
  const lines = [
    `Kidney-Risk Screening — ${String(guidance.risk_class || "").toUpperCase()} (screening result, not a diagnosis)`,
    "",
  ];
  for (const [k, v] of Object.entries(g)) lines.push(`${k}:`, v, "");
  lines.push(guidance.disclaimer || "");
  return lines.join("\n");
}

export function adminReportText(guidance) {
  if (!guidance) return "";
  const pred = guidance.prediction || {};
  const lines = [
    "ADMIN / RESEARCH REPORT — Kidney-Risk Screening",
    "================================================",
    `Risk class:        ${guidance.risk_class}`,
    `LLM mode:          ${guidance.mode}   (model: ${guidance.model})`,
    `Safety status:     ${guidance.status}`,
    `Probabilities:     ${JSON.stringify(pred.probabilities || {})}`,
    `Reliability:       ${pred.reliability?.level || "n/a"}`,
    `Retrieved query:   ${guidance.query || "n/a"}`,
    `Provenance:        ${(guidance.provenance || []).join(", ")}`,
    `Grounding ratio:   ${guidance.grounding_valid_ratio ?? "n/a"}`,
    `Violations:        ${(guidance.violations || []).map((v) => v.category).join(", ") || "none"}`,
    "",
    "Disclaimer: " + (guidance.disclaimer || ""),
  ];
  return lines.join("\n");
}
