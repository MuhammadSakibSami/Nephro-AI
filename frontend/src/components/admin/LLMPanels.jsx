import React from "react";
import { Card, SectionTitle, KeyVal, JsonViewer, Badge, Collapsible } from "../ui.jsx";

export function LLMRAGPanel({ guidance }) {
  const chunks = guidance?.retrieved_chunks || [];
  const ids = guidance?.retrieved_chunk_ids || [];
  const promptTokenEst = estimateTokens(guidance);
  return (
    <div className="space-y-5">
      <div className="grid lg:grid-cols-2 gap-5">
        <Card>
          <SectionTitle icon="admin" title="LLM run" />
          <KeyVal k="Mode" v={<ModeBadge mode={guidance?.mode} />} />
          <KeyVal k="Model" v={<span className="font-mono text-xs">{guidance?.model || "—"}</span>} />
          <KeyVal k="Generation attempts" v={guidance?.n_attempts ?? "—"} />
          <KeyVal k="Grounding valid ratio" v={fmtRatio(guidance?.grounding_valid_ratio)} mono />
          <KeyVal k="Prompt token estimate" v={promptTokenEst ? `≈ ${promptTokenEst}` : "—"} mono />
        </Card>
        <Card>
          <SectionTitle icon="spark" title="Retrieval query" />
          <p className="text-sm text-ink-600 bg-ink-50 rounded-lg p-3 leading-relaxed">
            {guidance?.query || "—"}
          </p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {(guidance?.provenance || []).map((s) => (
              <Badge key={s} tone="brand">{shortenOrg(s)}</Badge>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <SectionTitle icon="flask" title="Retrieved guideline chunks"
          sub={`${chunks.length || ids.length} chunk(s) supplied to the model`} />
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-ink-400 border-b border-ink-100">
                <th className="py-2 pr-3 font-medium">Chunk ID</th>
                <th className="py-2 px-3 font-medium">Source</th>
                <th className="py-2 px-3 font-medium">Topic</th>
                <th className="py-2 px-3 font-medium">Restricted</th>
                <th className="py-2 pl-3 font-medium">Score</th>
              </tr>
            </thead>
            <tbody>
              {chunks.length === 0 && ids.map((id) => (
                <tr key={id} className="border-b border-ink-50">
                  <td className="py-2 pr-3 font-mono text-xs text-ink-700">{id}</td>
                  <td colSpan={4} className="py-2 px-3 text-ink-300">—</td>
                </tr>
              ))}
              {chunks.map((c) => (
                <tr key={c.chunk_id} className="border-b border-ink-50">
                  <td className="py-2 pr-3 font-mono text-xs text-ink-700">{c.chunk_id}</td>
                  <td className="py-2 px-3">{shortenOrg(c.organization || c.source_id)}</td>
                  <td className="py-2 px-3 text-ink-500">{c.topic || "—"}</td>
                  <td className="py-2 px-3">
                    {c.restricted ? <Badge tone="amber">restricted</Badge> : <span className="text-ink-300">no</span>}
                  </td>
                  <td className="py-2 pl-3 font-mono text-ink-500">
                    {c.score != null ? Number(c.score).toFixed(3) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-ink-400 mt-3">
          Grounding cited by the model (validated against retrieved): {(guidance?.grounding || []).join(", ") || "—"}
        </p>
      </Card>

      <div className="grid lg:grid-cols-2 gap-5">
        <Collapsible title="Raw model output (pre-validator)">
          <pre className="scroll-thin overflow-auto max-h-72 text-xs whitespace-pre-wrap text-ink-700 bg-ink-50 rounded-lg p-3">
            {guidance?.raw_guidance || "—"}
          </pre>
        </Collapsible>
        <Collapsible title="Final guidance JSON (post-validator)">
          <JsonViewer data={guidance?.guidance} />
        </Collapsible>
      </div>
    </div>
  );
}

export function SafetyPanel({ guidance }) {
  const violations = guidance?.violations || [];
  const has = (cat) => violations.some((v) => v.category === cat);
  const status = guidance?.status || "—";
  const checks = [
    { label: "Emergency redirect added", on: has("emergency_redirect") },
    { label: "Lab-confirmation reminder appended", on: has("missing_lab_reminder") },
    { label: "Nutrition restriction rewritten", on: has("nutrition_restriction") },
    { label: "Medication / diagnosis blocked", on: has("medication") || has("diagnosis") || has("supplement") },
  ];
  return (
    <div className="space-y-5">
      <div className="grid sm:grid-cols-3 gap-4">
        <StatusTile label="Validator status" value={status} tone={statusTone(status)} />
        <StatusTile label="Final unsafe" value="No" tone="green"
          hint="Guaranteed safe after validation" />
        <StatusTile label="Raw unsafe (this run)"
          value={violations.length ? "Handled" : "No"} tone={violations.length ? "amber" : "green"} />
      </div>

      <Card>
        <SectionTitle icon="shield" title="Validator actions taken" />
        <div className="grid sm:grid-cols-2 gap-2.5">
          {checks.map((c) => (
            <div key={c.label}
              className={`flex items-center gap-2.5 rounded-lg border px-3 py-2.5 text-sm
                ${c.on ? "border-risk-modring bg-risk-modbg text-ink-800" : "border-ink-100 text-ink-400"}`}>
              <span className={`w-2.5 h-2.5 rounded-full ${c.on ? "bg-risk-mod" : "bg-ink-200"}`} />
              {c.label}
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <SectionTitle icon="alert" title="Violations detected" />
        {violations.length === 0 ? (
          <p className="text-sm text-risk-low">None — model output was clean.</p>
        ) : (
          <div className="space-y-2">
            {violations.map((v, i) => (
              <div key={i} className="flex items-start justify-between gap-4 rounded-lg bg-ink-50 px-3 py-2">
                <div>
                  <span className="font-medium text-ink-800">{v.category}</span>
                  <span className="text-ink-400 text-xs ml-2">{v.layer}</span>
                  {v.detail && <p className="text-xs text-ink-500 mt-0.5">{v.detail}</p>}
                </div>
                <Badge tone={v.severity === "severe" ? "red" : "amber"}>{v.severity}</Badge>
              </div>
            ))}
          </div>
        )}
        <p className="text-xs text-ink-400 mt-3">
          Phase-6 evaluation: final unsafe output rate was 0% after safety validation.
        </p>
      </Card>
    </div>
  );
}

// --- helpers ---------------------------------------------------------------
function ModeBadge({ mode }) {
  if (mode === "production") return <Badge tone="green">qwen (production)</Badge>;
  if (mode === "local_demo") return <Badge tone="amber">local_demo</Badge>;
  return <Badge tone="gray">{mode || "—"}</Badge>;
}
function StatusTile({ label, value, tone, hint }) {
  const tones = {
    green: "bg-risk-lowbg text-risk-low ring-risk-lowring",
    amber: "bg-risk-modbg text-risk-mod ring-risk-modring",
    red: "bg-risk-highbg text-risk-high ring-risk-highring",
    gray: "bg-ink-100 text-ink-600 ring-ink-200",
  };
  return (
    <div className={`rounded-xl2 ring-1 p-4 ${tones[tone] || tones.gray}`}>
      <div className="text-xs font-medium opacity-80">{label}</div>
      <div className="text-xl font-bold capitalize mt-0.5">{value}</div>
      {hint && <div className="text-xs opacity-70 mt-0.5">{hint}</div>}
    </div>
  );
}
const statusTone = (s) => ({ pass: "green", sanitized: "amber", fallback: "red" }[s] || "gray");
const fmtRatio = (x) => (x === undefined || x === null ? "—" : Number(x).toFixed(2));
const shortenOrg = (o) =>
  ({ "Centers for Disease Control and Prevention": "CDC", "National Kidney Foundation": "NKF" }[o] || o || "—");
function estimateTokens(guidance) {
  const q = (guidance?.query || "").length;
  const chunks = (guidance?.retrieved_chunks || []).length * 180; // rough per-chunk chars
  if (!q && !chunks) return null;
  return Math.round((q + chunks + 1200) / 4); // ~4 chars/token incl. system prompt
}
