import React from "react";
import { Card, SectionTitle, KeyVal, JsonViewer, Badge, Collapsible } from "../ui.jsx";
import { FIELD_BY_KEY } from "../../data/featureSchema.js";

const pct = (x) => (x === undefined || x === null ? "—" : `${(x * 100).toFixed(1)}%`);
const num = (x, d = 3) => (x === undefined || x === null ? "—" : Number(x).toFixed(d));

export function PredictionDetailsPanel({ guidance }) {
  const p = guidance?.prediction || {};
  const probs = p.probabilities || {};
  const s1 = p.stage1 || {};
  const s2 = p.stage2 || {};
  const mi = p.model_info || {};
  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <Card>
        <SectionTitle icon="admin" title="Final prediction" />
        <KeyVal k="Risk class" v={<Badge tone={badgeTone(p.risk_class)}>{p.risk_class || "—"}</Badge>} />
        <KeyVal k="Confidence" v={pct(p.confidence)} />
        <div className="mt-3 space-y-2">
          {["low", "moderate", "high"].map((c) => (
            <ProbBar key={c} label={c} value={probs[c]} />
          ))}
        </div>
      </Card>
      <Card>
        <SectionTitle icon="health" title="Two-stage cascade" />
        <KeyVal k="Stage-1 P(at-risk)" v={num(s1.p_atrisk)} mono />
        <KeyVal k="Stage-1 threshold" v={num(s1.threshold, 2)} mono />
        <KeyVal k="Sent to Stage-2" v={String(s1.sent_to_stage2 ?? "—")} />
        <KeyVal k="Stage-2 P(high | at-risk)" v={num(s2.p_high_given_atrisk)} mono />
        <KeyVal k="Stage-2 threshold" v={num(s2.threshold, 2)} mono />
        <div className="h-px bg-ink-100 my-2" />
        <KeyVal k="Model" v={mi.model_name} />
        <KeyVal k="Stage-1 / Stage-2 thr" v={`${num(mi.stage1_threshold, 2)} / ${num(mi.stage2_threshold, 2)}`} mono />
        <KeyVal k="Session ID" v={guidance?.prediction?.profile_id || guidance?.session_id} />
      </Card>
    </div>
  );
}

export function InputQualityPanel({ guidance }) {
  const p = guidance?.prediction || {};
  const iq = p.input_quality || {};
  const rel = p.reliability || {};
  const inputs = p.llm_context?.patient_inputs || {};
  const missing = iq.missing_features || [];
  const outOfRange = computeOutOfRange(inputs);
  return (
    <div className="space-y-5">
      <div className="grid lg:grid-cols-2 gap-5">
        <Card>
          <SectionTitle icon="clipboard" title="Completeness & reliability" />
          <KeyVal k="Supplied features" v={`${iq.n_supplied_features ?? "—"} / ${iq.n_model_features ?? "—"}`} />
          <KeyVal k="Missing features" v={iq.n_missing_features ?? "—"} />
          <KeyVal k="Reliability level" v={<Badge tone={relTone(rel.level)}>{rel.level || "—"}</Badge>} />
          <KeyVal k="Probability margin" v={num(rel.probability_margin)} mono />
          <KeyVal k="Near boundary" v={String(rel.near_boundary ?? "—")} />
          {(rel.notes || []).length > 0 && (
            <ul className="mt-2 text-sm text-ink-500 list-disc pl-5 space-y-0.5">
              {rel.notes.map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          )}
        </Card>
        <Card>
          <SectionTitle icon="alert" title="Missing & out-of-range" />
          <div className="text-sm">
            <p className="text-ink-500 mb-1">Missing (model-imputed):</p>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {missing.length ? missing.map((m) => (
                <span key={m} className="chip bg-ink-100 text-ink-600">{label(m)}</span>
              )) : <span className="text-risk-low">None</span>}
            </div>
            <p className="text-ink-500 mb-1">Out-of-normal-range values:</p>
            <div className="flex flex-wrap gap-1.5">
              {outOfRange.length ? outOfRange.map((m) => (
                <span key={m.key} className="chip bg-risk-modbg text-risk-mod">
                  {label(m.key)}: {m.value}
                </span>
              )) : <span className="text-risk-low">None flagged</span>}
            </div>
          </div>
        </Card>
      </div>
      <Collapsible title="Full input JSON (patient inputs)">
        <JsonViewer data={inputs} />
      </Collapsible>
    </div>
  );
}

export function ModelExplanationPanel({ guidance }) {
  const p = guidance?.prediction || {};
  const factors = p.explanation?.top_factors || [];
  const inputs = p.llm_context?.patient_inputs || {};
  return (
    <div className="space-y-5">
      <Card>
        <SectionTitle icon="spark" title="Top model signals"
          sub={p.explanation?.method ? `Method: ${p.explanation.method}` : undefined} />
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-ink-400 border-b border-ink-100">
                <th className="py-2 pr-3 font-medium">Feature</th>
                <th className="py-2 px-3 font-medium">Value</th>
                <th className="py-2 px-3 font-medium">Direction</th>
                <th className="py-2 px-3 font-medium">Contribution</th>
                <th className="py-2 pl-3 font-medium">Normal range</th>
              </tr>
            </thead>
            <tbody>
              {factors.length === 0 && (
                <tr><td colSpan={5} className="py-4 text-ink-400">No factor breakdown returned.</td></tr>
              )}
              {factors.map((f, i) => {
                const fld = FIELD_BY_KEY[f.feature];
                const raises = f.direction !== "pulls_away_from_prediction";
                return (
                  <tr key={i} className="border-b border-ink-50">
                    <td className="py-2 pr-3 font-medium text-ink-800">{f.label || f.feature}</td>
                    <td className="py-2 px-3 font-mono text-ink-700">
                      {f.current_value ?? inputs[f.feature] ?? "—"}{fld?.unit ? ` ${fld.unit}` : ""}
                    </td>
                    <td className="py-2 px-3">
                      <Badge tone={raises ? "amber" : "green"}>{raises ? "raises risk" : "lowers risk"}</Badge>
                    </td>
                    <td className="py-2 px-3 font-mono text-ink-700">
                      {f.absolute_contribution != null ? Number(f.absolute_contribution).toFixed(3) : "—"}
                    </td>
                    <td className="py-2 pl-3 text-ink-400">
                      {fld?.normal ? `${fld.normal[0]}–${fld.normal[1]}` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
      <Card>
        <SectionTitle icon="health" title="SHAP / LIME plots" sub="Placeholder for the paper extension." />
        <div className="rounded-xl border border-dashed border-ink-200 bg-ink-50/50 py-10 text-center text-ink-400 text-sm">
          Interpretability plot images can be dropped in here when generated
          (e.g. <code className="font-mono">/static/shap_summary.png</code>).
        </div>
      </Card>
    </div>
  );
}

// --- helpers ---------------------------------------------------------------
function ProbBar({ label, value }) {
  const tone = { low: "bg-risk-low", moderate: "bg-risk-mod", high: "bg-risk-high" }[label];
  const w = value != null ? Math.round(value * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="capitalize text-ink-500">{label}</span>
        <span className="font-mono text-ink-700">{value != null ? value.toFixed(3) : "—"}</span>
      </div>
      <div className="h-2 rounded-full bg-ink-100 overflow-hidden">
        <div className={`h-full ${tone}`} style={{ width: `${w}%` }} />
      </div>
    </div>
  );
}

const label = (k) => FIELD_BY_KEY[k]?.label || k;
const badgeTone = (r) => ({ low: "green", moderate: "amber", high: "red" }[String(r).toLowerCase()] || "gray");
const relTone = (l) => ({ high: "green", moderate: "amber", low: "gray" }[String(l).toLowerCase()] || "gray");

function computeOutOfRange(inputs) {
  const out = [];
  for (const [key, value] of Object.entries(inputs || {})) {
    const fld = FIELD_BY_KEY[key];
    if (!fld?.normal || value == null || value === "") continue;
    const n = Number(value);
    if (Number.isNaN(n)) continue;
    if (n < fld.normal[0] || n > fld.normal[1]) out.push({ key, value: `${n}${fld.unit ? " " + fld.unit : ""}` });
  }
  return out;
}
