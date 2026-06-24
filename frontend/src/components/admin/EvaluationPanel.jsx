import React from "react";
import { Card, SectionTitle, Icon } from "../ui.jsx";
import { ML_METRICS, LLM_METRICS, EVAL_META, fmtMetric } from "../../data/evaluationData.js";

export default function EvaluationPanel() {
  return (
    <div className="space-y-5">
      <div className="rounded-xl2 bg-brand-50 ring-1 ring-brand-100 p-4 flex items-start gap-3">
        <Icon name="shield" className="w-5 h-5 text-brand-600 mt-0.5 shrink-0" />
        <p className="text-sm text-ink-700">
          <span className="font-semibold">{EVAL_META.safetyClaim}</span>{" "}
          {EVAL_META.deferNote}
        </p>
      </div>

      <Card>
        <SectionTitle icon="health" title="ML model — test metrics" sub={EVAL_META.mlSource} />
        <MetricGrid metrics={ML_METRICS} />
      </Card>

      <Card>
        <SectionTitle icon="spark" title="LLM — Phase 6 evaluation (frozen)" sub={EVAL_META.llmSource} />
        <MetricGrid metrics={LLM_METRICS} />
      </Card>
    </div>
  );
}

function MetricGrid({ metrics }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {metrics.map((m) => (
        <div key={m.label} className="rounded-xl border border-ink-100 p-3.5">
          <div className="text-xs text-ink-500 leading-tight min-h-[2rem]">{m.label}</div>
          <div className="kpi mt-1">{fmtMetric(m)}</div>
          {(m.note || m.target) && (
            <div className="text-[11px] text-ink-400 mt-0.5">{m.note || `target ${m.target}`}</div>
          )}
        </div>
      ))}
    </div>
  );
}
