import React from "react";
import { Card, SectionTitle, Icon, Spinner, Badge } from "./ui.jsx";
import { DEMO_CASES } from "../data/demoCases.js";

const ACCENT = {
  low: { ring: "ring-risk-lowring", bg: "bg-risk-lowbg", text: "text-risk-low", badge: "green" },
  mod: { ring: "ring-risk-modring", bg: "bg-risk-modbg", text: "text-risk-mod", badge: "amber" },
  high: { ring: "ring-risk-highring", bg: "bg-risk-highbg", text: "text-risk-high", badge: "red" },
};

export default function DemoCases({ onRun, loadingId }) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-ink-900">Demo cases</h2>
        <p className="text-sm text-ink-500">
          One-click supervisor cases. Each runs the full pipeline (ML → RAG → LLM → safety validator)
          and opens the patient result; switch to Admin to inspect internals.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-5">
        {DEMO_CASES.map((c) => {
          const a = ACCENT[c.accent];
          const loading = loadingId === c.id;
          return (
            <Card key={c.id} className={`ring-1 ${a.ring} ${a.bg} border-0 flex flex-col`}>
              <div className="flex items-center justify-between">
                <Badge tone={a.badge}>{c.title}</Badge>
                <span className={`grid place-items-center w-9 h-9 rounded-lg bg-white/70 ${a.text}`}>
                  <Icon name="demo" />
                </span>
              </div>
              <h3 className="font-semibold text-ink-900 mt-3">{c.subtitle}</h3>
              <p className="text-sm text-ink-600 mt-1 flex-1">{c.expected}</p>
              <button className="btn-primary w-full mt-4" disabled={!!loadingId} onClick={() => onRun(c)}>
                {loading ? <><Spinner /> Running…</> : <><Icon name="spark" className="w-4 h-4" /> Run this case</>}
              </button>
            </Card>
          );
        })}
      </div>

      <Card className="bg-ink-50/60 border-ink-100">
        <SectionTitle icon="shield" title="What to expect"
          sub="Talking points for the supervisor demo." />
        <ul className="text-sm text-ink-600 space-y-1.5 list-disc pl-5">
          <li><b>Low risk:</b> calm, preventive education; validator status <code>pass</code>.</li>
          <li><b>Moderate (BP + glucose):</b> guidance personalized to the drivers; clinician + lab follow-up reminder.</li>
          <li><b>High / emergency BP:</b> validator status <code>sanitized</code> — an urgent-care redirect is added automatically.</li>
        </ul>
      </Card>
    </div>
  );
}
