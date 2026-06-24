import React from "react";
import { Card, SectionTitle, Icon, Chip, Badge } from "../ui.jsx";
import {
  riskMeta, reliabilityColor, RELIABILITY_LABEL, factorName, factorDirection, completenessPct,
} from "../../lib/risk.js";

const SECTION_ICON = {
  Summary: "spark", "Food habits": "flask", "Rest and sleep": "health",
  "Daily routine": "heart", "Follow-up reminder": "clipboard",
};

export default function PatientResult({ guidance, onReset }) {
  const prediction = guidance?.prediction || {};
  const meta = riskMeta(guidance?.risk_class);
  const isAtRisk = ["moderate", "high"].includes(String(guidance?.risk_class).toLowerCase());
  const urgent = guidance?.status === "sanitized" &&
    (guidance?.violations || []).some((v) => v.category === "emergency_redirect");

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-ink-900">Your screening result</h2>
        <button className="btn-ghost !py-2" onClick={onReset}>
          <Icon name="chevron" className="w-4 h-4 rotate-90" /> New screening
        </button>
      </div>

      {urgent && <UrgentBanner guidance={guidance} />}

      <div className="grid lg:grid-cols-3 gap-5">
        <RiskCard meta={meta} guidance={guidance} />
        <ReliabilityCard prediction={prediction} />
      </div>

      <KeySignals prediction={prediction} />

      <GuidanceCard guidance={guidance} />

      <SafetyNote isAtRisk={isAtRisk} guidance={guidance} />
    </div>
  );
}

function RiskCard({ meta, guidance }) {
  return (
    <Card className={`lg:col-span-2 ${meta.bg} ring-1 ${meta.ring} border-0`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-ink-500">Screening result — not a diagnosis</p>
          <div className="flex items-center gap-3 mt-1.5">
            <span className={`w-3.5 h-3.5 rounded-full ${meta.dot}`} />
            <span className={`text-4xl font-extrabold ${meta.text}`}>{meta.label} risk</span>
          </div>
          <p className="text-ink-600 mt-2 max-w-md">{meta.blurb}</p>
        </div>
        <Badge tone={meta.key === "low" ? "green" : meta.key === "moderate" ? "amber" : "red"}>
          {String(guidance?.risk_class || "").toUpperCase()}
        </Badge>
      </div>
    </Card>
  );
}

function ReliabilityCard({ prediction }) {
  const level = prediction?.reliability?.level;
  const pct = completenessPct(prediction);
  return (
    <Card>
      <SectionTitle icon="shield" title="Reliability" />
      <div className={`text-xl font-bold ${reliabilityColor(level)}`}>
        {RELIABILITY_LABEL[String(level || "").toLowerCase()] || "—"}
      </div>
      <div className="mt-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-ink-500">Input completeness</span>
          <span className="font-semibold text-ink-800">{pct ?? "—"}%</span>
        </div>
        <div className="h-2 rounded-full bg-ink-100 overflow-hidden">
          <div className="h-full bg-brand-500" style={{ width: `${pct ?? 0}%` }} />
        </div>
      </div>
      <p className="text-xs text-ink-400 mt-3">
        More complete inputs give a more reliable screening result.
      </p>
    </Card>
  );
}

function KeySignals({ prediction }) {
  const factors = (prediction?.explanation?.top_factors || []).slice(0, 5);
  if (!factors.length) return null;
  return (
    <Card>
      <SectionTitle icon="spark" title="Main signals influencing this result"
        sub="The everyday health factors that most affected your screening." />
      <div className="flex flex-wrap gap-2.5">
        {factors.map((f, i) => {
          const dir = factorDirection(f);
          const tone = dir === "raises" ? "mod" : "low";
          return (
            <Chip key={i} tone={tone}>
              <Icon name={dir === "raises" ? "alert" : "check"} className="w-3.5 h-3.5" />
              {factorName(f)}
              <span className="opacity-70">· {dir === "raises" ? "raises" : "lowers"}</span>
            </Chip>
          );
        })}
      </div>
    </Card>
  );
}

function GuidanceCard({ guidance }) {
  const sections = guidance?.guidance || {};
  const order = ["Summary", "Food habits", "Rest and sleep", "Daily routine", "Follow-up reminder"];
  const entries = order.filter((k) => sections[k]).map((k) => [k, sections[k]]);
  return (
    <Card>
      <SectionTitle icon="clipboard" title="Personalized kidney-risk education"
        sub="Guideline-grounded, general wellness guidance based on your result."
        right={guidance?.provenance?.length
          ? <span className="text-xs text-ink-400">Based on {guidance.provenance.length} guideline source(s)</span>
          : null} />
      <div className="space-y-4">
        {entries.length === 0 && <p className="text-ink-400 text-sm">No guidance available.</p>}
        {entries.map(([title, text]) => (
          <div key={title} className="rounded-xl border border-ink-100 p-4">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="grid place-items-center w-7 h-7 rounded-lg bg-brand-50 text-brand-600">
                <Icon name={SECTION_ICON[title] || "spark"} className="w-4 h-4" />
              </span>
              <h4 className="font-semibold text-ink-800">{title}</h4>
            </div>
            <p className="text-ink-600 leading-relaxed text-[15px]">{text}</p>
          </div>
        ))}
      </div>
      {guidance?.provenance?.length > 0 && (
        <p className="text-xs text-ink-400 mt-4">
          Based on guidelines: {guidance.provenance.join(", ")}.
        </p>
      )}
    </Card>
  );
}

function UrgentBanner({ guidance }) {
  return (
    <div className="rounded-xl2 bg-risk-highbg ring-1 ring-risk-highring p-4 flex items-start gap-3">
      <span className="grid place-items-center w-10 h-10 rounded-xl bg-risk-high text-white shrink-0">
        <Icon name="alert" />
      </span>
      <div>
        <h3 className="font-bold text-risk-high">Seek urgent medical care</h3>
        <p className="text-sm text-ink-700 mt-0.5">
          {guidance?.guidance?.Summary?.split(". ")[0] ||
            "If you have urgent symptoms such as chest pain, trouble breathing, or severe swelling, seek emergency medical care now."}
        </p>
      </div>
    </div>
  );
}

function SafetyNote({ isAtRisk }) {
  return (
    <Card className="bg-ink-50/60 border-ink-100">
      <div className="flex items-start gap-3">
        <span className="grid place-items-center w-9 h-9 rounded-lg bg-white text-brand-600 border border-ink-100 shrink-0">
          <Icon name="shield" />
        </span>
        <div className="text-sm text-ink-600 leading-relaxed">
          <p className="font-semibold text-ink-800">Education only — not a diagnosis or treatment.</p>
          <p className="mt-1">
            This screening and guidance do not diagnose chronic kidney disease, prescribe
            treatment, or replace a clinician. No medication or restrictive-diet advice is given.
          </p>
          {isAtRisk && (
            <p className="mt-1.5 font-medium text-ink-700">
              Because your result is moderate or high, please follow up with a clinician and ask
              about simple kidney tests (a blood test and a urine test) to confirm your result.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}
