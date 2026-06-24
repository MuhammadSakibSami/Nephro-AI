import React, { useState } from "react";
import { Tabs, Card, EmptyState, Badge, Icon } from "../ui.jsx";
import { PredictionDetailsPanel, InputQualityPanel, ModelExplanationPanel } from "./PredictionPanels.jsx";
import { LLMRAGPanel, SafetyPanel } from "./LLMPanels.jsx";
import EvaluationPanel from "./EvaluationPanel.jsx";
import ExportButtons from "./ExportButtons.jsx";

const TABS = [
  { id: "prediction", label: "Prediction" },
  { id: "input", label: "Input Quality" },
  { id: "explain", label: "Model Explanation" },
  { id: "llm", label: "LLM / RAG" },
  { id: "safety", label: "Safety Validator" },
  { id: "eval", label: "Evaluation" },
  { id: "export", label: "Export / Tools" },
];

export default function AdminDashboard({ guidance, onRunDemo }) {
  const [tab, setTab] = useState("prediction");
  const hasRun = !!guidance;
  // Evaluation tab is static — always available. Others need a run.
  const needsRun = !hasRun && !["eval"].includes(tab);

  return (
    <div className="space-y-5">
      <Header guidance={guidance} />
      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {needsRun ? (
        <Card>
          <EmptyState icon="admin" title="No screening loaded"
            sub="Run a screening from the Patient view, or load a demo case, to populate the research panels." />
          <div className="flex justify-center gap-2 pb-2">
            <button className="btn-ghost" onClick={() => onRunDemo?.("moderate")}>
              <Icon name="demo" className="w-4 h-4" /> Load a demo case
            </button>
          </div>
        </Card>
      ) : (
        <>
          {tab === "prediction" && <PredictionDetailsPanel guidance={guidance} />}
          {tab === "input" && <InputQualityPanel guidance={guidance} />}
          {tab === "explain" && <ModelExplanationPanel guidance={guidance} />}
          {tab === "llm" && <LLMRAGPanel guidance={guidance} />}
          {tab === "safety" && <SafetyPanel guidance={guidance} />}
          {tab === "eval" && <EvaluationPanel />}
          {tab === "export" && <ExportButtons guidance={guidance} />}
        </>
      )}
    </div>
  );
}

function Header({ guidance }) {
  return (
    <div className="flex items-center justify-between flex-wrap gap-3">
      <div>
        <h2 className="text-lg font-bold text-ink-900">Admin / Research dashboard</h2>
        <p className="text-sm text-ink-500">Model internals, RAG/LLM details, safety, and thesis evaluation.</p>
      </div>
      {guidance && (
        <div className="flex items-center gap-2">
          <Badge tone={({ low: "green", moderate: "amber", high: "red" })[String(guidance.risk_class).toLowerCase()] || "gray"}>
            {String(guidance.risk_class).toUpperCase()}
          </Badge>
          <Badge tone={guidance.mode === "production" ? "green" : "amber"}>{guidance.mode}</Badge>
          <Badge tone={({ pass: "green", sanitized: "amber", fallback: "red" })[guidance.status] || "gray"}>
            {guidance.status}
          </Badge>
        </div>
      )}
    </div>
  );
}
