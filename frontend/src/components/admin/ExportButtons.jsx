import React, { useState } from "react";
import { Card, SectionTitle, Icon } from "../ui.jsx";
import { downloadJSON, downloadText, copyText, patientSummaryText, adminReportText } from "../../lib/exportUtils.js";

export default function ExportButtons({ guidance }) {
  const [toast, setToast] = useState("");
  const flash = (msg) => { setToast(msg); setTimeout(() => setToast(""), 1600); };

  const disabled = !guidance;
  const stamp = guidance?.prediction?.profile_id || "session";

  const doCopy = async (text, name) => {
    const ok = await copyText(text);
    flash(ok ? `${name} copied` : "Copy failed");
  };

  return (
    <Card>
      <SectionTitle icon="download" title="Export / demo tools"
        right={toast && <span className="text-sm text-risk-low font-medium">{toast}</span>} />
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
        <Btn icon="download" disabled={disabled} label="Prediction JSON"
          onClick={() => downloadJSON(`prediction_${stamp}.json`, guidance.prediction)} />
        <Btn icon="download" disabled={disabled} label="Guidance JSON"
          onClick={() => downloadJSON(`guidance_${stamp}.json`, stripPrediction(guidance))} />
        <Btn icon="download" disabled={disabled} label="Admin report (.txt)"
          onClick={() => downloadText(`admin_report_${stamp}.txt`, adminReportText(guidance))} />
        <Btn icon="copy" disabled={disabled} label="Copy patient summary"
          onClick={() => doCopy(patientSummaryText(guidance), "Patient summary")} />
        <Btn icon="copy" disabled={disabled} label="Copy admin/debug report"
          onClick={() => doCopy(adminReportText(guidance), "Admin report")} />
        <Btn icon="copy" disabled={disabled} label="Copy full JSON"
          onClick={() => doCopy(JSON.stringify(guidance, null, 2), "Full JSON")} />
      </div>
      {disabled && <p className="text-xs text-ink-400 mt-3">Run a screening or demo case to enable exports.</p>}
    </Card>
  );
}

function Btn({ icon, label, onClick, disabled }) {
  return (
    <button className="btn-ghost justify-start" onClick={onClick} disabled={disabled}>
      <Icon name={icon} className="w-4 h-4" /> {label}
    </button>
  );
}

function stripPrediction(g) {
  const { prediction, ...rest } = g || {};
  return rest;
}
