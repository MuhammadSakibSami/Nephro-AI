import React, { useEffect, useState } from "react";
import { Card, SectionTitle, KeyVal, Badge, Icon, Spinner, JsonViewer } from "./ui.jsx";
import { api } from "../api/client.js";

export default function SystemHealth() {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [llm, setLlm] = useState(null);
  const [err, setErr] = useState("");

  const refresh = async () => {
    setLoading(true); setErr("");
    const [h, l] = await Promise.allSettled([api.health(), api.llmHealth()]);
    setHealth(h.status === "fulfilled" ? h.value : null);
    setLlm(l.status === "fulfilled" ? l.value : null);
    if (h.status === "rejected" && l.status === "rejected") setErr(String(h.reason?.message || h.reason));
    setLoading(false);
  };

  useEffect(() => { refresh(); }, []);

  const backendOk = !!health || !!llm;
  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-ink-900">System health</h2>
          <p className="text-sm text-ink-500">Backend, ML model, and LLM mode.</p>
        </div>
        <button className="btn-ghost !py-2" onClick={refresh} disabled={loading}>
          {loading ? <Spinner /> : <Icon name="spark" className="w-4 h-4" />} Refresh
        </button>
      </div>

      <div className="grid sm:grid-cols-3 gap-4">
        <Tile label="Backend" ok={backendOk} value={backendOk ? "Online" : "Offline"} />
        <Tile label="ML model" ok={health?.model_loaded} value={health?.model_loaded ? "Loaded" : "Not loaded"} />
        <Tile label="LLM mode"
          ok={llm?.mode === "production"}
          tone={llm?.mode === "production" ? "green" : llm?.mode ? "amber" : "gray"}
          value={llm?.mode || "unknown"} />
      </div>

      {err && (
        <Card className="bg-risk-highbg ring-1 ring-risk-highring border-0">
          <div className="flex items-center gap-2 text-risk-high text-sm">
            <Icon name="alert" className="w-4 h-4" /> {err}
          </div>
          <p className="text-xs text-ink-500 mt-1">API base: <code className="font-mono">{api.baseUrl}</code></p>
        </Card>
      )}

      <div className="grid lg:grid-cols-2 gap-5">
        <Card>
          <SectionTitle icon="health" title="Backend /health" />
          <KeyVal k="Status" v={health?.status} />
          <KeyVal k="Model loaded" v={String(health?.model_loaded ?? "—")} />
          <KeyVal k="Model path" v={<span className="font-mono text-xs break-all">{health?.model_path}</span>} />
          {health?.runtime_versions && (
            <div className="mt-3">
              <p className="text-xs text-ink-400 mb-1">Runtime versions</p>
              <JsonViewer data={health.runtime_versions} maxHeight="10rem" />
            </div>
          )}
        </Card>
        <Card>
          <SectionTitle icon="admin" title="LLM /generate-guidance/health" />
          <KeyVal k="Configured model" v={<span className="font-mono text-xs">{llm?.configured_model}</span>} />
          <KeyVal k="Runtime loaded" v={String(llm?.runtime_loaded ?? "—")} />
          <KeyVal k="Model spec" v={<span className="font-mono text-xs">{llm?.model_spec || "—"}</span>} />
          <KeyVal k="Mode" v={<Badge tone={llm?.mode === "production" ? "green" : "amber"}>{llm?.mode || "—"}</Badge>} />
          <p className="text-xs text-ink-400 mt-3">
            <code className="font-mono">local_demo</code> uses a deterministic safe generator; outputs still pass the safety validator.
          </p>
        </Card>
      </div>
    </div>
  );
}

function Tile({ label, value, ok, tone }) {
  const t = tone || (ok ? "green" : "gray");
  const map = {
    green: "bg-risk-lowbg text-risk-low ring-risk-lowring",
    amber: "bg-risk-modbg text-risk-mod ring-risk-modring",
    gray: "bg-ink-100 text-ink-600 ring-ink-200",
  };
  return (
    <div className={`rounded-xl2 ring-1 p-4 ${map[t]}`}>
      <div className="text-xs font-medium opacity-80">{label}</div>
      <div className="text-xl font-bold capitalize mt-0.5">{value}</div>
    </div>
  );
}
