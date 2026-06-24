import React, { useCallback, useEffect, useState } from "react";
import AppShell from "./components/AppShell.jsx";
import PatientView from "./components/patient/PatientView.jsx";
import AdminDashboard from "./components/admin/AdminDashboard.jsx";
import DemoCases from "./components/DemoCases.jsx";
import SystemHealth from "./components/SystemHealth.jsx";
import { api } from "./api/client.js";
import { buildPayload } from "./data/featureSchema.js";
import { DEMO_CASES } from "./data/demoCases.js";

export default function App() {
  const [view, setView] = useState("patient");
  const [values, setValues] = useState({});
  const [guidance, setGuidance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingDemoId, setLoadingDemoId] = useState(null);
  const [error, setError] = useState("");
  const [status, setStatus] = useState({ backendOk: false, modelLoaded: false, llmMode: null });

  // Poll health for the sidebar status.
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const [h, l] = await Promise.allSettled([api.health(), api.llmHealth()]);
        if (!alive) return;
        setStatus({
          backendOk: h.status === "fulfilled" || l.status === "fulfilled",
          modelLoaded: h.status === "fulfilled" ? !!h.value?.model_loaded : false,
          llmMode: l.status === "fulfilled" ? l.value?.mode : null,
        });
      } catch {
        if (alive) setStatus((s) => ({ ...s, backendOk: false }));
      }
    };
    check();
    const t = setInterval(check, 15000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const runScreening = useCallback(async (payload) => {
    setLoading(true); setError("");
    try {
      const result = await api.generateGuidance(payload);
      setGuidance(result);
      // refresh llm mode from the live response
      setStatus((s) => ({ ...s, backendOk: true, llmMode: result?.mode || s.llmMode }));
      return result;
    } catch (e) {
      setError(e.message || String(e));
      setGuidance(null);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const onSubmitForm = useCallback(async () => {
    try {
      await runScreening(buildPayload(values));
    } catch { /* surfaced via error banner */ }
  }, [values, runScreening]);

  const onRunDemo = useCallback(async (caseOrId) => {
    const c = typeof caseOrId === "string" ? DEMO_CASES.find((x) => x.id === caseOrId) : caseOrId;
    if (!c) return;
    setValues(c.values);
    setLoadingDemoId(c.id);
    try {
      const { patient_id, ...payload } = c.values; // drop frontend-only field
      await runScreening(payload);
      setView("patient");
    } catch { /* error banner */ }
    finally { setLoadingDemoId(null); }
  }, [runScreening]);

  const onReset = useCallback(() => { setGuidance(null); setError(""); }, []);

  return (
    <AppShell view={view} setView={setView} status={status}>
      {view === "patient" && (
        <PatientView values={values} setValues={setValues} guidance={guidance}
          loading={loading} error={error} onSubmit={onSubmitForm} onReset={onReset} />
      )}
      {view === "admin" && <AdminDashboard guidance={guidance} onRunDemo={onRunDemo} />}
      {view === "demo" && <DemoCases onRun={onRunDemo} loadingId={loadingDemoId} />}
      {view === "health" && <SystemHealth />}
    </AppShell>
  );
}
