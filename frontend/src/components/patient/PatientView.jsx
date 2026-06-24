import React from "react";
import PatientForm from "./PatientForm.jsx";
import PatientResult from "./PatientResult.jsx";
import { Icon } from "../ui.jsx";

export default function PatientView({ values, setValues, guidance, loading, error, onSubmit, onReset }) {
  return (
    <div className="space-y-5">
      {error && <ErrorBanner error={error} />}
      {guidance ? (
        <PatientResult guidance={guidance} onReset={onReset} />
      ) : (
        <PatientForm values={values} setValues={setValues} onSubmit={onSubmit} loading={loading} />
      )}
    </div>
  );
}

function ErrorBanner({ error }) {
  return (
    <div className="rounded-xl2 bg-risk-highbg ring-1 ring-risk-highring p-4 flex items-start gap-3">
      <Icon name="alert" className="w-5 h-5 text-risk-high mt-0.5 shrink-0" />
      <div className="text-sm">
        <p className="font-semibold text-risk-high">Could not reach the screening service</p>
        <p className="text-ink-600 mt-0.5">{error}</p>
        <p className="text-ink-400 mt-1">
          Make sure the backend is running and <code className="font-mono">VITE_API_BASE_URL</code> points to it.
        </p>
      </div>
    </div>
  );
}
