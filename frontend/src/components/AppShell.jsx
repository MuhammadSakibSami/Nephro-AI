import React from "react";
import { Icon, Badge } from "./ui.jsx";
import { api } from "../api/client.js";

const NAV = [
  { id: "patient", label: "Patient View", icon: "patient" },
  { id: "admin", label: "Admin / Research", icon: "admin" },
  { id: "demo", label: "Demo Cases", icon: "demo" },
  { id: "health", label: "System Health", icon: "health" },
];

function ModeBadge({ mode }) {
  if (!mode) return <Badge tone="gray">LLM: unknown</Badge>;
  if (mode === "production") return <Badge tone="green">LLM: qwen</Badge>;
  if (mode === "local_demo") return <Badge tone="amber">LLM: local demo</Badge>;
  return <Badge tone="gray">LLM: {mode}</Badge>;
}

export default function AppShell({ view, setView, status, children }) {
  const backendOk = status?.backendOk;
  return (
    <div className="min-h-full lg:flex">
      {/* Sidebar (desktop) */}
      <aside className="hidden lg:flex lg:flex-col w-64 shrink-0 bg-white border-r border-ink-100 p-4 sticky top-0 h-screen">
        <Brand />
        <nav className="mt-6 space-y-1">
          {NAV.map((n) => (
            <NavButton key={n.id} item={n} active={view === n.id} onClick={() => setView(n.id)} />
          ))}
        </nav>
        <div className="mt-auto space-y-2 text-xs">
          <StatusRow ok={backendOk} label={`Backend ${backendOk ? "online" : "offline"}`} />
          <StatusRow ok={status?.modelLoaded} label={`ML model ${status?.modelLoaded ? "loaded" : "—"}`} />
          <div className="pt-1"><ModeBadge mode={status?.llmMode} /></div>
          <p className="text-ink-400 leading-snug pt-2">
            {api.baseUrl}
          </p>
        </div>
      </aside>

      {/* Mobile top nav */}
      <div className="lg:hidden sticky top-0 z-20 bg-white border-b border-ink-100">
        <div className="flex items-center justify-between px-4 py-3">
          <Brand compact />
          <ModeBadge mode={status?.llmMode} />
        </div>
        <nav className="flex gap-1 px-2 pb-2 overflow-x-auto scroll-thin">
          {NAV.map((n) => (
            <button key={n.id} onClick={() => setView(n.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm whitespace-nowrap
                ${view === n.id ? "bg-brand-50 text-brand-700 font-semibold" : "text-ink-500"}`}>
              <Icon name={n.icon} className="w-4 h-4" /> {n.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Main */}
      <main className="flex-1 min-w-0">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6">{children}</div>
        <DisclaimerFooter />
      </main>
    </div>
  );
}

function Brand({ compact }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="grid place-items-center w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-sm">
        <Icon name="health" className="w-5 h-5" />
      </span>
      {!compact && (
        <div className="leading-tight">
          <div className="font-extrabold text-ink-900">Nephro AI</div>
          <div className="text-[11px] text-ink-400 font-medium">Kidney Risk Screening</div>
        </div>
      )}
      {compact && <span className="font-extrabold text-ink-900">Nephro AI</span>}
    </div>
  );
}

function NavButton({ item, active, onClick }) {
  return (
    <button onClick={onClick}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition
        ${active ? "bg-brand-50 text-brand-700" : "text-ink-600 hover:bg-ink-50"}`}>
      <Icon name={item.icon} className="w-5 h-5" />
      {item.label}
    </button>
  );
}

function StatusRow({ ok, label }) {
  return (
    <div className="flex items-center gap-2 text-ink-500">
      <span className={`w-2 h-2 rounded-full ${ok ? "bg-risk-low" : "bg-ink-300"}`} />
      {label}
    </div>
  );
}

function DisclaimerFooter() {
  return (
    <footer className="border-t border-ink-100 mt-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-5 text-xs text-ink-400 flex items-start gap-2">
        <Icon name="shield" className="w-4 h-4 mt-0.5 shrink-0" />
        <p>
          The LLM component provides personalized, guideline-grounded kidney-risk
          <span className="font-medium text-ink-500"> education</span> after ML-based screening.
          It does not diagnose CKD, prescribe treatment, or replace clinician consultation.
        </p>
      </div>
    </footer>
  );
}
