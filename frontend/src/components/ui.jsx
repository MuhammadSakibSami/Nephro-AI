import React, { useState } from "react";

// --- Minimal inline icon set (no icon dependency) ---------------------------
const PATHS = {
  user: "M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10ZM3 22a9 9 0 0 1 18 0",
  ruler: "M3 8l5-5 13 13-5 5L3 8Zm5 0l2 2m1-5 2 2m1-5 2 2M6 11l2 2",
  heart: "M12 21s-7.5-4.6-10-9.2C.6 9 1.5 5.5 4.6 4.6 7 3.9 9 5 12 8c3-3 5-4.1 7.4-3.4C22.5 5.5 23.4 9 22 11.8 19.5 16.4 12 21 12 21Z",
  flask: "M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 18l-5-9V3",
  clipboard: "M9 4h6a1 1 0 0 1 1 1v1h2a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h2V5a1 1 0 0 1 1-1Z",
  patient: "M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10ZM3 22a9 9 0 0 1 18 0",
  admin: "M12 3l8 4v5c0 4.4-3.1 8.4-8 9-4.9-.6-8-4.6-8-9V7l8-4Z",
  demo: "M8 5v14l11-7L8 5Z",
  health: "M3 12h4l2 6 4-12 2 6h6",
  shield: "M12 3l8 4v5c0 4.4-3.1 8.4-8 9-4.9-.6-8-4.6-8-9V7l8-4Z",
  download: "M12 3v12m0 0 4-4m-4 4-4-4M5 21h14",
  copy: "M9 9h10v10H9zM5 15H4V5h10v1",
  alert: "M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z",
  check: "M20 6 9 17l-5-5",
  chevron: "M6 9l6 6 6-6",
  spark: "M5 3v4M3 5h4M6 17v4M4 19h4M13 3l2.5 6.5L22 12l-6.5 2.5L13 21l-2.5-6.5L4 12l6.5-2.5L13 3Z",
};

export function Icon({ name, className = "w-5 h-5" }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <path d={PATHS[name] || PATHS.user} />
    </svg>
  );
}

export function Card({ className = "", children }) {
  return <div className={`card p-5 ${className}`}>{children}</div>;
}

export function SectionTitle({ icon, title, sub, right }) {
  return (
    <div className="flex items-start justify-between gap-3 mb-4">
      <div className="flex items-center gap-3">
        {icon && (
          <span className="grid place-items-center w-9 h-9 rounded-lg bg-brand-50 text-brand-600">
            <Icon name={icon} />
          </span>
        )}
        <div>
          <h3 className="font-semibold text-ink-900 leading-tight">{title}</h3>
          {sub && <p className="text-sm text-ink-500">{sub}</p>}
        </div>
      </div>
      {right}
    </div>
  );
}

export function Chip({ children, tone = "brand" }) {
  const tones = {
    brand: "bg-brand-50 text-brand-700",
    gray: "bg-ink-100 text-ink-700",
    low: "bg-risk-lowbg text-risk-low",
    mod: "bg-risk-modbg text-risk-mod",
    high: "bg-risk-highbg text-risk-high",
  };
  return <span className={`chip ${tones[tone] || tones.brand}`}>{children}</span>;
}

export function Badge({ children, tone = "gray" }) {
  const tones = {
    gray: "bg-ink-100 text-ink-600", green: "bg-risk-lowbg text-risk-low",
    amber: "bg-risk-modbg text-risk-mod", red: "bg-risk-highbg text-risk-high",
    brand: "bg-brand-50 text-brand-700",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Spinner({ className = "w-4 h-4" }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-90" fill="currentColor"
        d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4Z" />
    </svg>
  );
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="flex gap-1 overflow-x-auto scroll-thin border-b border-ink-100 mb-5">
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={`px-4 py-2.5 text-sm font-medium whitespace-nowrap rounded-t-lg transition
            ${active === t.id
              ? "text-brand-700 border-b-2 border-brand-600 bg-brand-50/40"
              : "text-ink-500 hover:text-ink-800 border-b-2 border-transparent"}`}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Collapsible({ title, defaultOpen = false, right, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="card overflow-hidden">
      <button onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-3.5 text-left hover:bg-ink-50/60">
        <span className="font-semibold text-ink-800">{title}</span>
        <span className="flex items-center gap-3">
          {right}
          <Icon name="chevron" className={`w-5 h-5 text-ink-400 transition ${open ? "rotate-180" : ""}`} />
        </span>
      </button>
      {open && <div className="px-5 pb-5 pt-1 border-t border-ink-100">{children}</div>}
    </div>
  );
}

export function KeyVal({ k, v, mono }) {
  return (
    <div className="flex items-start justify-between gap-4 py-1.5 border-b border-ink-50 last:border-0">
      <span className="text-sm text-ink-500">{k}</span>
      <span className={`text-sm text-ink-900 text-right ${mono ? "font-mono" : "font-medium"}`}>
        {v === undefined || v === null || v === "" ? <span className="text-ink-300">—</span> : v}
      </span>
    </div>
  );
}

export function JsonViewer({ data, maxHeight = "20rem" }) {
  return (
    <pre style={{ maxHeight }}
      className="scroll-thin overflow-auto rounded-lg bg-ink-900 text-ink-50 text-xs leading-relaxed p-4 font-mono">
      {JSON.stringify(data ?? {}, null, 2)}
    </pre>
  );
}

export function EmptyState({ icon = "spark", title, sub }) {
  return (
    <div className="text-center py-12 px-6">
      <span className="inline-grid place-items-center w-14 h-14 rounded-2xl bg-brand-50 text-brand-500 mb-3">
        <Icon name={icon} className="w-7 h-7" />
      </span>
      <h3 className="font-semibold text-ink-800">{title}</h3>
      {sub && <p className="text-sm text-ink-500 mt-1 max-w-sm mx-auto">{sub}</p>}
    </div>
  );
}
