import React, { useMemo } from "react";
import { Card, SectionTitle, Icon, Spinner } from "../ui.jsx";
import {
  FORM_SECTIONS, NUMBER, BINARY, SELECT, validateValues,
} from "../../data/featureSchema.js";

export default function PatientForm({ values, setValues, onSubmit, loading }) {
  const { errors, missing, suppliedCount, requiredCount } = useMemo(
    () => validateValues(values), [values]
  );
  const completeness = requiredCount
    ? Math.round((Math.min(suppliedCount, requiredCount) / requiredCount) * 100) : 0;
  const hasErrors = Object.keys(errors).length > 0;

  const set = (key, v) => setValues((prev) => ({ ...prev, [key]: v }));

  return (
    <div className="space-y-5">
      <Card>
        <SectionTitle icon="health" title="Kidney-Risk Screening"
          sub="Enter routine, non-invasive values. Missing fields are estimated by the model."
          right={<CompletenessPill pct={completeness} />} />
        <p className="text-sm text-ink-500 -mt-1">
          This is a screening tool, not a diagnosis. Results come with educational guidance only.
        </p>
      </Card>

      {FORM_SECTIONS.map((section) => (
        <Card key={section.id}>
          <SectionTitle icon={section.icon} title={section.title} />
          <div className="grid sm:grid-cols-2 gap-x-5 gap-y-4">
            {section.fields.map((f) => (
              <Field key={f.key} field={f} value={values[f.key]} error={errors[f.key]}
                onChange={(v) => set(f.key, v)} />
            ))}
          </div>
        </Card>
      ))}

      <div className="sticky bottom-3 z-10">
        <Card className="flex flex-col sm:flex-row items-center justify-between gap-3 !py-4 shadow-pop">
          <div className="text-sm text-ink-500">
            {missing.length === 0 ? (
              <span className="text-risk-low font-medium">All recommended fields provided.</span>
            ) : (
              <span>{missing.length} field(s) missing — the model will estimate them.</span>
            )}
            {hasErrors && <span className="text-risk-high font-medium"> · Fix highlighted values.</span>}
          </div>
          <button className="btn-primary w-full sm:w-auto px-6" disabled={loading || hasErrors}
            onClick={onSubmit}>
            {loading ? <><Spinner /> Screening…</> : <><Icon name="spark" className="w-4 h-4" /> Run Kidney-Risk Screening</>}
          </button>
        </Card>
      </div>
    </div>
  );
}

function CompletenessPill({ pct }) {
  const tone = pct >= 90 ? "low" : pct >= 60 ? "mod" : "high";
  const color = { low: "text-risk-low", mod: "text-risk-mod", high: "text-risk-high" }[tone];
  return (
    <div className="text-right">
      <div className={`text-2xl font-bold ${color}`}>{pct}%</div>
      <div className="text-xs text-ink-400 -mt-1">complete</div>
    </div>
  );
}

function Field({ field, value, error, onChange }) {
  const v = value ?? "";
  return (
    <div>
      <label className="label flex items-center justify-between">
        <span>
          {field.label}
          {field.optional && <span className="text-ink-400 font-normal"> (optional)</span>}
        </span>
        {field.unit && <span className="text-xs text-ink-400 font-normal">{field.unit}</span>}
      </label>

      {field.type === SELECT ? (
        <select className="input" value={v} onChange={(e) => onChange(e.target.value)}>
          <option value="">Select…</option>
          {field.options.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      ) : field.type === BINARY ? (
        <YesNo value={v} onChange={onChange} />
      ) : (
        <input className={`input ${error ? "border-risk-high ring-2 ring-red-100" : ""}`}
          inputMode={field.type === NUMBER ? "decimal" : "text"}
          type={field.type === NUMBER ? "number" : "text"}
          placeholder={field.placeholder} value={v}
          onChange={(e) => onChange(e.target.value)} />
      )}

      <div className="mt-1 min-h-[1rem]">
        {error ? (
          <span className="text-xs text-risk-high">{error}</span>
        ) : field.normal ? (
          <span className="text-xs text-ink-400">Normal: {field.normal[0]}–{field.normal[1]} {field.unit}</span>
        ) : field.help ? (
          <span className="text-xs text-ink-400">{field.help}</span>
        ) : null}
      </div>
    </div>
  );
}

function YesNo({ value, onChange }) {
  const opts = [{ v: 1, label: "Yes" }, { v: 0, label: "No" }];
  return (
    <div className="flex gap-2">
      {opts.map((o) => (
        <button key={o.v} type="button" onClick={() => onChange(String(o.v))}
          className={`flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition
            ${String(value) === String(o.v)
              ? "border-brand-500 bg-brand-50 text-brand-700"
              : "border-ink-200 text-ink-600 hover:bg-ink-50"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
