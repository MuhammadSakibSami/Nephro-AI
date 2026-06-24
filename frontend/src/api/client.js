// Central API client for the kidney-risk backend.
// Configure the base URL via .env: VITE_API_BASE_URL=http://localhost:8000

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

async function request(path, { method = "GET", body, signal } = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const detail = data?.detail || data?.message || `HTTP ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

export const api = {
  baseUrl: BASE_URL,

  // ML two-stage prediction.
  predict: (features, signal) =>
    request("/predict", { method: "POST", body: features, signal }),

  // Full pipeline: predict -> RAG -> LLM -> safety validator.
  // Returns guidance + (enriched) `prediction` + admin metadata.
  generateGuidance: (features, signal) =>
    request("/generate-guidance", { method: "POST", body: features, signal }),

  // Backend / LLM health.
  health: (signal) => request("/health", { signal }),
  llmHealth: (signal) => request("/generate-guidance/health", { signal }),
};
