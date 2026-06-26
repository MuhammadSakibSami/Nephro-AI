"""LLM Component — Serving glue for the app (LLM_PLAN.md Phase 10, minimal MVP).

Wires the LLM runtime (RAG + base/demo model + safety validator) into a single
response builder used by the FastAPI `/generate-guidance` endpoint and by the
supervisor demo.

Model selection (env `CKD_LLM_MODEL`, default "auto"):
  * "qwen"   -> Qwen3-4B via HuggingFace transformers (GPU / Colab)  -> mode "production"
  * "ollama" -> Qwen3-4B via Ollama REST API (laptop GPU, no ML stack) -> mode "production"
  * "auto"   -> qwen if transformers available, else ollama if running, else mock
  * falls back to the deterministic MockGuidanceModel                -> mode "local_demo"

**Local demo mode** lets the app run with no GPU: guidance is produced by a
deterministic safe generator, still passed through the SAME safety validator, and
clearly labelled. The real Qwen3 evaluation evidence is the frozen Phase-6 results.

This module imports only `ckd_risk_app.llm.*` (no ML/joblib stack), so it can be
imported for the demo without loading the cascade model.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

# Import the LLM package whether the project is run as the top-level repo
# (``uvicorn backend.main:app`` from the repo root → ``backend`` and ``llm`` are
# sibling top-level packages) OR as a sub-package (``ckd_risk_app.backend...``).
try:  # standalone repo layout (backend/ and llm/ at the root)
    import sys as _sys
    import pathlib as _pathlib
    _ROOT = str(_pathlib.Path(__file__).resolve().parents[1])  # dir containing backend/ + llm/
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from llm.serving.contract import LLMInput
    from llm.serving.llm_runtime import GuidanceRuntime
    from llm.serving.prompts import SCOPE_STATEMENT
except ImportError:  # nested package layout (ckd_risk_app.backend / ckd_risk_app.llm)
    from ..llm.serving.contract import LLMInput
    from ..llm.serving.llm_runtime import GuidanceRuntime
    from ..llm.serving.prompts import SCOPE_STATEMENT

_runtime: Optional[GuidanceRuntime] = None

LOCAL_DEMO_NOTE = (
    "Local demo mode: guidance is produced by a deterministic safe generator "
    "(not the production LLM), but still passes through the same safety validator. "
    "The production system uses Qwen3-4B-Instruct-2507 + guideline RAG; see the "
    "frozen Phase-6 evaluation for real-model results."
)


def get_runtime() -> GuidanceRuntime:
    """Lazy singleton runtime (loads the retriever + model once)."""
    global _runtime
    if _runtime is None:
        spec = os.getenv("CKD_LLM_MODEL", "auto")
        _runtime = GuidanceRuntime(model=spec)
    return _runtime


def runtime_mode(rt: GuidanceRuntime) -> str:
    spec = rt.model.spec
    if spec.startswith("qwen") or spec.startswith("ollama"):
        return "production"
    return "local_demo"


def build_guidance_response(predict_response: Union[Dict[str, Any], LLMInput],
                            k: int = 5) -> Dict[str, Any]:
    """Run the LLM runtime and shape an API-ready response.

    `predict_response` is a `/predict` response dict (or an `LLMInput`).
    """
    rt = get_runtime()
    result = rt.generate_guidance(predict_response, k=k)
    mode = runtime_mode(rt)

    payload = result.to_dict()
    payload["mode"] = mode
    payload["scope_statement"] = SCOPE_STATEMENT
    payload["disclaimer"] = (
        SCOPE_STATEMENT if mode == "production"
        else f"{SCOPE_STATEMENT} {LOCAL_DEMO_NOTE}"
    )
    # Include the full ML prediction so the frontend gets everything in one call
    # (admin Prediction/Input-Quality/Explanation panels + patient key signals).
    if isinstance(predict_response, dict):
        payload["prediction"] = predict_response
    return payload


def health() -> Dict[str, Any]:
    """Lightweight status for the endpoint (does not force model load)."""
    spec = os.getenv("CKD_LLM_MODEL", "auto")
    loaded = _runtime is not None
    return {
        "configured_model": spec,
        "runtime_loaded": loaded,
        "model_spec": _runtime.model.spec if loaded else None,
        "mode": runtime_mode(_runtime) if loaded else None,
    }
