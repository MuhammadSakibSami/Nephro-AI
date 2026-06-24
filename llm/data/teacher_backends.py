"""LLM Component — Teacher backends for RAFT data (LLM_PLAN.md Phase 7).

Pluggable "teachers" that produce a gold answer for a training example given an
`LLMInput` and the retrieved chunks (golden + distractors). All teachers route
their output through `safety_validator.enforce`, so every gold answer is
guaranteed safe + well-formed before it can enter the dataset.

  * SelfDistillTeacher — base Qwen3 (or offline mock) + RAG + validator. Bulk
    (~80%) of the data. No external API; internalizes the validated behavior.
  * GeminiTeacher      — Google Gemini (free tier) for the ~20% subset (harder /
    emergency / diverse cases). Only synthetic, de-identified profiles are sent.

`get_teacher(spec, ...)` returns one by name: "self" | "gemini".

Gold generation uses the FULL system prompt (best teacher output); the training
example itself is later paired with the COMPACT prompt (Phase 7 generate_dataset).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from ..serving.prompts import build_messages, build_system_prompt, build_user_prompt
from ..serving.safety_validator import context_from_llm_input, enforce

_RETRY_NOTE = ("\n\n(Reminder: reply using the EXACT output format with all five "
               "labeled sections. Do not diagnose, name or change any medication, "
               "or give numeric nutrient targets.)")


class SelfDistillTeacher:
    """Base chat model + RAG context + validator -> guaranteed-safe gold answer."""

    def __init__(self, model, max_retries: int = 1):
        self.model = model
        self.max_retries = max_retries

    @property
    def spec(self) -> str:
        return f"self:{getattr(self.model, 'spec', 'model')}"

    def generate_gold(self, inp, chunks: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        ctx = context_from_llm_input(inp)
        captured = {"raw": ""}

        def _gen(attempt: int) -> str:
            messages = build_messages(inp, chunks)
            if attempt > 0:
                messages[-1]["content"] += _RETRY_NOTE
            raw = self.model.generate(messages)
            captured["raw"] = raw
            return raw

        res = enforce(_gen, ctx, max_retries=self.max_retries)
        return res.safe_text, captured["raw"], res.status


class GeminiTeacher:
    """Google Gemini teacher (free tier friendly). Lazy-imports google-genai."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash",
                 max_retries: int = 1, temperature: float = 0.4,
                 request_delay_s: float = 4.0, max_output_tokens: int = 900):
        from google import genai  # lazy
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.temperature = temperature
        self.request_delay_s = request_delay_s   # be gentle on free-tier RPM
        self.max_output_tokens = max_output_tokens

    @property
    def spec(self) -> str:
        return f"gemini:{self.model_name}"

    def _call(self, system: str, user: str) -> str:
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        if self.request_delay_s:
            time.sleep(self.request_delay_s)
        return (resp.text or "").strip()

    def generate_gold(self, inp, chunks: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        ctx = context_from_llm_input(inp)
        system = build_system_prompt()
        captured = {"raw": ""}

        def _gen(attempt: int) -> str:
            user = build_user_prompt(inp, chunks)
            if attempt > 0:
                user += _RETRY_NOTE
            raw = self._call(system, user)
            captured["raw"] = raw
            return raw

        res = enforce(_gen, ctx, max_retries=self.max_retries)
        return res.safe_text, captured["raw"], res.status


def get_teacher(spec: str, **kwargs):
    spec = (spec or "self").strip().lower()
    if spec == "self":
        from ..serving.model_backends import get_model
        model = kwargs.pop("model", None)
        if model is None or isinstance(model, str):
            model = get_model(model or "auto", **kwargs)
        return SelfDistillTeacher(model)
    if spec == "gemini":
        return GeminiTeacher(**kwargs)
    raise ValueError(f"Unknown teacher spec: {spec!r}")
