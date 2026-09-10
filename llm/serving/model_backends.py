"""LLM Component — Model Backends (LLM_PLAN.md Phase 5).

Three interchangeable chat models behind one interface (`generate(messages) -> str`):

  * QwenChatModel     — production A. Qwen3-4B-Instruct-2507 via HuggingFace
                        transformers. GPU / Colab path. Phase 8 swaps in the
                        QLoRA adapter. Set CKD_LLM_MODEL=qwen.

  * OllamaChatModel   — production B. Any Ollama-served model via its local REST
                        API (no Python ML stack needed). Recommended for local
                        laptops with ≥4 GB VRAM (e.g. RTX 3050 Ti): pull
                        qwen3:4b-instruct-2507-q4_K_M (~3 GB) and set
                        CKD_LLM_MODEL=ollama. Switch to QwenChatModel later with
                        a one-line env change — same interface, no other edits.

  * MockGuidanceModel — deterministic, dependency-free. Produces a well-formed,
                        safe guidance answer so the full runtime (retrieve →
                        prompt → generate → validate) is testable offline. NOT
                        a quality stand-in for either production backend.

`get_model(spec)` selects:
  "qwen" | "ollama" | "mock" | "auto" (qwen if transformers+torch available,
  else ollama if Ollama is running locally, else mock).
"""
from __future__ import annotations

import re
from typing import Dict, List

from .contract import GROUNDING_LABEL, GuidanceOutput, OUTPUT_SECTION_LABELS


# ---------------------------------------------------------------------------
# Production: Qwen3-4B-Instruct-2507
# ---------------------------------------------------------------------------
class QwenChatModel:
    """Qwen3-4B-Instruct-2507 (non-thinking instruct variant) via transformers."""

    DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 max_new_tokens: int = 700, temperature: float = 0.3,
                 top_p: float = 0.9, adapter_path: str | None = None):
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._torch = torch

        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto",
        )
        # Phase 8: attach a fine-tuned QLoRA adapter if provided.
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    @property
    def spec(self) -> str:
        # Adapter status is baked into `spec` (not just `adapter_path`) because
        # every caller that logs/records a run reads `spec`, not internals - so
        # "is the fine-tuned model actually active" is answerable from a run
        # record alone, without inspecting the process that produced it.
        if self.adapter_path:
            from pathlib import Path
            p = Path(self.adapter_path)
            # Last two segments (e.g. "v1/adapter"): the leaf alone is often
            # just "adapter" for every version, which defeats traceability.
            tag = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
            return f"qwen:{self.model_name}+lora:{tag}"
        return f"qwen:{self.model_name}"

    def _device(self):
        """Real (non-meta) device to move inputs to. With device_map="auto",
        some parameters may be on CPU/meta, so avoid blindly using a meta device."""
        torch = self._torch
        try:
            device = self.model.get_input_embeddings().weight.device
            if device.type == "meta":
                raise RuntimeError("input embedding is on meta device")
            return device
        except Exception:
            for param in self.model.parameters():
                if getattr(param, "device", None) is not None and param.device.type != "meta":
                    return param.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a reply from chat messages.

        Important Colab/Transformers fix:
        Some tokenizer/chat-template paths return a BatchEncoding-like object.
        Passing that object positionally to ``model.generate`` or reading
        ``inputs.shape`` can trigger ``KeyError: 'shape'``.  To avoid that,
        we first render the chat prompt as plain text, then tokenize it into
        a normal dict of tensors, call ``generate(**model_inputs)``, and use
        ``model_inputs["input_ids"].shape`` for prompt-length slicing.
        """
        torch = self._torch

        # 1) Render messages using the model's official chat template.
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2) Tokenize the prompt into a BatchEncoding/dict of tensors.
        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        # 3) Move inputs to a real device.
        model_inputs = {k: v.to(self._device()) for k, v in model_inputs.items()}

        # 4) Generate from keyword inputs. Do not pass BatchEncoding positionally.
        with torch.inference_mode():
            out = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 5) Decode only the newly generated tokens, not the prompt.
        prompt_len = model_inputs["input_ids"].shape[-1]
        gen = out[0][prompt_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        """Generate replies for several independent conversations in one
        forward pass. LEFT-padding means every row's real prompt ends at the
        same index, so generation starts at the same column for the whole
        batch and slicing out the new tokens doesn't need per-row lengths.
        Used by eval harnesses (Phase 9) to cut GPU-bound wall-clock time —
        not used by the single-request serving path.
        """
        torch = self._torch

        prompt_texts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]

        old_side = self.tokenizer.padding_side
        old_pad = self.tokenizer.pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        try:
            model_inputs = self.tokenizer(
                prompt_texts, return_tensors="pt", truncation=True,
                max_length=4096, padding=True,
            )
        finally:
            self.tokenizer.padding_side = old_side
            self.tokenizer.pad_token = old_pad

        model_inputs = {k: v.to(self._device()) for k, v in model_inputs.items()}

        with torch.inference_mode():
            out = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = model_inputs["input_ids"].shape[-1]
        return [
            self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in out
        ]


# ---------------------------------------------------------------------------
# Production B: any Ollama-served model via local REST API
# ---------------------------------------------------------------------------
class OllamaChatModel:
    """Qwen3-4B (or any model) served by Ollama via its local REST API.

    No Python ML stack required — Ollama handles model loading and GPU offload.
    Ideal for local laptops with limited VRAM (e.g. RTX 3050 Ti / 4 GB):
      Q4_K_M of Qwen3-4B needs ~3 GB VRAM, leaving 1 GB headroom.

    Setup (one-time):
      1. Install Ollama from https://ollama.com
      2. ollama pull qwen3:4b-instruct-2507-q4_K_M
      3. Set CKD_LLM_MODEL=ollama in .env  (Ollama serve starts automatically)

    To switch to the HuggingFace backend later (e.g. on a Colab/GPU server):
      change CKD_LLM_MODEL=qwen — no other code changes needed.

    Optional env vars (read by get_model):
      OLLAMA_MODEL    override the default model tag (default: qwen3:4b-instruct-2507-q4_K_M)
      OLLAMA_BASE_URL override the Ollama server URL (default: http://localhost:11434)
    """

    DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 900,
        temperature: float = 0.65,
        top_p: float = 0.9,
    ):
        import requests as _r  # already in requirements.txt
        self._requests = _r
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    @property
    def spec(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """POST to Ollama /api/chat (non-streaming) and return the assistant reply."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        resp = self._requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Offline: deterministic mock that returns a safe, well-formed answer
# ---------------------------------------------------------------------------
_RISK_RE = re.compile(r"Risk level:\s*.*\(([a-z]+)\)", re.I)
_CHUNK_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")
_FACTOR_RE = re.compile(r"-\s+(.+?)\s+\((?:raises|lowers) this risk level\)")
_PROFILE_FLAG_RE = re.compile(r"-\s+(.+?):\s+(Yes|No)", re.I)
_PROFILE_NUM_RE  = re.compile(r"-\s+(.+?):\s+([\d.]+)")


def _extract_profile_flags(user: str) -> Dict[str, bool]:
    """Extract Yes/No boolean flags from the PATIENT PROFILE block."""
    flags: Dict[str, bool] = {}
    in_profile = False
    for line in user.splitlines():
        if "PATIENT PROFILE" in line:
            in_profile = True
            continue
        if in_profile and line.strip().startswith("TOP FACTORS"):
            break
        if in_profile:
            m = _PROFILE_FLAG_RE.search(line)
            if m:
                flags[m.group(1).lower()] = m.group(2).lower() == "yes"
    return flags


def _extract_profile_nums(user: str) -> Dict[str, float]:
    """Extract numeric values from the PATIENT PROFILE block."""
    nums: Dict[str, float] = {}
    in_profile = False
    for line in user.splitlines():
        if "PATIENT PROFILE" in line:
            in_profile = True
            continue
        if in_profile and line.strip().startswith("TOP FACTORS"):
            break
        if in_profile:
            m = _PROFILE_NUM_RE.search(line)
            if m:
                try:
                    nums[m.group(1).lower()] = float(m.group(2))
                except ValueError:
                    pass
    return nums


class MockGuidanceModel:
    """Deterministic safe-output model for offline pipeline testing.

    Reads the rendered prompt (risk level, patient profile, top factors, chunk
    ids) and emits a well-formed, patient-specific `GuidanceOutput`.
    `unsafe_first` lets a test force an unsafe first attempt to exercise the
    validator's retry/fallback path.
    """

    def __init__(self, unsafe_first: bool = False):
        self.unsafe_first = unsafe_first
        self._calls = 0

    @property
    def spec(self) -> str:
        return "mock"

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self._calls += 1
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        risk = (_RISK_RE.search(user).group(1).lower() if _RISK_RE.search(user) else "low")
        chunk_ids = _CHUNK_RE.findall(user)[:2]
        factors = _FACTOR_RE.findall(user)[:3]
        flags = _extract_profile_flags(user)
        nums  = _extract_profile_nums(user)

        if self.unsafe_first and self._calls == 1:
            return self._render(
                summary="You have chronic kidney disease and must start medication.",
                chunk_ids=chunk_ids)

        at_risk = risk in {"moderate", "high"}

        # --- derive patient-specific cues from extracted profile ---
        has_dm  = flags.get("has diabetes (diagnosed)", False)
        has_pre = flags.get("has prediabetes / borderline diabetes", False)
        has_htn = flags.get("has high blood pressure (diagnosed)", False)
        on_bp   = flags.get("currently on blood pressure medicine", False)
        has_chol = flags.get("has high cholesterol (diagnosed)", False)
        sbp = nums.get("systolic blood pressure", None)
        bmi_v = nums.get("bmi", None)

        any_dm = has_dm or has_pre
        high_bp = has_htn or (sbp is not None and sbp > 130)

        # Summary
        focus = (" Key influences include " + " and ".join(factors) + "." if factors else "")
        summary = (f"Your screening shows a {risk} risk for kidney problems.{focus} "
                   "This is not a diagnosis — it is a screening result to guide next steps.")

        # Food habits
        food_parts: List[str] = [
            "Eat a balanced diet with plenty of vegetables, fruits, and whole grains."
        ]
        if any_dm:
            food_parts.append(
                "Because your blood sugar is elevated, limit sugary drinks, white bread, "
                "and processed snacks, and choose low-glycaemic foods instead.")
        if high_bp:
            food_parts.append(
                "Since blood pressure is a concern, try to reduce salt — avoid adding "
                "extra salt at the table and go easy on salty packaged foods.")
        if has_chol:
            food_parts.append(
                "For your cholesterol, favour heart-healthy fats (olive oil, fish, nuts) "
                "and limit fried or fatty processed foods.")
        if bmi_v is not None and bmi_v >= 25:
            food_parts.append(
                "Keeping portions moderate and choosing filling whole foods can help "
                "support a healthy weight over time.")
        food = " ".join(food_parts)

        # Rest
        rest = ("Aim for 7 to 9 hours of sleep each night. A regular bedtime helps your "
                "body recover. Keep your bedroom quiet and dark, and avoid screens close "
                "to bedtime.")

        # Daily routine
        routine_parts: List[str] = [
            "Try to move your body each day — even a 20-30 minute walk makes a difference."
        ]
        if on_bp or has_htn:
            routine_parts.append(
                "Check your blood pressure regularly and write down the readings to share "
                "with your doctor.")
        if has_dm:
            routine_parts.append(
                "Track your blood sugar daily and note any readings that seem unusually "
                "high or low.")
        routine_parts.append("Drink enough water throughout the day to stay hydrated.")
        routine = " ".join(routine_parts)

        # Follow-up
        follow_parts: List[str] = []
        if at_risk:
            follow_parts.append(
                f"Because your screening result is {risk}, it is important to see a "
                "doctor or kidney specialist. Ask for a urine test (to check for protein) "
                "and a blood test (to measure kidney function).")
        if any_dm:
            follow_parts.append("Ask your doctor about an HbA1c test to check your blood sugar control.")
        if has_htn or high_bp:
            follow_parts.append("Have your blood pressure reviewed by your doctor.")
        if not follow_parts:
            follow_parts.append("Keep up regular check-ups and ask your clinician about "
                                "routine kidney-health tests.")
        follow = " ".join(follow_parts)

        return self._render(summary=summary, food=food, rest=rest,
                            routine=routine, follow=follow, chunk_ids=chunk_ids)

    @staticmethod
    def _render(summary: str, chunk_ids: List[str],
                food: str = "Eat balanced, mostly whole foods.",
                rest: str = "Keep a steady sleep routine.",
                routine: str = "Stay gently active and hydrated.",
                follow: str = "Check in with your clinician about routine kidney checks.") -> str:
        out = GuidanceOutput(
            summary=summary, food_habits=food, rest_and_sleep=rest,
            daily_routine=routine, follow_up_reminder=follow, grounding=chunk_ids,
        )
        return out.render()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_model(spec: str = "auto", **kwargs):
    """Return a model backend for the given spec string.

    Specs / aliases
    ---------------
    qwen         : QwenChatModel (HuggingFace transformers, GPU/Colab)
                   aliases: qwen3, production, prod
    ollama       : OllamaChatModel (local Ollama REST API, laptop GPU)
                   Reads OLLAMA_MODEL / OLLAMA_BASE_URL from env if set.
    mock         : MockGuidanceModel (offline, deterministic)
                   aliases: local_demo, demo, local
    auto         : try qwen → try ollama → fall back to mock
    """
    import os

    spec = (spec or "auto").strip().lower()

    if spec in {"mock", "local_demo", "demo", "local"}:
        return MockGuidanceModel(**kwargs)

    if spec in {"qwen", "qwen3", "production", "prod"}:
        return QwenChatModel(**kwargs)

    if spec == "ollama":
        model = kwargs.pop("model", os.getenv("OLLAMA_MODEL", OllamaChatModel.DEFAULT_MODEL))
        base_url = kwargs.pop("base_url", os.getenv("OLLAMA_BASE_URL", OllamaChatModel.DEFAULT_BASE_URL))
        return OllamaChatModel(model=model, base_url=base_url, **kwargs)

    if spec == "auto":
        # Priority: HuggingFace transformers → Ollama (if running) → mock
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return QwenChatModel(**kwargs)
        except Exception as exc:
            print(f"[model] Qwen3 (transformers) unavailable ({type(exc).__name__}); trying Ollama.")
        try:
            import requests as _r
            if _r.get(f"{OllamaChatModel.DEFAULT_BASE_URL}/api/tags", timeout=2).ok:
                print("[model] Ollama is running — using OllamaChatModel.")
                return OllamaChatModel(
                    model=os.getenv("OLLAMA_MODEL", OllamaChatModel.DEFAULT_MODEL),
                    base_url=os.getenv("OLLAMA_BASE_URL", OllamaChatModel.DEFAULT_BASE_URL),
                )
        except Exception:
            pass
        print("[model] Neither transformers nor Ollama available; using MockGuidanceModel (offline, test-only).")
        return MockGuidanceModel()

    raise ValueError(f"Unknown model spec: {spec!r}")
