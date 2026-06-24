"""LLM Component — Model Backends (LLM_PLAN.md Phase 5).

Two interchangeable chat models behind one interface (`generate(messages) -> str`):

  * QwenChatModel    — production. Qwen3-4B-Instruct-2507 via transformers.
                       Runs in Colab/GPU. The MVP uses the BASE (un-fine-tuned)
                       model; Phase 8 swaps in the QLoRA adapter / merged model.
  * MockGuidanceModel — deterministic, dependency-free. Produces a well-formed,
                       safe guidance answer from the prompt so the full runtime
                       (retrieve -> prompt -> generate -> validate) is testable
                       offline. NOT a quality stand-in for Qwen3.

`get_model(spec)` selects: "qwen" | "mock" | "auto" (qwen if transformers+torch
are importable, else mock).
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
        return f"qwen:{self.model_name}"

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

        # 3) Move inputs to a real device. With device_map="auto", some
        # parameters may be on CPU/meta, so avoid blindly using a meta device.
        try:
            device = self.model.get_input_embeddings().weight.device
            if device.type == "meta":
                raise RuntimeError("input embedding is on meta device")
        except Exception:
            device = None
            for param in self.model.parameters():
                if getattr(param, "device", None) is not None and param.device.type != "meta":
                    device = param.device
                    break
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

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


# ---------------------------------------------------------------------------
# Offline: deterministic mock that returns a safe, well-formed answer
# ---------------------------------------------------------------------------
_RISK_RE = re.compile(r"Risk level:\s*.*\(([a-z]+)\)", re.I)
_CHUNK_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")
_FACTOR_RE = re.compile(r"-\s+(.+?)\s+\((?:raises|lowers) this risk level\)")


class MockGuidanceModel:
    """Deterministic safe-output model for offline pipeline testing.

    Reads the rendered prompt (risk level, top factors, retrieved chunk ids) and
    emits a complete, safe `GuidanceOutput`. `unsafe_first` lets a test force an
    unsafe first attempt to exercise the validator's retry/fallback path.
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
        factors = _FACTOR_RE.findall(user)[:2]

        if self.unsafe_first and self._calls == 1:
            # Deliberately unsafe (diagnosis) to test the guardrail.
            return self._render(
                summary="You have chronic kidney disease and must start medication.",
                chunk_ids=chunk_ids)

        at_risk = risk in {"moderate", "high"}
        focus = (" Your result was influenced by " + " and ".join(factors) + "."
                 if factors else "")
        summary = (f"Your kidney-risk screening result is {risk}. This is general "
                   f"information, not a diagnosis.{focus}")
        follow = ("Please check in with a clinician and ask about simple kidney tests "
                  "(a blood test and a urine test) to confirm your result."
                  if at_risk else
                  "Keep up regular check-ups and ask your clinician about routine kidney "
                  "checks if you have any concerns.")
        return self._render(
            summary=summary,
            food=("Aim for balanced meals with plenty of vegetables, fruits, and whole "
                  "foods, and go easy on very salty and heavily processed foods."),
            rest="Try for a steady sleep routine of about 7 to 9 hours each night.",
            routine=("Stay active most days with activities you enjoy, drink water through "
                     "the day, and take short breaks to ease stress."),
            follow=follow, chunk_ids=chunk_ids)

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
    spec = (spec or "auto").strip().lower()
    # Friendly aliases: the deployment "mode" names map to model specs.
    if spec in {"mock", "local_demo", "demo", "local"}:
        return MockGuidanceModel(**kwargs)
    if spec in {"qwen", "qwen3", "production", "prod"}:
        return QwenChatModel(**kwargs)
    if spec == "auto":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return QwenChatModel(**kwargs)
        except Exception as exc:
            print(f"[model] Qwen3 unavailable ({type(exc).__name__}); using MockGuidanceModel "
                  f"(offline, test-only).")
            return MockGuidanceModel()
    raise ValueError(f"Unknown model spec: {spec!r}")
