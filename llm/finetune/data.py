"""LLM Component — Fine-tuning data pipeline (LLM_PLAN.md Phase 8).

Turns the Phase-7 dataset (`train.jsonl` / `val.jsonl`) into tokenized SFT
examples with **loss masked to the assistant turn only**.

Masking strategy (tokenizer-version independent):

    prompt_text = chat_template(messages[:-1], add_generation_prompt=True)
    full_text   = chat_template(messages)            # includes assistant + <|im_end|>
    labels      = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

`prompt_text` is always a literal prefix of `full_text` for the Qwen3 template,
so masking the first `len(prompt_ids)` tokens supervises exactly the assistant
response (including its end-of-turn token, so the model learns to stop).

The pure functions here (`mask_prompt_prefix`, `fit_to_length`) take plain token
id lists, so the masking logic is unit-testable without torch or a real
tokenizer. Torch is imported lazily inside the collator only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_examples(path: str | Path, allow_empty: bool = False) -> List[Dict[str, Any]]:
    """Read a Phase-7 JSONL file. Each row must carry a 3-turn `messages` list.

    `allow_empty=True` returns `[]` for a present-but-empty file instead of
    raising — used for `val.jsonl`, which Phase 7 leaves empty on small runs
    (`int(n * val_frac)` rounds to 0).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run Phase 7 first "
            "(python -m llm.data.generate_dataset ...)."
        )
    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError(f"{path}:{i} — expected a 'messages' list of >=2 turns.")
        if msgs[-1].get("role") != "assistant":
            raise ValueError(f"{path}:{i} — last message must be the assistant turn.")
        rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"{path} is empty.")
    return rows


# ---------------------------------------------------------------------------
# Pure masking helpers (unit-testable without a tokenizer)
# ---------------------------------------------------------------------------
def mask_prompt_prefix(full_ids: Sequence[int], prompt_len: int) -> List[int]:
    """Labels = full_ids with the first `prompt_len` positions set to IGNORE_INDEX."""
    if prompt_len < 0:
        raise ValueError("prompt_len must be >= 0")
    prompt_len = min(prompt_len, len(full_ids))
    return [IGNORE_INDEX] * prompt_len + list(full_ids[prompt_len:])


def fit_to_length(input_ids: Sequence[int], labels: Sequence[int],
                  max_seq_len: int) -> Dict[str, Any]:
    """Truncate an example to `max_seq_len` (right-truncation).

    Returns the fitted example plus flags:
      * `truncated`  — the tail of the answer was cut (the model may not learn
                       to emit the stop token for this example).
      * `supervised` — number of non-masked label positions. Zero means the
                       prompt alone filled the window, so the example teaches
                       nothing and should be dropped.
    """
    ids = list(input_ids[:max_seq_len])
    labs = list(labels[:max_seq_len])
    return {
        "input_ids": ids,
        "labels": labs,
        "attention_mask": [1] * len(ids),
        "truncated": len(input_ids) > max_seq_len,
        "supervised": sum(1 for t in labs if t != IGNORE_INDEX),
    }


# ---------------------------------------------------------------------------
# Tokenizer-backed encoding
# ---------------------------------------------------------------------------
def render_pair(tokenizer, messages: List[Dict[str, str]]) -> tuple[str, str]:
    """Render (prompt_text, full_text) with the model's official chat template."""
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise RuntimeError(
            "Chat template is not prefix-stable: the rendered prompt is not a "
            "prefix of the full conversation, so assistant-only loss masking "
            "cannot be applied safely. Check the tokenizer/template version."
        )
    return prompt_text, full_text


def encode_example(tokenizer, messages: List[Dict[str, str]],
                   max_seq_len: int) -> Dict[str, Any]:
    """Tokenize one conversation and mask loss to the assistant turn."""
    prompt_text, full_text = render_pair(tokenizer, messages)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = mask_prompt_prefix(full_ids, len(prompt_ids))
    return fit_to_length(full_ids, labels, max_seq_len)


def encode_dataset(tokenizer, rows: Iterable[Dict[str, Any]], max_seq_len: int,
                   verbose: bool = True) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Encode every row; drop examples with no supervised tokens.

    Returns `(examples, stats)`. `examples` hold only the tensor-ready keys.
    """
    kept: List[Dict[str, Any]] = []
    lengths: List[int] = []
    answer_lengths: List[int] = []
    n_truncated = 0
    n_dropped = 0

    for row in rows:
        enc = encode_example(tokenizer, row["messages"], max_seq_len)
        if enc["supervised"] == 0:
            n_dropped += 1
            continue
        if enc["truncated"]:
            n_truncated += 1
        lengths.append(len(enc["input_ids"]))
        answer_lengths.append(enc["supervised"])
        kept.append({
            "input_ids": enc["input_ids"],
            "labels": enc["labels"],
            "attention_mask": enc["attention_mask"],
        })

    stats = {
        "n_examples": len(kept),
        "n_dropped_no_supervision": n_dropped,
        "n_truncated": n_truncated,
        "max_seq_len": max_seq_len,
        "token_len": _describe(lengths),
        "answer_len": _describe(answer_lengths),
    }
    if verbose:
        print(f"  encoded {stats['n_examples']} examples "
              f"(dropped {n_dropped}, truncated {n_truncated})")
        print(f"  total tokens  : {_fmt(stats['token_len'])}")
        print(f"  answer tokens : {_fmt(stats['answer_len'])}")
        if n_truncated:
            print(f"  [warn] {n_truncated} example(s) hit max_seq_len={max_seq_len}; "
                  "their answer tails were cut. Consider raising --max-seq-len.")
    return kept, stats


def _describe(values: List[int]) -> Dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    s = sorted(values)
    return {
        "min": s[0],
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
        "max": s[-1],
    }


def _fmt(d: Dict[str, int]) -> str:
    return f"min={d['min']} p50={d['p50']} p95={d['p95']} max={d['max']}"


def suggest_max_seq_len(stats: Dict[str, Any], cap: int = 4096) -> int:
    """Round the observed p95 length up to a multiple of 256 (bounded by `cap`)."""
    p95 = int(stats.get("token_len", {}).get("p95", 0))
    if p95 <= 0:
        return 2048
    return min(cap, max(512, ((p95 + 255) // 256) * 256))


# ---------------------------------------------------------------------------
# Collator (pads a batch; torch imported lazily)
# ---------------------------------------------------------------------------
class PaddedCollator:
    """Right-pad `input_ids`/`labels`/`attention_mask` to the longest in-batch."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: Optional[int] = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            longest = ((longest + m - 1) // m) * m

        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = longest - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


class ListDataset:
    """Minimal map-style dataset over pre-tokenized examples."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.rows[i]
