"""LLM Component — RAFT Dataset Generation (LLM_PLAN.md Phase 7).

Builds the SFT dataset for fine-tuning Qwen3:

  training profile
    -> retrieve golden chunks + add 1-3 distractors (RAFT)
    -> teacher (self-distill bulk, Gemini subset) -> gold answer (validated safe)
    -> hard gate (scan_unsafe == 0, all 5 sections, mentions risk)
    -> Unicode-normalize
    -> training example: [COMPACT system, user(+chunks), assistant=gold]

Outputs train.jsonl / val.jsonl + data_card.md (counts, teacher mix, pass rate,
drop reasons). The gold is generated under the FULL prompt; the example pairs the
COMPACT prompt with it (prompt-compression objective).

Offline (no GPU/key):
    python -m ckd_risk_app.llm.data.generate_dataset --n 12 --self-model mock --teacher-share 0 --out-dir phase7_data

Colab (real):
    from ckd_risk_app.llm.data.generate_dataset import generate
    generate(n=600, teacher_share=0.2, self_model="qwen", gemini_key="...", out_dir="/content/phase7_data")
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..rag.retriever import Retriever
from ..serving.contract import LLMInput, parse_guidance
from ..serving.prompts import build_compact_system_prompt, build_user_prompt
from ..serving.safety_validator import context_from_llm_input, scan_unsafe
from .teacher_backends import get_teacher
from .train_profiles import generate_train_profiles

# Unicode -> ASCII normalization (avoid tokenizer quirks in training targets).
_UNI = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "—": "-", "–": "-", "…": "...", " ": " ", "•": "-",
}


def normalize_unicode(s: str) -> str:
    for k, v in _UNI.items():
        s = s.replace(k, v)
    return s


def _format_raw_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
    """Format a raw KB chunk record like the retriever's output (for distractors)."""
    return {
        "chunk_id": c.get("chunk_id"), "text": c.get("text"),
        "source": c.get("source_id"), "source_id": c.get("source_id"),
        "title": c.get("title"), "organization": c.get("organization"),
        "topic": c.get("topic"), "restricted": bool(c.get("restricted")),
        "safety_label": c.get("safety_label"), "score": 0.0, "rank": -1,
    }


def _sample_distractors(all_chunks: List[Dict[str, Any]], golden_ids: set,
                        golden_topics: set, rng: random.Random, n: int) -> List[Dict[str, Any]]:
    off_topic = [c for c in all_chunks
                 if c["chunk_id"] not in golden_ids and c.get("topic") not in golden_topics]
    pool = off_topic or [c for c in all_chunks if c["chunk_id"] not in golden_ids]
    n = min(n, len(pool))
    return [_format_raw_chunk(c) for c in rng.sample(pool, n)] if n else []


def _gate(gold: str, ctx: Dict[str, Any], risk: str) -> Optional[str]:
    """Return a drop-reason if the gold answer fails the gate, else None."""
    if scan_unsafe(gold, ctx):
        return "unsafe"
    parsed = parse_guidance(gold)
    if not parsed.is_structurally_valid():
        return "incomplete_sections"
    if risk.lower() not in gold.lower():
        return "missing_risk_mention"
    return None


def generate(n: int = 600, teacher_share: float = 0.2, k_golden: int = 4,
             distractors: int = 2, self_model: str = "auto",
             gemini_key: Optional[str] = None, gemini_model: str = "gemini-2.5-flash",
             out_dir: str | Path = "phase7_data", seed: int = 42,
             val_frac: float = 0.1, profile_seed: int = 7) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    profiles = generate_train_profiles(n, seed=profile_seed)
    retriever = Retriever.load()
    all_chunks = retriever.store.chunks

    self_teacher = get_teacher("self", model=self_model)
    gem_teacher = None
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
    if teacher_share > 0 and gemini_key:
        gem_teacher = get_teacher("gemini", api_key=gemini_key, model_name=gemini_model)
    elif teacher_share > 0:
        print("[warn] teacher_share>0 but no Gemini key; using 100% self-distillation.")

    print(f"Profiles: {len(profiles)} | self={self_teacher.spec} | "
          f"gemini={'on' if gem_teacher else 'off'} | k_golden={k_golden} | distractors={distractors}\n")

    examples: List[Dict[str, Any]] = []
    drops: Counter = Counter()
    teacher_used: Counter = Counter()

    for i, profile in enumerate(profiles, 1):
        inp = LLMInput.from_predict_response(profile)
        ctx = context_from_llm_input(inp)

        golden = retriever.retrieve_for_llm_input(inp, k=k_golden)
        golden_ids = {c["chunk_id"] for c in golden}
        golden_topics = {c.get("topic") for c in golden}
        distract = _sample_distractors(all_chunks, golden_ids, golden_topics, rng, distractors)
        chunks = golden + distract
        rng.shuffle(chunks)

        use_gemini = bool(gem_teacher) and (rng.random() < teacher_share)
        teacher = gem_teacher if use_gemini else self_teacher
        try:
            gold, _raw, status = teacher.generate_gold(inp, chunks)
        except Exception as exc:   # never let one bad call kill the run
            drops["teacher_error"] += 1
            if i % 25 == 0:
                print(f"[{i}] teacher error: {exc}")
            continue

        reason = _gate(gold, ctx, profile["risk_class"])
        if reason:
            drops[reason] += 1
            continue

        gold = normalize_unicode(gold)
        examples.append({
            "profile_id": profile["profile_id"],
            "risk_class": profile["risk_class"],
            "teacher": teacher.spec,
            "messages": [
                {"role": "system", "content": build_compact_system_prompt()},
                {"role": "user", "content": normalize_unicode(build_user_prompt(inp, chunks))},
                {"role": "assistant", "content": gold},
            ],
            "meta": {
                "golden_ids": sorted(golden_ids),
                "distractor_ids": [c["chunk_id"] for c in distract],
                "teacher_status": status,
            },
        })
        teacher_used[teacher.spec.split(":")[0]] += 1
        if i % 25 == 0:
            print(f"[{i}/{len(profiles)}] kept={len(examples)} drops={sum(drops.values())}")

    # Split (stratified-ish by shuffle) and write.
    rng.shuffle(examples)
    n_val = int(len(examples) * val_frac)
    val, train = examples[:n_val], examples[n_val:]
    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val, out_dir / "val.jsonl")
    stats = _write_data_card(out_dir / "data_card.md", profiles, examples, train, val,
                             drops, teacher_used, k_golden, distractors, gemini_model)

    print(f"\nKept {len(examples)}/{len(profiles)} "
          f"(pass rate {stats['pass_rate']}). train={len(train)} val={len(val)}")
    print("Teacher mix:", dict(teacher_used), "| Drops:", dict(drops))
    print(f"Outputs -> {out_dir}/train.jsonl, val.jsonl, data_card.md")
    return {"train": train, "val": val, "drops": dict(drops), "stats": stats}


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_data_card(path: Path, profiles, examples, train, val, drops, teacher_used,
                     k_golden, distractors, gemini_model) -> Dict[str, Any]:
    n_prof = len(profiles)
    kept = len(examples)
    pass_rate = round(kept / n_prof, 3) if n_prof else 0.0
    by_class = Counter(e["risk_class"] for e in examples)
    total_teacher = sum(teacher_used.values()) or 1
    gem_share = round(teacher_used.get("gemini", 0) / total_teacher, 3)

    lines = [
        "# Phase 7 — RAFT Training Data — Data Card", "",
        f"- Date: {datetime.now(timezone.utc).date().isoformat()}",
        f"- Profiles generated: {n_prof} | examples kept: {kept} (pass rate {pass_rate})",
        f"- Split: train={len(train)}, val={len(val)}",
        f"- Class mix (kept): low={by_class['low']}, moderate={by_class['moderate']}, high={by_class['high']}",
        f"- Teacher mix: {dict(teacher_used)} (Gemini share {gem_share}; model {gemini_model})",
        f"- Retrieval: k_golden={k_golden}, distractors_per_example={distractors}",
        f"- Drops: {dict(drops)}",
        "",
        "## Method",
        "- Targets generated under the FULL prompt; examples pair the COMPACT prompt "
        "with the gold (prompt-compression objective).",
        "- Bulk targets via self-distillation (base Qwen3 + RAG + validator); a subset "
        "via Gemini (free tier) for harder/diverse/emergency cases.",
        "- **Every gold answer passed `scan_unsafe`** (0 unsafe), all 5 sections present, "
        "mentions the risk level. Unicode normalized to ASCII.",
        "- Teacher used only on synthetic, de-identified profiles; no real patient data; "
        "no external API at inference.",
        "- Profiles drawn from the training distribution; the 30 held-out eval profiles "
        "and the ML test set were NOT used.",
    ]
    if examples:
        ex = examples[0]
        lines += ["", "## Example (first kept)", "",
                  f"- profile {ex['profile_id']} ({ex['risk_class']}), teacher {ex['teacher']}",
                  "", "**Assistant target:**", "```", ex["messages"][2]["content"][:900], "```"]
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"pass_rate": pass_rate, "kept": kept, "gem_share": gem_share}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate RAFT SFT dataset (Phase 7)")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--teacher-share", type=float, default=0.2)
    p.add_argument("--k-golden", type=int, default=4)
    p.add_argument("--distractors", type=int, default=2)
    p.add_argument("--self-model", default="auto", help="auto | qwen | mock")
    p.add_argument("--gemini-key", default=None, help="or set GEMINI_API_KEY")
    p.add_argument("--gemini-model", default="gemini-2.5-flash")
    p.add_argument("--out-dir", default="phase7_data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    generate(n=args.n, teacher_share=args.teacher_share, k_golden=args.k_golden,
             distractors=args.distractors, self_model=args.self_model,
             gemini_key=args.gemini_key, gemini_model=args.gemini_model,
             out_dir=args.out_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
