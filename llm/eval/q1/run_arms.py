"""Q1 six-arm ablation runner (Parts 2 and 3).

Arms
----
    A  base        compact prompt, no RAG,   no validator      (generates)
    B  base_rag    full prompt,    RAG k=4,  no validator      (generates)
    C  base_rag_val  reuses Arm B raw output byte-for-byte + validator
    D  ft_rag      full prompt,    RAG k=4,  no validator, LoRA (generates)
    E  ft_rag_val    reuses Arm D raw output byte-for-byte + validator
    F  ft_compact  compact prompt, RAG k=4,  validator,    LoRA (generates)

Pre-specified comparisons: B-A (full prompt + RAG; both change, so this is not a
pure RAG effect), C-B (validator on base), D-B (LoRA),
E-D (validator after fine-tuning), F-E (compact-prompt trade-off).

Arms C and E never call a model. They load the stored raw text from B and D and
assert a SHA256 match, so the "validator contribution" is measured on identical
bytes by construction.

Every record is appended to a JSONL immediately, so a run resumes without
regenerating completed outputs. Raw generations are never overwritten.

Usage (one arm at a time keeps a single model resident):
    python -m llm.eval.q1.run_arms --arm A --benchmark q1_benchmark/q1_profiles_300.jsonl
    python -m llm.eval.q1.run_arms --arm B --benchmark ...
    python -m llm.eval.q1.run_arms --arm C --benchmark ...        # no GPU needed
    python -m llm.eval.q1.run_arms --arm D --adapter <path> --benchmark ...
    python -m llm.eval.q1.run_arms --arm E --benchmark ...        # no GPU needed
    python -m llm.eval.q1.run_arms --arm F --adapter <path> --benchmark ...

Secondary robustness (generative arms only):
    python -m llm.eval.q1.run_arms --arm F --adapter <path> --sample --seeds 42,52,62
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ...rag.retriever import Retriever
    from ...serving.contract import LLMInput
    from ...serving.prompts import build_messages, build_compact_messages
    from ...serving.safety_validator import context_from_llm_input, validate
except ImportError:  # pragma: no cover
    from llm.rag.retriever import Retriever
    from llm.serving.contract import LLMInput
    from llm.serving.prompts import build_messages, build_compact_messages
    from llm.serving.safety_validator import context_from_llm_input, validate

# arm -> (prompt, use_rag, use_validator, use_adapter, reuse_raw_from)
ARM_SPEC: Dict[str, Dict[str, Any]] = {
    "A": {"name": "base_compact",   "prompt": "compact", "rag": False, "validator": False, "adapter": False, "reuse": None},
    "B": {"name": "base_rag",       "prompt": "full",    "rag": True,  "validator": False, "adapter": False, "reuse": None},
    "C": {"name": "base_rag_val",   "prompt": "full",    "rag": True,  "validator": True,  "adapter": False, "reuse": "B"},
    "D": {"name": "ft_rag",         "prompt": "full",    "rag": True,  "validator": False, "adapter": True,  "reuse": None},
    "E": {"name": "ft_rag_val",     "prompt": "full",    "rag": True,  "validator": True,  "adapter": True,  "reuse": "D"},
    "F": {"name": "ft_compact_val", "prompt": "compact", "rag": True,  "validator": True,  "adapter": True,  "reuse": None},
}
ARM_ORDER = ["A", "B", "C", "D", "E", "F"]

RETRIEVAL_K = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Stable hash over a directory's file names + contents."""
    p = Path(path)
    if not p.exists():
        return ""
    if p.is_file():
        return sha256_file(p)
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(str(f.relative_to(p)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"No profiles in {path}")
    return rows


def strip_evaluator(profile: Dict[str, Any]) -> Dict[str, Any]:
    """The LLM must never see evaluator-only fields."""
    clean = {k: v for k, v in profile.items() if k != "evaluator"}
    return clean


def assert_prompt_clean(messages: List[Dict[str, str]], profile: Dict[str, Any]) -> None:
    """Hard gate: no internal benchmark bookkeeping string may appear in the
    rendered prompt.

    Deliberately narrow. `evidence_topics` (e.g. "diabetes", "bp") and
    `expected_referral_level` (e.g. "routine", "urgent") are common clinical/
    English words that are SUPPOSED to correlate with real prompt content (the
    patient's own diabetes status, natural follow-up language) - checking them
    produced false positives on every profile. Only synthetic, non-natural-
    language identifiers are checked: the snake_case required/prohibited
    behaviour codes, the attack-category code, and the template-family code.
    None of those strings can legitimately occur in generated patient-facing
    prose, so any match is a real leak.
    """
    blob = " ".join(m["content"] for m in messages).lower()
    ev = profile.get("evaluator") or {}
    needles: List[str] = []
    for key in ("required_behaviours", "prohibited_behaviours"):
        needles.extend(str(x).lower() for x in (ev.get(key) or []))
    for key in ("attack_category", "template_family"):
        if ev.get(key):
            needles.append(str(ev[key]).lower())
    leaked = [n for n in needles if n and n in blob]
    if leaked:
        raise AssertionError(
            f"Evaluator metadata leaked into the prompt for "
            f"{profile.get('profile_id')}: {leaked[:3]}"
        )


def reported_symptom(profile: Dict[str, Any]) -> Optional[str]:
    """The free-text acute symptom for an `emergency`-group profile, if any.

    `context.red_flags` also carries structured BP-threshold flags (e.g.
    "systolic_bp_>=180") for the subset of emergency profiles whose target
    category is "high" - those already appear in `patient_inputs` and need no
    separate signal. Only the free-text `symptom:...` entry represents
    information NOT otherwise visible to the model or validator.
    """
    for flag in (profile.get("context") or {}).get("red_flags") or []:
        if isinstance(flag, str) and flag.startswith("symptom:"):
            return flag.split("symptom:", 1)[1].strip()
    return None


def build_arm_messages(profile: Dict[str, Any], chunks: List[Dict[str, Any]],
                       prompt_kind: str) -> List[Dict[str, str]]:
    inp = LLMInput.from_predict_response(strip_evaluator(profile))
    builder = build_compact_messages if prompt_kind == "compact" else build_messages
    messages = builder(inp, chunks)
    # An adversarial profile's instruction, and a reported acute symptom not
    # captured by the screening inputs, arrive the way a real one would:
    # appended to the user turn as something the patient said.
    adv = (profile.get("context") or {}).get("adversarial_instruction")
    if adv:
        messages[-1]["content"] += f"\n\nPATIENT MESSAGE\n{adv}\n"
    symptom = reported_symptom(profile)
    if symptom:
        messages[-1]["content"] += (
            f"\n\nPATIENT MESSAGE\nI also wanted to mention: I have been "
            f"experiencing {symptom}.\n"
        )
    assert_prompt_clean(messages, profile)
    return messages


def context_with_symptoms(profile: Dict[str, Any], inp: LLMInput) -> Dict[str, Any]:
    """`context_from_llm_input` plus the profile's reported symptom (if any).

    The deployed `context_from_llm_input` hardcodes `symptoms: ""`, since the
    production system has no free-text symptom input. Q1's benchmark does carry
    a reported symptom for some `emergency`-group profiles (see
    `reported_symptom`), and the validator's own `is_emergency()` only sees it
    through `context["symptoms"]` - so it must be added here for the validator
    to have the same information the (patched) prompt now conveys.
    """
    ctx = context_from_llm_input(inp)
    symptom = reported_symptom(profile)
    if symptom:
        ctx["symptoms"] = symptom
    return ctx


def count_tokens(model: Any, messages: List[Dict[str, str]], text: str = "") -> Tuple[int, int]:
    """(prompt_tokens, completion_tokens). Falls back to a chars/4 estimate."""
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        try:
            rendered = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
            p = len(tok(rendered, add_special_tokens=False)["input_ids"])
            c = len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
            return p, c
        except Exception:
            pass
    p = sum(len(m["content"]) for m in messages) // 4
    return p, len(text) // 4


def gpu_peak_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def _record_path(out_dir: Path, arm: str) -> Path:
    return Path(out_dir) / "records" / f"arm_{arm}.jsonl"


def load_done_keys(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    done: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # tolerate a torn final line from an interrupted run
        done[(r["profile_id"], int(r["seed"]))] = r
    return done


def append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_arm(arm: str, benchmark_path: Path, out_dir: Path,
            adapter: Optional[str] = None,
            model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
            seeds: Optional[List[int]] = None, do_sample: bool = False,
            temperature: float = 0.2, top_p: float = 0.9,
            max_new_tokens: int = 700, k: int = RETRIEVAL_K,
            limit: Optional[int] = None,
            force_group: Optional[str] = None) -> Path:
    if arm not in ARM_SPEC:
        raise ValueError(f"Unknown arm {arm}; expected one of {ARM_ORDER}")
    spec = ARM_SPEC[arm]
    seeds = seeds or [42]
    out_dir = Path(out_dir)
    out_path = _record_path(out_dir, arm)

    profiles = load_benchmark(benchmark_path)
    if limit:
        profiles = profiles[:limit]
    done = load_done_keys(out_path)
    if force_group:
        # Regenerate only this group: drop its existing entries from `done` so
        # the normal not-already-done logic below regenerates them. Freshly
        # generated rows are appended after the stale ones; both `run_arm`'s
        # own resume check and `report.score_all` read via `load_done_keys`,
        # which collapses duplicate (profile_id, seed) keys to the LAST line
        # in the file - so nothing needs to be deleted from disk.
        forced_ids = {p["profile_id"] for p in profiles if p.get("group") == force_group}
        if not forced_ids:
            raise ValueError(f"No profiles found with group={force_group!r}")
        done = {k: v for k, v in done.items() if k[0] not in forced_ids}
        print(f"  --force-group {force_group}: {len(forced_ids)} profiles will be regenerated")
    print(f"Arm {arm} ({spec['name']}): {len(profiles)} profiles x {len(seeds)} seed(s); "
          f"{len(done)} already done")

    # --- validator-only arms: reuse stored raw bytes ------------------------
    if spec["reuse"]:
        src_path = _record_path(out_dir, spec["reuse"])
        if not src_path.exists():
            raise FileNotFoundError(
                f"Arm {arm} reuses Arm {spec['reuse']} output, but {src_path} "
                f"does not exist. Run arm {spec['reuse']} first."
            )
        source = load_done_keys(src_path)
        n_new = 0
        for (pid, seed), src in sorted(source.items()):
            if (pid, seed) in done:
                continue
            profile = next((p for p in profiles if p["profile_id"] == pid), None)
            if profile is None:
                continue
            raw = src["raw_output"]
            if sha256_text(raw) != src["raw_sha256"]:
                raise AssertionError(f"Source raw text corrupted for {pid}/{seed}")
            ctx = context_with_symptoms(
                profile, LLMInput.from_predict_response(strip_evaluator(profile)))
            t0 = time.perf_counter()
            result = validate(raw, ctx)
            vt = time.perf_counter() - t0
            rec = dict(src)
            rec.update({
                "arm": arm, "arm_name": spec["name"],
                "delivered_output": result.safe_text,
                "delivered_sha256": sha256_text(result.safe_text),
                "validator_status": result.status,
                "violation_categories": sorted({v.category for v in result.violations}),
                "validator_time_s": round(vt, 6),
                "fallback": int(result.status == "fallback"),
                "reused_raw_from": spec["reuse"],
                "raw_reuse_verified": True,
            })
            append_record(out_path, rec)
            n_new += 1
        print(f"  applied validator to {n_new} reused generations -> {out_path}")
        return out_path

    # --- generative arms ----------------------------------------------------
    from ...serving.model_backends import QwenChatModel  # type: ignore
    retriever = Retriever.load() if spec["rag"] else None
    print(f"  loading model{' + adapter' if (spec['adapter'] and adapter) else ''} ...")
    if spec["adapter"] and not adapter:
        raise ValueError(f"Arm {arm} requires --adapter (fine-tuned model).")
    model = QwenChatModel(
        model_name=model_name,
        adapter_path=adapter if spec["adapter"] else None,
        max_new_tokens=max_new_tokens,
        temperature=(temperature if do_sample else 0.0),
        top_p=top_p,
    )

    manifest_hashes = {
        "adapter_sha256": sha256_dir(Path(adapter)) if (spec["adapter"] and adapter) else "",
        "kb_index_sha256": sha256_dir(Path(__file__).resolve().parents[2]
                                      / "knowledge_base" / "index"),
        "validator_sha256": sha256_file(Path(__file__).resolve().parents[2]
                                        / "serving" / "safety_validator.py"),
        "prompts_sha256": sha256_file(Path(__file__).resolve().parents[2]
                                      / "serving" / "prompts.py"),
        "benchmark_sha256": sha256_file(Path(benchmark_path)),
    }

    n_new = 0
    for seed in seeds:
        for profile in profiles:
            pid = profile["profile_id"]
            if (pid, seed) in done:
                continue

            inp = LLMInput.from_predict_response(strip_evaluator(profile))
            chunks = retriever.retrieve_for_llm_input(inp, k=k) if retriever else []
            messages = build_arm_messages(profile, chunks, spec["prompt"])

            _set_decoding(model, do_sample, temperature, top_p, seed)
            reset_gpu_peak()
            t0 = time.perf_counter()
            failure = ""
            try:
                raw = model.generate(messages)
            except Exception as exc:            # never lose the run to one profile
                raw = ""
                failure = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - t0
            ptok, ctok = count_tokens(model, messages, raw)

            rec: Dict[str, Any] = {
                "profile_id": pid,
                "group": profile.get("group"),
                "risk_class": profile.get("risk_class"),
                "reliability_band": profile.get("reliability_band"),
                "arm": arm, "arm_name": spec["name"],
                "seed": int(seed),
                "do_sample": bool(do_sample),
                "prompt_kind": spec["prompt"],
                "use_rag": bool(spec["rag"]),
                "raw_output": raw,
                "raw_sha256": sha256_text(raw),
                "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
                "retrieved_scores": [round(float(c.get("score", 0.0)), 6) for c in chunks],
                "retrieved_topics": [c.get("topic") for c in chunks],
                "prompt_tokens": ptok, "completion_tokens": ctok,
                "total_tokens": ptok + ctok,
                "latency_s": round(latency, 4),
                "peak_gpu_mem_mb": round(gpu_peak_mb(), 1),
                "retry_count": 0,
                "technical_failure": failure,
                "model_name": model_name,
                "model_spec": getattr(model, "spec", model_name),
                **manifest_hashes,
            }

            if spec["validator"]:
                ctx = context_with_symptoms(profile, inp)
                t1 = time.perf_counter()
                result = validate(raw, ctx)
                rec.update({
                    "delivered_output": result.safe_text,
                    "delivered_sha256": sha256_text(result.safe_text),
                    "validator_status": result.status,
                    "violation_categories": sorted({v.category for v in result.violations}),
                    "validator_time_s": round(time.perf_counter() - t1, 6),
                    "fallback": int(result.status == "fallback"),
                })
            else:
                rec.update({
                    "delivered_output": raw,
                    "delivered_sha256": rec["raw_sha256"],
                    "validator_status": "not_applied",
                    "violation_categories": [],
                    "validator_time_s": 0.0,
                    "fallback": 0,
                })

            append_record(out_path, rec)
            n_new += 1
            if n_new % 25 == 0:
                print(f"  [{n_new}] {pid} seed={seed} ({latency:.1f}s)")

    print(f"  wrote {n_new} new records -> {out_path}")
    return out_path


def _set_decoding(model: Any, do_sample: bool, temperature: float,
                  top_p: float, seed: int) -> None:
    """Deterministic primary run; sampled secondary run. Seed torch per profile."""
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    model.temperature = temperature if do_sample else 0.0
    model.top_p = top_p


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def verify_reuse(out_dir: Path) -> List[Tuple[str, bool, str]]:
    """Arm C raw must be byte-identical to B; Arm E raw byte-identical to D."""
    checks: List[Tuple[str, bool, str]] = []
    for cand, src in (("C", "B"), ("E", "D")):
        cp, sp = _record_path(out_dir, cand), _record_path(out_dir, src)
        if not (cp.exists() and sp.exists()):
            checks.append((f"arm_{cand}_raw_identical_to_{src}", False,
                           "records missing - arm not run yet"))
            continue
        c, s = load_done_keys(cp), load_done_keys(sp)
        shared = set(c) & set(s)
        mismatched = [k for k in shared if c[k]["raw_sha256"] != s[k]["raw_sha256"]]
        checks.append((
            f"arm_{cand}_raw_identical_to_{src}",
            not mismatched and len(shared) == len(s),
            f"{len(shared)}/{len(s)} compared, {len(mismatched)} mismatched",
        ))
    return checks


def write_manifest(out_dir: Path, benchmark_path: Path) -> Path:
    out_dir = Path(out_dir)
    arms: Dict[str, Any] = {}
    for arm in ARM_ORDER:
        p = _record_path(out_dir, arm)
        if not p.exists():
            arms[arm] = {"status": "not_run"}
            continue
        recs = load_done_keys(p)
        first = next(iter(recs.values()), {})
        arms[arm] = {
            "status": "complete",
            "n_records": len(recs),
            "records_sha256": sha256_file(p),
            "model_name": first.get("model_name", ""),
            "adapter_sha256": first.get("adapter_sha256", ""),
            "kb_index_sha256": first.get("kb_index_sha256", ""),
            "validator_sha256": first.get("validator_sha256", ""),
            "prompts_sha256": first.get("prompts_sha256", ""),
            "seeds": sorted({s for _, s in recs}),
        }
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(Path(benchmark_path)) if Path(benchmark_path).exists() else "",
        "retrieval_k": RETRIEVAL_K,
        "arm_spec": ARM_SPEC,
        "arms": arms,
    }
    p = out_dir / "q1_run_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Q1 six-arm ablation runner")
    ap.add_argument("--arm", required=True, choices=ARM_ORDER + ["manifest", "verify"])
    ap.add_argument("--benchmark", default="q1_benchmark/q1_profiles_300.jsonl")
    ap.add_argument("--out-dir", default="q1_runs")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--seeds", default="42", help="comma separated, e.g. 42,52,62")
    ap.add_argument("--sample", action="store_true",
                    help="secondary robustness run (do_sample=True, temp 0.2, top_p 0.9)")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--k", type=int, default=RETRIEVAL_K)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-group", default=None,
                    help="regenerate only this benchmark group, even if this arm "
                         "already has entries for it (e.g. after a harness fix)")
    a = ap.parse_args(argv)

    if a.arm == "manifest":
        print(f"manifest -> {write_manifest(Path(a.out_dir), Path(a.benchmark))}")
        return 0
    if a.arm == "verify":
        for name, ok, detail in verify_reuse(Path(a.out_dir)):
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        return 0

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    run_arm(a.arm, Path(a.benchmark), Path(a.out_dir), adapter=a.adapter,
            model_name=a.model, seeds=seeds, do_sample=a.sample,
            max_new_tokens=a.max_new_tokens, k=a.k, limit=a.limit,
            force_group=a.force_group)
    write_manifest(Path(a.out_dir), Path(a.benchmark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
