"""LLM Component — Knowledge-Base Safety Audit (LLM_PLAN.md Phase 2).

Independent gate that re-scans the FINAL chunks.jsonl and asserts the
three-tier safety contract holds. Run AFTER ingest, BEFORE freezing the KB or
starting RAG / RAFT data generation.

Audit gate (all must pass):
  * safe_public chunks with medication/drug content      == 0
  * safe_public chunks with dialysis/KRT content         == 0
  * safe_public chunks with transplant content           == 0
  * safe_public chunks with a numeric nutrition target   == 0
  * safe_public chunks with a direct numeric BP target   == 0
  * every restricted chunk has safety_label == restricted_clinician_referral
  * no chunk carries the exclude_from_retrieval label (those must be dropped)

This re-scans with the same risk patterns ingest uses; a failure means the
exclusion vocabulary needs expanding (then re-run ingest). Retrieval-relevance
(>=80% on 10 queries) is checked separately in Phase 3.

Run:
    python -m ckd_risk_app.llm.rag.audit
Exit code 0 = pass, 1 = fail.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .ingest import (
    DEFAULT_OUT,
    EXCLUDE_PATTERNS,
    EXCLUDED,
    RESTRICTED,
    SAFE_PUBLIC,
    has_numeric_nutrition_target,
)


def load_chunks(path: Path = DEFAULT_OUT) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {path}. Run ingest first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(chunks: List[dict]) -> Dict[str, int]:
    """Return a dict of violation_name -> count (0 everywhere = pass)."""
    safe = [c for c in chunks if c.get("safety_label") == SAFE_PUBLIC]

    violations: Dict[str, int] = {
        "safe_public_medication": sum(bool(EXCLUDE_PATTERNS["medication"].search(c["text"])) for c in safe),
        "safe_public_dialysis":   sum(bool(EXCLUDE_PATTERNS["dialysis"].search(c["text"])) for c in safe),
        "safe_public_transplant": sum(bool(EXCLUDE_PATTERNS["transplant"].search(c["text"])) for c in safe),
        "safe_public_dosing":     sum(bool(EXCLUDE_PATTERNS["dosing"].search(c["text"])) for c in safe),
        "safe_public_bp_target":  sum(bool(EXCLUDE_PATTERNS["bp_target"].search(c["text"])) for c in safe),
        "safe_public_numeric_nutrition": sum(has_numeric_nutrition_target(c["text"]) for c in safe),
        "restricted_mislabeled": sum(
            1 for c in chunks if c.get("restricted") and c.get("safety_label") != RESTRICTED),
        "excluded_label_present": sum(1 for c in chunks if c.get("safety_label") == EXCLUDED),
        "missing_safety_label": sum(1 for c in chunks if not c.get("safety_label")),
    }
    return violations


def print_report(chunks: List[dict], violations: Dict[str, int]) -> bool:
    from collections import Counter
    by_label = Counter(c.get("safety_label") for c in chunks)

    print("=" * 60)
    print(f"KB SAFETY AUDIT — {len(chunks)} chunks")
    print("=" * 60)
    print("By safety label: " + ", ".join(f"{k}={v}" for k, v in by_label.most_common()))
    print("\nGate checks (target = 0 for all):")
    ok = True
    for name, count in violations.items():
        status = "PASS" if count == 0 else "FAIL"
        if count:
            ok = False
        print(f"  [{status}] {name}: {count}")

    # Examples for any failures (first 3) to aid fixing.
    if not ok:
        print("\nExample offending safe_public chunks:")
        safe = [c for c in chunks if c.get("safety_label") == SAFE_PUBLIC]
        shown = 0
        for c in safe:
            hits = [cat for cat, pat in EXCLUDE_PATTERNS.items() if pat.search(c["text"])]
            if has_numeric_nutrition_target(c["text"]):
                hits.append("numeric_nutrition")
            if hits and shown < 3:
                snippet = c["text"][:160].encode("ascii", "replace").decode()
                print(f"  - {c['chunk_id']} {hits}: {snippet}")
                shown += 1

    print("\n" + ("AUDIT PASSED — KB safe to freeze." if ok
                  else "AUDIT FAILED — expand exclusion vocabulary in ingest.py and re-run."))
    return ok


def main() -> int:
    chunks = load_chunks()
    violations = audit(chunks)
    ok = print_report(chunks, violations)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
