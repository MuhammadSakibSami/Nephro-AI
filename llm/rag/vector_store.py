"""LLM Component — Vector Store (LLM_PLAN.md Phase 3).

Persists chunk embeddings + metadata and answers nearest-neighbor queries.

Two backends, auto-selected:
  * FAISS (IndexFlatIP) — used when faiss + numpy are available (Colab/prod).
  * Pure-Python brute force — cosine over stored vectors; runs anywhere.

On-disk layout (knowledge_base/index/):
  meta.json      - {embedder_spec, dim, count, created, chunks:[...row-aligned]}
  vectors.json   - {"vectors": [[...], ...]}  (row-aligned with chunks)
  bge_faiss.index- optional FAISS binary (written only if faiss available)

Vectors are L2-normalized, so inner product == cosine similarity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LLM_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = LLM_DIR / "knowledge_base" / "index"
META_PATH = INDEX_DIR / "meta.json"
VECTORS_PATH = INDEX_DIR / "vectors.json"
FAISS_PATH = INDEX_DIR / "bge_faiss.index"


@dataclass
class SearchHit:
    rank: int
    score: float
    chunk: Dict


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_index(vectors: List[List[float]], chunks: List[Dict], embedder_spec: str,
               embedder_state: Optional[Dict] = None,
               index_dir: Path = INDEX_DIR) -> None:
    if len(vectors) != len(chunks):
        raise ValueError("vectors and chunks must be row-aligned")
    index_dir.mkdir(parents=True, exist_ok=True)
    dim = len(vectors[0]) if vectors else 0

    meta = {
        "embedder_spec": embedder_spec,
        "embedder_state": embedder_state,
        "dim": dim,
        "count": len(vectors),
        "created": datetime.now(timezone.utc).isoformat(),
        "chunks": chunks,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    VECTORS_PATH.write_text(json.dumps({"vectors": vectors}), encoding="utf-8")

    # Optional FAISS index for fast search in prod.
    faiss_written = _try_build_faiss(vectors, dim, index_dir)
    print(f"[store] saved {len(vectors)} vectors (dim={dim}) -> {index_dir}"
          + (" + FAISS index" if faiss_written else " (no FAISS; brute-force at query time)"))


def _try_build_faiss(vectors: List[List[float]], dim: int, index_dir: Path) -> bool:
    try:
        import faiss
        import numpy as np
    except ImportError:
        if FAISS_PATH.exists():
            FAISS_PATH.unlink()  # stale index would mismatch fresh vectors
        return False
    arr = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(dim)
    index.add(arr)
    faiss.write_index(index, str(index_dir / FAISS_PATH.name))
    return True


# ---------------------------------------------------------------------------
# Load + search
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, meta: Dict, vectors: List[List[float]]):
        self.embedder_spec: str = meta["embedder_spec"]
        self.embedder_state: Optional[Dict] = meta.get("embedder_state")
        self.dim: int = meta["dim"]
        self.chunks: List[Dict] = meta["chunks"]
        self.vectors = vectors
        self._faiss = None
        self._np = None
        self._maybe_load_faiss()

    def _maybe_load_faiss(self) -> None:
        if not FAISS_PATH.exists():
            return
        try:
            import faiss
            import numpy as np
            self._faiss = faiss.read_index(str(FAISS_PATH))
            self._np = np
        except Exception:
            self._faiss = None

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "VectorStore":
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Build it first: "
                f"python -m ckd_risk_app.llm.rag.embed_index")
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        vectors = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]
        return cls(meta, vectors)

    def search(self, query_vec: Sequence[float], k: int = 4) -> List[SearchHit]:
        if self._faiss is not None and self._np is not None:
            ranked = self._search_faiss(query_vec, k)
        else:
            ranked = self._search_bruteforce(query_vec, k)
        return [SearchHit(rank=i + 1, score=score, chunk=self.chunks[idx])
                for i, (idx, score) in enumerate(ranked)]

    def _search_faiss(self, query_vec, k) -> List[Tuple[int, float]]:
        q = self._np.asarray([query_vec], dtype="float32")
        scores, idxs = self._faiss.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def _search_bruteforce(self, query_vec, k) -> List[Tuple[int, float]]:
        scored = [(i, sum(a * b for a, b in zip(query_vec, vec)))
                  for i, vec in enumerate(self.vectors)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]
