"""LLM Component — Embedders (LLM_PLAN.md Phase 3).

Two interchangeable embedders behind one interface:

  * BGEM3Embedder  — production. BAAI/bge-m3 dense embeddings (1024-d) via
                     sentence-transformers. Used in Colab / GPU environments.
  * HashingEmbedder — pure-Python, dependency-free lexical fallback. Lets the
                     retrieval pipeline + relevance gate run anywhere (no torch,
                     numpy, or model download). NOT for production quality.

The embedder's `spec` string (e.g. "bge-m3" or "hashing-2048") is saved in the
index metadata so the retriever rebuilds the SAME embedder — query and index
always live in the same vector space.

Embeddings are L2-normalized, so cosine similarity == dot product.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")

# Embedders share a small interface so they are interchangeable + persistable:
#   spec               -> string that rebuilds the embedder class/params
#   fit(texts)         -> learn corpus stats (no-op for stateless embedders)
#   state() / load_state(state) -> persist/restore fitted state in the index meta
#   embed(texts) / embed_one(text)


# ---------------------------------------------------------------------------
# Pure-Python lexical fallbacks
# ---------------------------------------------------------------------------
class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (stdlib only), raw TF."""

    def __init__(self, dim: int = 2048):
        self.dim = dim

    @property
    def spec(self) -> str:
        return f"hashing-{self.dim}"

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim

    def fit(self, texts: Sequence[str]) -> "HashingEmbedder":
        return self

    def state(self) -> Optional[Dict]:
        return None

    def load_state(self, state: Optional[Dict]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN.findall(text.lower()):
                vec[self._bucket(tok)] += 1.0
            out.append(_l2(vec))
        return out

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class HashingTfidfEmbedder:
    """Hashing embedder with IDF weighting (fitted, pure-Python).

    IDF downweights common words (e.g. "kidney", "disease") and upweights
    distinctive ones (e.g. "smoking", "sodium", "tested"), which markedly
    improves lexical retrieval over raw TF. This is the default offline
    fallback when BGE-M3 is unavailable.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.idf: List[float] = [1.0] * dim

    @property
    def spec(self) -> str:
        return f"hashing-tfidf-{self.dim}"

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim

    def fit(self, texts: Sequence[str]) -> "HashingTfidfEmbedder":
        n_docs = max(len(texts), 1)
        df = [0] * self.dim
        for text in texts:
            for b in {self._bucket(t) for t in _TOKEN.findall(text.lower())}:
                df[b] += 1
        self.idf = [math.log((n_docs + 1) / (d + 1)) + 1.0 for d in df]
        return self

    def state(self) -> Dict:
        return {"dim": self.dim, "idf": self.idf}

    def load_state(self, state: Optional[Dict]) -> None:
        if state and "idf" in state:
            self.dim = state.get("dim", self.dim)
            self.idf = state["idf"]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok, tf in Counter(_TOKEN.findall(text.lower())).items():
                b = self._bucket(tok)
                vec[b] += tf * self.idf[b]
            out.append(_l2(vec))
        return out

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


def _l2(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


# ---------------------------------------------------------------------------
# Production: BGE-M3
# ---------------------------------------------------------------------------
class BGEM3Embedder:
    """BAAI/bge-m3 dense embeddings via sentence-transformers (lazy-loaded)."""

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(self.MODEL_NAME)
        self._batch_size = batch_size
        self.dim = self._model.get_sentence_embedding_dimension()

    @property
    def spec(self) -> str:
        return "bge-m3"

    def fit(self, texts: Sequence[str]) -> "BGEM3Embedder":
        return self  # pretrained; nothing to fit

    def state(self) -> Optional[Dict]:
        return None

    def load_state(self, state: Optional[Dict]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return [list(map(float, v)) for v in vecs]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_embedder(spec: str = "auto"):
    """Build an embedder from a spec string.

    spec:
      "bge-m3"             -> BGEM3Embedder (raises if sentence-transformers missing)
      "hashing-tfidf[-<d>]"-> HashingTfidfEmbedder (IDF-weighted, default fallback)
      "hashing[-<d>]"      -> HashingEmbedder (raw TF)
      "auto"               -> BGE-M3 if available, else HashingTfidfEmbedder
    """
    spec = (spec or "auto").strip().lower()
    if spec.startswith("hashing-tfidf"):
        parts = spec.split("-")
        dim = int(parts[-1]) if parts[-1].isdigit() else 4096
        return HashingTfidfEmbedder(dim=dim)
    if spec.startswith("hashing"):
        parts = spec.split("-")
        dim = int(parts[-1]) if parts[-1].isdigit() else 2048
        return HashingEmbedder(dim=dim)
    if spec == "bge-m3":
        return BGEM3Embedder()
    if spec == "auto":
        try:
            return BGEM3Embedder()
        except Exception as exc:  # sentence-transformers/torch not installed
            print(f"[embedder] BGE-M3 unavailable ({type(exc).__name__}); "
                  f"falling back to HashingTfidfEmbedder (lexical TF-IDF, test-only).")
            return HashingTfidfEmbedder()
    raise ValueError(f"Unknown embedder spec: {spec!r}")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product (== cosine for L2-normalized vectors)."""
    return sum(x * y for x, y in zip(a, b))
