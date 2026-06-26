"""LLM Component — Build the RAG Index (LLM_PLAN.md Phase 3).

Reads the Phase-2 corpus (knowledge_base/processed/chunks.jsonl), embeds every
chunk, and writes the vector index to knowledge_base/index/.

Production (Colab/GPU):
    pip install -q sentence-transformers faiss-cpu
    python -m ckd_risk_app.llm.rag.embed_index            # uses BGE-M3 + FAISS

Laptop with Ollama (no torch, real semantic quality): build with BGE-M3 served
by Ollama. Run `ollama pull bge-m3` first, then:
    python -m ckd_risk_app.llm.rag.embed_index --embedder ollama-bge-m3

Local / offline (no torch, no Ollama): falls back to the pure-Python hashing
embedder so the retriever and relevance gate still run:
    python -m ckd_risk_app.llm.rag.embed_index --embedder hashing

The embedder used is recorded in the index metadata; the retriever reuses it
exactly, so query and index always share one vector space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .embedder import get_embedder
from .vector_store import save_index, INDEX_DIR

LLM_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = LLM_DIR / "knowledge_base" / "processed" / "chunks.jsonl"


def load_chunks(path: Path = CHUNKS_PATH) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {path}. Run Phase 2 ingest first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embedding_text(chunk: Dict) -> str:
    """Text fed to the embedder. Light enrichment with the passage title/topic."""
    title = (chunk.get("passage_title") or "").strip()
    prefix = f"{title}. " if title else ""
    return f"{prefix}{chunk.get('text', '')}".strip()


def build(embedder_spec: str = "auto", chunks_path: Path = CHUNKS_PATH) -> None:
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    embedder = get_embedder(embedder_spec)
    print(f"Embedder: {embedder.spec}")

    texts = [embedding_text(c) for c in chunks]
    print(f"Fitting + embedding {len(texts)} chunks...")
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    save_index(vectors, chunks, embedder.spec, embedder.state(), INDEX_DIR)
    # Quick provenance summary.
    from collections import Counter
    by_label = Counter(c.get("safety_label") for c in chunks)
    print("Index safety mix: " + ", ".join(f"{k}={v}" for k, v in by_label.most_common()))
    print("Done.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the RAG vector index from chunks.jsonl")
    p.add_argument("--embedder", default="auto",
                   help="auto | bge-m3 | ollama-bge-m3 | hashing | hashing-<dim> "
                        "(default: auto)")
    p.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    args = p.parse_args(argv)
    build(args.embedder, args.chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
