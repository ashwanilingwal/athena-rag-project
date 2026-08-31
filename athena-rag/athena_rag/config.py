"""Every tunable knob in one place, overridable by environment variable.

These knobs are the whole game in retrieval quality. Change one at a time, re-run
the eval harness, keep or revert. Never change two and guess which one helped.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # --- storage
    dsn: str = os.environ.get(
        "ATHENA_DSN", "postgresql://athena:athena@127.0.0.1:5432/athena"
    )

    # --- chunking
    chunk_tokens: int = _int("CHUNK_TOKENS", 512)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 64)

    # --- embeddings
    # "ollama" hits a real model; "hash" is a deterministic offline stand-in so the
    # whole service is testable in CI with no GPU, no model download, no network.
    provider: str = os.environ.get("EMBED_PROVIDER", "ollama")
    embedding_model: str = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    dimensions: int = _int("EMBED_DIM", 768)
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    chat_model: str = os.environ.get("CHAT_MODEL", "llama3.2")

    # --- retrieval
    top_k: int = _int("TOP_K", 5)
    candidates: int = _int("CANDIDATES", 30)  # per-arm depth before fusion
    rrf_k: int = _int("RRF_K", 60)  # RRF damping constant

    # --- grpc
    port: int = _int("GRPC_PORT", 50051)
    max_workers: int = _int("GRPC_WORKERS", 8)


settings = Settings()
