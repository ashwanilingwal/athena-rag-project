"""Embedders behind one Protocol, so the service is testable with no model.

`HashEmbedder` is not a toy — it is what lets the full gRPC service, the SQL, the
fusion logic and the tests run in CI in under a second. Only relevance quality
needs a real model; correctness does not.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Literal, Protocol

import numpy as np

Kind = Literal["doc", "query"]


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str], kind: Kind = "doc") -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine distance reduces to a dot product.

    pgvector does NOT normalise for you. `<=>` computes cosine distance correctly
    either way, but normalising here keeps scores comparable across arms and lets
    you switch to `<#>` (inner product, faster) without changing meaning.
    """
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens vectors. Offline, no network, no model.

    Lexical overlap produces vector similarity, which is enough for tests that
    assert "the right chunk came back" without asserting semantic understanding.
    """

    def __init__(self, dimensions: int = 768) -> None:
        self.model = "hash-embed"
        self.dimensions = dimensions

    def embed(self, texts: list[str], kind: Kind = "doc") -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
                out[i, h % self.dimensions] += 1.0
        return _normalise(out)


class OllamaEmbedder:
    """Real embeddings via a locally running Ollama.

    nomic-embed-text is asymmetric: it was trained with distinct prefixes for the
    thing being stored and the thing being asked. Dropping the prefixes costs a few
    points of recall and is one of the most common silent bugs in hand-rolled RAG.
    """

    PREFIX = {"doc": "search_document: ", "query": "search_query: "}

    def __init__(self, model: str, host: str, dimensions: int) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.dimensions = dimensions

    def embed(self, texts: list[str], kind: Kind = "doc") -> np.ndarray:
        prefix = self.PREFIX[kind] if "nomic" in self.model else ""
        payload = json.dumps(
            {"model": self.model, "input": [prefix + t for t in texts]}
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        return _normalise(np.array(body["embeddings"], dtype=np.float32))


def build(settings) -> Embedder:
    if settings.provider == "hash":
        return HashEmbedder(settings.dimensions)
    return OllamaEmbedder(
        settings.embedding_model, settings.ollama_host, settings.dimensions
    )
