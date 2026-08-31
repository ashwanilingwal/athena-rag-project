"""The library surface. Knows nothing about gRPC.

This separation is the point of "library-first, thin adapter": every behaviour here
is testable with a plain function call, and the transport can be swapped for FastAPI
or a CLI without touching a line of it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from . import chunking, embeddings
from .config import Settings, settings as default_settings
from .generate import ExtractiveGenerator, OllamaGenerator, build_prompt, looks_grounded
from .retrieval import Retrieved, Retriever
from .store import Hit, PgVectorStore


@dataclass
class IndexResult:
    chunks_indexed: int
    took_ms: int
    replaced_existing: bool


class Athena:
    def __init__(self, settings: Settings | None = None, store=None, embedder=None, generator=None):
        self.settings = settings or default_settings
        self.store = store or PgVectorStore(self.settings.dsn)
        self.embedder = embedder or embeddings.build(self.settings)
        self.generator = generator or (
            ExtractiveGenerator()
            if self.settings.provider == "hash"
            else OllamaGenerator(self.settings.chat_model, self.settings.ollama_host)
        )
        self.retriever = Retriever(self.store, self.embedder, self.settings)

    # --------------------------------------------------------------- indexing

    def index_document(
        self,
        tenant_id: str,
        document_id: str,
        uri: str,
        *,
        content_type: str = "text/markdown",
        chunk_tokens: int = 0,
        chunk_overlap: int = 0,
    ) -> IndexResult:
        t0 = time.perf_counter()
        text = chunking.load(uri, content_type)
        chunks = chunking.chunk(
            text,
            chunk_tokens=chunk_tokens or self.settings.chunk_tokens,
            chunk_overlap=chunk_overlap or self.settings.chunk_overlap,
            source=uri,
        )
        if not chunks:
            return IndexResult(0, int((time.perf_counter() - t0) * 1000), False)

        vectors = self.embedder.embed([c.text for c in chunks], kind="doc")
        n, replaced = self.store.replace_document(tenant_id, document_id, chunks, vectors)
        return IndexResult(n, int((time.perf_counter() - t0) * 1000), replaced)

    def delete_document(self, tenant_id: str, document_id: str) -> int:
        return self.store.delete_document(tenant_id, document_id)

    # -------------------------------------------------------------- retrieval

    def retrieve(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int = 0,
        strategy: str = "HYBRID_RRF",
        document_ids: list[str] | None = None,
    ) -> Retrieved:
        return self.retriever.retrieve(
            tenant_id,
            query,
            top_k=top_k or self.settings.top_k,
            strategy=strategy,
            document_ids=document_ids,
        )

    # -------------------------------------------------------------- answering

    def answer_stream(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int = 0,
        strategy: str = "HYBRID_RRF",
        history: list[tuple[str, str]] | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Yields ("sources", Retrieved) then ("token", str)* then ("done", dict).

        The frame order is the contract, not an implementation detail: the client
        renders citations while generation is still running.
        """
        retrieved = self.retrieve(tenant_id, query, top_k=top_k, strategy=strategy)
        yield "sources", retrieved

        if not retrieved.passages:
            yield "token", "I don't know based on the indexed documents."
            yield "done", {"grounded": False, "generate_ms": 0, "prompt_chars": 0, "completion_chars": 0}
            return

        messages = build_prompt(query, retrieved.passages, history)
        prompt_chars = sum(len(m["content"]) for m in messages)

        t0 = time.perf_counter()
        buf: list[str] = []
        for token in self.generator.stream(messages):
            buf.append(token)
            yield "token", token
        generate_ms = int((time.perf_counter() - t0) * 1000)

        full = "".join(buf)
        yield "done", {
            "grounded": looks_grounded(full),
            "generate_ms": generate_ms,
            "prompt_chars": prompt_chars,
            "completion_chars": len(full),
        }

    # ------------------------------------------------------------------ admin

    def health(self) -> dict:
        return {
            "ready": self.store.ping(),
            "embedding_model": self.embedder.model,
            "dimensions": self.embedder.dimensions,
            "indexed_chunks": self.store.count(),
        }
