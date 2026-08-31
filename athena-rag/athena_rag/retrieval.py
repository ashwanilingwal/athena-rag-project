"""Fuse the dense and lexical arms into one ranking.

Why RRF fuses *ranks* and not *scores*: cosine similarity lives on [-1, 1] and
clusters tightly around 0.6-0.9 for real corpora; ts_rank is unbounded and depends
on term frequency and document length. They are not on the same scale and no fixed
normalisation makes them comparable across queries. Rank is the one signal both
arms produce that means the same thing. The price is that RRF discards magnitude —
a runaway-best dense hit is treated the same as a merely-first one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .store import Hit


@dataclass
class Retrieved:
    passages: list[Hit]
    embed_ms: int = 0
    vector_ms: int = 0
    lexical_ms: int = 0
    rerank_ms: int = 0


def rrf(runs: list[list[Hit]], k: int = 60, limit: int = 5) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum over runs of 1/(k + rank).

    k damps the head of each list. With k=60, rank 1 contributes 0.0164 and rank 2
    contributes 0.0161 — nearly equal, so a chunk must place well in BOTH arms to
    win. Lower k makes the fusion behave more like "whatever dense said".
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for run in runs:
        for rank, hit in enumerate(run, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the copy with more informative text/metadata (they are identical
            # rows, so first writer wins is fine and avoids a second fetch).
            best.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[Hit] = []
    for chunk_id, score in ordered:
        h = best[chunk_id]
        out.append(Hit(h.chunk_id, h.document_id, h.text, float(score), h.metadata))
    return out


def rerank_lexical_overlap(query: str, hits: list[Hit], limit: int) -> list[Hit]:
    """A cheap stand-in for a cross-encoder: unigram overlap with the query.

    Deliberately honest about what it is. A real cross-encoder (ms-marco-MiniLM)
    scores the query and passage *jointly* rather than comparing two independently
    produced vectors, which is why it wins. Swapping this function out for one is a
    one-line change and the natural stretch goal — see README §Stretch.
    """
    terms = {t for t in query.lower().split() if len(t) > 2}
    if not terms:
        return hits[:limit]

    def score(h: Hit) -> float:
        words = h.text.lower().split()
        if not words:
            return 0.0
        overlap = sum(1 for w in words if w.strip(".,:;()[]") in terms)
        return overlap / (len(words) ** 0.5)

    ranked = sorted(hits, key=score, reverse=True)[:limit]
    for i, h in enumerate(ranked):
        h.score = float(score(h))
    return ranked


class Retriever:
    def __init__(self, store, embedder, settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def retrieve(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int,
        strategy: str,
        document_ids: list[str] | None = None,
    ) -> Retrieved:
        s = self.settings
        depth = max(s.candidates, top_k)
        r = Retrieved(passages=[])

        need_dense = strategy in ("DENSE", "HYBRID_RRF", "HYBRID_RRF_RERANK")
        need_lex = strategy in ("LEXICAL", "HYBRID_RRF", "HYBRID_RRF_RERANK")

        dense_hits: list[Hit] = []
        lex_hits: list[Hit] = []

        if need_dense:
            t0 = time.perf_counter()
            qv = self.embedder.embed([query], kind="query")[0]
            r.embed_ms = int((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            dense_hits = self.store.dense(tenant_id, qv, depth, document_ids)
            r.vector_ms = int((time.perf_counter() - t0) * 1000)

        if need_lex:
            t0 = time.perf_counter()
            lex_hits = self.store.lexical(tenant_id, query, depth, document_ids)
            r.lexical_ms = int((time.perf_counter() - t0) * 1000)

        if strategy == "DENSE":
            r.passages = dense_hits[:top_k]
        elif strategy == "LEXICAL":
            r.passages = lex_hits[:top_k]
        else:
            fused = rrf([dense_hits, lex_hits], k=self.settings.rrf_k, limit=depth)
            if strategy == "HYBRID_RRF_RERANK":
                t0 = time.perf_counter()
                r.passages = rerank_lexical_overlap(query, fused, top_k)
                r.rerank_ms = int((time.perf_counter() - t0) * 1000)
            else:
                r.passages = fused[:top_k]

        return r
