"""Offline test suite. EMBED_PROVIDER=hash, no model, no network.

Run: EMBED_PROVIDER=hash pytest -q
Requires a Postgres with pgvector at ATHENA_DSN (docker compose up -d postgres).
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import grpc
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EMBED_PROVIDER", "hash")

from athena_rag import chunking  # noqa: E402
from athena_rag.config import Settings  # noqa: E402
from athena_rag.pipeline import Athena  # noqa: E402
from athena_rag.retrieval import rrf  # noqa: E402
from athena_rag.store import Hit, PgVectorStore  # noqa: E402
from gen import rag_pb2, rag_pb2_grpc  # noqa: E402
from server import serve  # noqa: E402

CORPUS = ROOT.parent / "corpus"
TENANT_A = str(uuid.uuid4())
TENANT_B = str(uuid.uuid4())


@pytest.fixture(scope="session")
def athena():
    settings = Settings(provider="hash")
    a = Athena(settings=settings)
    yield a
    a.store.close()


@pytest.fixture(scope="session")
def indexed(athena):
    """Index the corpus for tenant A and one file for tenant B."""
    ids = {}
    for path in sorted(CORPUS.glob("*.md")):
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"A/{path.name}"))
        athena.index_document(TENANT_A, doc_id, str(path))
        ids[path.stem] = doc_id
    b_doc = str(uuid.uuid5(uuid.NAMESPACE_URL, "B/retrieval.md"))
    athena.index_document(TENANT_B, b_doc, str(CORPUS / "retrieval.md"))
    yield ids
    for doc_id in ids.values():
        athena.delete_document(TENANT_A, doc_id)
    athena.delete_document(TENANT_B, b_doc)


# ------------------------------------------------------------------- chunking


def test_chunking_carries_heading_path():
    text = "# Guide\n\n## Retry policy\n\nRetry three times with backoff.\n"
    chunks = chunking.chunk(text, chunk_tokens=128, chunk_overlap=16)
    assert chunks
    assert "Guide > Retry policy" in chunks[0].text
    assert chunks[0].metadata["heading"] == "Guide > Retry policy"


def test_chunking_dedupes_repeated_heading_levels():
    text = "# MRTR\n\n# MRTR\n\nbody text here\n"
    chunks = chunking.chunk(text, chunk_tokens=128, chunk_overlap=0)
    assert "MRTR > MRTR" not in chunks[0].text


def test_chunking_respects_size_budget():
    body = "\n\n".join(f"Paragraph number {i} with some filler words." * 3 for i in range(40))
    chunks = chunking.chunk(body, chunk_tokens=64, chunk_overlap=8)
    assert len(chunks) > 1
    assert all(len(c.text) <= 64 * chunking.CHARS_PER_TOKEN * 1.6 for c in chunks)


# ---------------------------------------------------------------------- fusion


def _hits(*ids):
    return [Hit(i, "doc", f"text {i}", 1.0, {}) for i in ids]


def test_rrf_prefers_chunks_found_by_both_arms():
    """Appearing in both arms beats ranking first in only one.

    Note what this test does NOT claim: that being 2nd in both beats being 1st in
    one and 3rd in the other. With k=60 those are 0.03226 vs 0.03227 — effectively
    a tie. Large k flattens rank differences on purpose, so the surviving signal is
    agreement between arms, not position within them.
    """
    dense = _hits("dense_only", "both", "x")
    lexical = _hits("lex_only", "both", "y")
    fused = rrf([dense, lexical], k=60, limit=3)
    assert fused[0].chunk_id == "both"


def test_rrf_k_trades_agreement_against_head_position():
    """k is the knob between "trust agreement" and "trust the top hit".

    "deep_both" sits at rank 10 in both arms; "top_dense" is rank 1 in one arm only.
    At k=60 agreement wins (2/70 > 1/61). At k=1 the top hit wins (1/2 > 2/11).
    Knowing which behaviour you want is a tuning decision, not a default.
    """
    dense = _hits("top_dense", *[f"f{i}" for i in range(8)], "deep_both")
    lexical = _hits("top_lex", *[f"g{i}" for i in range(8)], "deep_both")

    assert rrf([dense, lexical], k=60, limit=1)[0].chunk_id == "deep_both"
    assert rrf([dense, lexical], k=1, limit=1)[0].chunk_id == "top_dense"


def test_rrf_is_stable_with_one_empty_arm():
    fused = rrf([_hits("a", "b"), []], k=60, limit=5)
    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_rrf_deduplicates_across_arms():
    fused = rrf([_hits("a", "b"), _hits("a", "b")], k=60, limit=10)
    assert len(fused) == 2


# ----------------------------------------------------------------- retrieval


def test_dense_retrieval_finds_relevant_chunk(athena, indexed):
    r = athena.retrieve(TENANT_A, "chunk overlap percentage", top_k=3, strategy="DENSE")
    assert r.passages
    assert any("overlap" in p.text.lower() for p in r.passages)


def test_lexical_retrieval_finds_exact_term(athena, indexed):
    r = athena.retrieve(TENANT_A, "reciprocal rank fusion", top_k=3, strategy="LEXICAL")
    assert r.passages
    assert any("fusion" in p.text.lower() for p in r.passages)


def test_hybrid_returns_ranked_passages(athena, indexed):
    r = athena.retrieve(TENANT_A, "how do I evaluate retrieval quality", top_k=5)
    assert 0 < len(r.passages) <= 5
    scores = [p.score for p in r.passages]
    assert scores == sorted(scores, reverse=True)


def test_rerank_strategy_runs_and_narrows(athena, indexed):
    r = athena.retrieve(TENANT_A, "cross encoder reranker", top_k=2,
                        strategy="HYBRID_RRF_RERANK")
    assert len(r.passages) <= 2
    assert r.rerank_ms >= 0


def test_document_scope_filter(athena, indexed):
    only = [indexed["chunking"]]
    r = athena.retrieve(TENANT_A, "retrieval", top_k=5, document_ids=only)
    assert r.passages
    assert {p.document_id for p in r.passages} == set(only)


# ------------------------------------------------------------- isolation etc.


def test_tenant_isolation(athena, indexed):
    """The security boundary. Tenant B indexed only retrieval.md."""
    r = athena.retrieve(TENANT_B, "chunk overlap percentage", top_k=10)
    assert all(p.document_id != indexed["chunking"] for p in r.passages)


def test_reindex_is_idempotent(athena, indexed):
    doc_id = indexed["chunking"]
    before = athena.store.count(TENANT_A)
    res = athena.index_document(TENANT_A, doc_id, str(CORPUS / "chunking.md"))
    after = athena.store.count(TENANT_A)
    assert res.replaced_existing is True
    assert before == after  # delete-then-insert, not append


def test_delete_removes_only_that_document(athena):
    doc_id = str(uuid.uuid4())
    athena.index_document(TENANT_A, doc_id, str(CORPUS / "evaluation.md"))
    total = athena.store.count(TENANT_A)
    removed = athena.delete_document(TENANT_A, doc_id)
    assert removed > 0
    assert athena.store.count(TENANT_A) == total - removed


def test_missing_uri_raises(athena):
    with pytest.raises(FileNotFoundError):
        athena.index_document(TENANT_A, str(uuid.uuid4()), "/nope/missing.md")


# ------------------------------------------------------------------- answering


def test_answer_stream_frame_order(athena, indexed):
    kinds = [k for k, _ in athena.answer_stream(TENANT_A, "what is chunk overlap", top_k=3)]
    assert kinds[0] == "sources"
    assert kinds[-1] == "done"
    assert "token" in kinds


def test_answer_without_matches_is_not_grounded(athena, indexed):
    empty_tenant = str(uuid.uuid4())
    frames = list(athena.answer_stream(empty_tenant, "anything at all"))
    done = frames[-1][1]
    assert done["grounded"] is False


# ------------------------------------------------------------------ transport


@pytest.fixture(scope="session")
def grpc_stub(athena):
    server = serve(port=50077, athena=athena)
    channel = grpc.insecure_channel("localhost:50077")
    yield rag_pb2_grpc.RagServiceStub(channel)
    channel.close()
    server.stop(0)


def test_grpc_health(grpc_stub):
    resp = grpc_stub.Health(rag_pb2.HealthRequest())
    assert resp.ready is True
    assert resp.dimensions == 768


def test_grpc_retrieve_returns_ranked_passages(grpc_stub, indexed):
    resp = grpc_stub.Retrieve(
        rag_pb2.RetrieveRequest(tenant_id=TENANT_A, query="reciprocal rank fusion",
                                top_k=3, strategy=rag_pb2.HYBRID_RRF, trace_id="t-1")
    )
    assert len(resp.passages) == 3
    assert [p.rank for p in resp.passages] == [1, 2, 3]
    assert resp.took_ms >= 0


def test_grpc_empty_query_is_invalid_argument(grpc_stub):
    with pytest.raises(grpc.RpcError) as exc:
        grpc_stub.Retrieve(rag_pb2.RetrieveRequest(tenant_id=TENANT_A, query="  "))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_index_missing_file_is_not_found(grpc_stub):
    with pytest.raises(grpc.RpcError) as exc:
        grpc_stub.IndexDocument(
            rag_pb2.IndexDocumentRequest(tenant_id=TENANT_A, document_id=str(uuid.uuid4()),
                                         uri="/nope/missing.md")
        )
    assert exc.value.code() == grpc.StatusCode.NOT_FOUND


def test_grpc_answer_streams_sources_before_tokens(grpc_stub, indexed):
    payloads = []
    for chunk in grpc_stub.Answer(
        rag_pb2.AnswerRequest(tenant_id=TENANT_A, query="what is chunk overlap",
                              top_k=3, strategy=rag_pb2.HYBRID_RRF, trace_id="t-2")
    ):
        payloads.append(chunk.WhichOneof("payload"))
    assert payloads[0] == "retrieval"
    assert payloads[-1] == "complete"
    assert payloads.count("token") > 0
