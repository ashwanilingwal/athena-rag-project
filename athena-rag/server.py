"""gRPC adapter. Translation only — no business logic lives in this file.

If you find yourself writing an `if` here that isn't about protobuf defaults or
error mapping, it belongs in athena_rag/ instead.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent import futures

import grpc

sys.path.insert(0, os.path.dirname(__file__))

from athena_rag.config import settings  # noqa: E402
from athena_rag.pipeline import Athena  # noqa: E402
from athena_rag.retrieval import Retrieved  # noqa: E402
from gen import rag_pb2, rag_pb2_grpc  # noqa: E402

log = logging.getLogger("athena-rag")

STRATEGY = {
    rag_pb2.STRATEGY_UNSPECIFIED: "HYBRID_RRF",
    rag_pb2.DENSE: "DENSE",
    rag_pb2.LEXICAL: "LEXICAL",
    rag_pb2.HYBRID_RRF: "HYBRID_RRF",
    rag_pb2.HYBRID_RRF_RERANK: "HYBRID_RRF_RERANK",
}


def _jlog(trace_id: str, event: str, **fields) -> None:
    """Structured JSON on one line, always carrying the trace id Java sent.

    This is the whole cross-language tracing story: same trace_id in both logs,
    so `grep <id>` across two services reconstructs the request.
    """
    print(json.dumps({"svc": "athena-rag", "trace_id": trace_id, "event": event, **fields}), flush=True)


def _to_retrieve_response(r: Retrieved, took_ms: int) -> rag_pb2.RetrieveResponse:
    return rag_pb2.RetrieveResponse(
        passages=[
            rag_pb2.Passage(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                text=h.text,
                score=h.score,
                rank=i + 1,
                metadata=h.metadata,
            )
            for i, h in enumerate(r.passages)
        ],
        took_ms=took_ms,
        timings=rag_pb2.Timings(
            embed_ms=r.embed_ms,
            vector_ms=r.vector_ms,
            lexical_ms=r.lexical_ms,
            rerank_ms=r.rerank_ms,
        ),
    )


class RagServicer(rag_pb2_grpc.RagServiceServicer):
    def __init__(self, athena: Athena) -> None:
        self.athena = athena

    def IndexDocument(self, request, context):
        try:
            res = self.athena.index_document(
                request.tenant_id,
                request.document_id,
                request.uri,
                content_type=request.content_type or "text/markdown",
                chunk_tokens=request.options.chunk_tokens,
                chunk_overlap=request.options.chunk_overlap,
            )
        except FileNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"uri not readable: {request.uri}")
        except Exception as exc:  # noqa: BLE001
            _jlog(request.trace_id, "index_failed", error=str(exc))
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        _jlog(
            request.trace_id,
            "indexed",
            document_id=request.document_id,
            chunks=res.chunks_indexed,
            took_ms=res.took_ms,
            replaced=res.replaced_existing,
        )
        return rag_pb2.IndexDocumentResponse(
            chunks_indexed=res.chunks_indexed,
            took_ms=res.took_ms,
            replaced_existing=res.replaced_existing,
        )

    def Retrieve(self, request, context):
        if not request.query.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")
        t0 = time.perf_counter()
        r = self.athena.retrieve(
            request.tenant_id,
            request.query,
            top_k=request.top_k,
            strategy=STRATEGY[request.strategy],
            document_ids=list(request.document_ids) or None,
        )
        took = int((time.perf_counter() - t0) * 1000)
        _jlog(request.trace_id, "retrieved", n=len(r.passages), took_ms=took,
              strategy=STRATEGY[request.strategy])
        return _to_retrieve_response(r, took)

    def Answer(self, request, context):
        if not request.query.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")
        t0 = time.perf_counter()
        history = [(t.role, t.content) for t in request.history]

        for kind, payload in self.athena.answer_stream(
            request.tenant_id,
            request.query,
            top_k=request.top_k,
            strategy=STRATEGY[request.strategy],
            history=history,
        ):
            # Client hung up: stop generating. Java cancels this when the browser
            # disconnects from the SSE stream, which is what makes cancellation
            # propagate all the way to Ollama.
            if not context.is_active():
                _jlog(request.trace_id, "answer_cancelled")
                return

            if kind == "sources":
                took = int((time.perf_counter() - t0) * 1000)
                yield rag_pb2.AnswerChunk(retrieval=_to_retrieve_response(payload, took))
            elif kind == "token":
                yield rag_pb2.AnswerChunk(token=payload)
            else:
                _jlog(request.trace_id, "answered", grounded=payload["grounded"],
                      generate_ms=payload["generate_ms"])
                yield rag_pb2.AnswerChunk(
                    complete=rag_pb2.AnswerComplete(
                        prompt_chars=payload["prompt_chars"],
                        completion_chars=payload["completion_chars"],
                        grounded=payload["grounded"],
                        timings=rag_pb2.Timings(generate_ms=payload["generate_ms"]),
                    )
                )

    def DeleteDocument(self, request, context):
        n = self.athena.delete_document(request.tenant_id, request.document_id)
        _jlog("", "deleted", document_id=request.document_id, chunks=n)
        return rag_pb2.DeleteDocumentResponse(chunks_deleted=n)

    def Health(self, request, context):
        h = self.athena.health()
        return rag_pb2.HealthResponse(
            ready=h["ready"],
            embedding_model=h["embedding_model"],
            dimensions=h["dimensions"],
            indexed_chunks=h["indexed_chunks"],
        )


def serve(port: int | None = None, athena: Athena | None = None) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=settings.max_workers))
    rag_pb2_grpc.add_RagServiceServicer_to_server(RagServicer(athena or Athena()), server)
    bind = f"[::]:{port or settings.port}"
    server.add_insecure_port(bind)
    server.start()
    _jlog("", "listening", addr=bind, provider=settings.provider)
    return server


if __name__ == "__main__":
    serve().wait_for_termination()
