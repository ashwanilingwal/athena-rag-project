"""Postgres + pgvector persistence. The only module that writes SQL.

Tenant isolation is enforced in every single query here. It is not a filter applied
by a caller who might forget — there is no code path in this file that can read a
chunk without a tenant_id predicate. That is the answer to "what stops tenant A
retrieving tenant B's chunks".
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import numpy as np
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .chunking import Chunk


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, str]


def _vec(v: np.ndarray) -> str:
    """pgvector's text input format. Cast with ::vector at the call site."""
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def _chunk_uuid(tenant_id: str, document_id: str, ordinal: int) -> str:
    """Deterministic chunk ids: re-indexing the same document yields the same ids.

    Makes the upsert genuinely idempotent rather than merely re-running.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}/{document_id}/{ordinal}"))


class PgVectorStore:
    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8) -> None:
        self.pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            open=True,
            kwargs={"row_factory": dict_row},
        )

    def close(self) -> None:
        self.pool.close()

    # ------------------------------------------------------------------ write

    def replace_document(
        self,
        tenant_id: str,
        document_id: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> tuple[int, bool]:
        """Delete-then-insert inside one transaction.

        Idempotency guarantee: if the process dies mid-transaction the whole thing
        rolls back and the previous version survives. There is no window in which
        the document is half-indexed and answerable. The cost is that an update is
        never partial — a 400-page PDF re-index holds one transaction open.
        """
        with self.pool.connection() as conn, conn.transaction():
            deleted = conn.execute(
                "delete from chunks where tenant_id = %s and document_id = %s",
                (tenant_id, document_id),
            ).rowcount
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into chunks
                      (id, tenant_id, document_id, ordinal, text, metadata, embedding)
                    values (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    [
                        (
                            _chunk_uuid(tenant_id, document_id, c.ordinal),
                            tenant_id,
                            document_id,
                            c.ordinal,
                            c.text,
                            json.dumps(c.metadata),
                            _vec(vectors[i]),
                        )
                        for i, c in enumerate(chunks)
                    ],
                )
        return len(chunks), deleted > 0

    def delete_document(self, tenant_id: str, document_id: str) -> int:
        with self.pool.connection() as conn:
            return conn.execute(
                "delete from chunks where tenant_id = %s and document_id = %s",
                (tenant_id, document_id),
            ).rowcount

    # ------------------------------------------------------------------- read

    def dense(
        self,
        tenant_id: str,
        query_vec: np.ndarray,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[Hit]:
        """Cosine KNN via HNSW.

        `<=>` is cosine DISTANCE (0 = identical), so score = 1 - distance to keep
        "higher is better" consistent across both retrieval arms.
        """
        scope, params = self._scope(tenant_id, document_ids)
        sql = f"""
            select id, document_id, text, metadata,
                   1 - (embedding <=> %s::vector) as score
            from chunks
            where {scope} and embedding is not null
            order by embedding <=> %s::vector
            limit %s
        """
        v = _vec(query_vec)
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (v, *params, v, limit)).fetchall()
        return [self._hit(r) for r in rows]

    def lexical(
        self,
        tenant_id: str,
        query: str,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[Hit]:
        """Postgres full-text search over the generated tsvector column.

        ts_rank is not BM25 — it lacks BM25's document-length saturation — but it is
        in the same family, costs zero extra infrastructure, and stays consistent
        with the dense arm because both read the same row. websearch_to_tsquery
        accepts quotes and OR the way a user actually types.
        """
        scope, params = self._scope(tenant_id, document_ids)
        sql = f"""
            select id, document_id, text, metadata,
                   ts_rank(tsv, websearch_to_tsquery('english', %s)) as score
            from chunks
            where {scope} and tsv @@ websearch_to_tsquery('english', %s)
            order by score desc
            limit %s
        """
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (query, *params, query, limit)).fetchall()
        return [self._hit(r) for r in rows]

    def count(self, tenant_id: str | None = None) -> int:
        with self.pool.connection() as conn:
            if tenant_id:
                r = conn.execute(
                    "select count(*) as n from chunks where tenant_id = %s", (tenant_id,)
                ).fetchone()
            else:
                r = conn.execute("select count(*) as n from chunks").fetchone()
        return int(r["n"])

    def ping(self) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("select 1 as ok").fetchone()["ok"] == 1

    # ----------------------------------------------------------------- helper

    @staticmethod
    def _scope(tenant_id: str, document_ids: list[str] | None):
        if document_ids:
            return "tenant_id = %s and document_id = any(%s)", (tenant_id, document_ids)
        return "tenant_id = %s", (tenant_id,)

    @staticmethod
    def _hit(row: dict) -> Hit:
        return Hit(
            chunk_id=str(row["id"]),
            document_id=str(row["document_id"]),
            text=row["text"],
            score=float(row["score"]),
            metadata={k: str(v) for k, v in (row["metadata"] or {}).items()},
        )
