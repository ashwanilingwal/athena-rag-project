# Athena — Polyglot Document Q&A Platform

A Spring Boot API that ingests documents asynchronously and answers questions over
them with citations, backed by a Python retrieval engine over a gRPC contract.

**Day 1 is complete and running.** Day 2 (the Java service) and Day 3 (measurement,
CI, packaging) are scoped below.

```
proto/rag.proto        the contract — written before any code on either side
athena-rag/            Python 3.12 retrieval engine  ✅ built, 23 tests green
  athena_rag/          the library: chunking → embeddings → store → retrieval → generate
  server.py            gRPC adapter (translation only, no business logic)
  sql/V2__chunks.sql   the one table Python writes to
  cli.py               index / search / ask, for poking without a Java client
athena-api/            Java 21 + Spring Boot                ⬜ Day 2
corpus/                3 sample documents to index
```

---

## Run it

```bash
docker compose up -d postgres
cd athena-rag && pip install -r requirements.txt && make proto

make test                                   # 23 tests, ~0.3s, no model needed
T=$(uuidgen)
EMBED_PROVIDER=hash python cli.py index  $T ../corpus
EMBED_PROVIDER=hash python cli.py search $T "how does fusion combine two ranked lists"
```

For real embeddings, `docker compose up -d ollama && docker compose exec ollama
ollama pull nomic-embed-text llama3.2`, then drop `EMBED_PROVIDER=hash`.

Verified end-to-end output from the gRPC server:

```
index retrieval.md    chunks=4  took=8ms
Retrieve HYBRID_RRF "how does reciprocal rank fusion combine lists"
  [1] 0.0328  'Hybrid Retrieval > Reciprocal rank fusion'
  [2] 0.0161  'Evaluating retrieval > Hit rate and MRR'
  timings: embed=0ms vector=1ms lexical=0ms total=2ms
Answer  frame sequence: retrieval -> 5 tokens -> complete   grounded=True
Health  ready=True model=hash-embed dim=768 chunks=10
```

---

## The seam

**Java owns state and correctness. Python owns inference and relevance.**

Python never writes to `documents` or `jobs` and never decides whether a request is
authorised. Java never computes an embedding and never decides which chunk is
relevant. Every field in `rag.proto` exists because one side needs something the
other is authoritative about.

Three design decisions worth being able to defend:

**Shared database, split ownership.** Python writes `chunks`; Java reads document
metadata. The alternative — Python returns chunks over gRPC and Java persists them —
keeps write ownership clean but doubles the data over the wire and puts a 768-float
vector through protobuf on every chunk. Shared DB is the right call at this scale;
say so *and* name the trade-off.

**Sources before tokens.** `AnswerChunk` is a `oneof` and the retrieval frame is
always sent first, so the client renders citations while generation is still
running. That is a product decision with a latency justification, not an accident of
implementation order.

**Delete-then-insert in one transaction.** Re-indexing a document is idempotent
because chunk ids are `uuid5(tenant/document/ordinal)` and the delete and insert
share a transaction. If the process dies mid-index, the previous version survives
intact — there is no window where a document is half-answerable. The cost: a
400-page re-index holds one transaction open, which is a real objection and the
reason a chunked-commit strategy would be the next iteration.

---

## What's actually implemented

| Piece | Where | Note |
| --- | --- | --- |
| Heading-aware chunking | `chunking.py` | carries the heading path into chunk text; dedupes `Title > Title` |
| Asymmetric embeddings | `embeddings.py` | `search_document:` / `search_query:` prefixes for nomic — dropping these silently costs recall |
| Offline embedder | `embeddings.py` | `HashEmbedder` — full service testable in CI with no model |
| Dense search | `store.py` | HNSW, `1 - (embedding <=> q)` so higher is better in both arms |
| Lexical search | `store.py` | generated `tsvector` + GIN + `websearch_to_tsquery` |
| RRF fusion | `retrieval.py` | fuses ranks, not scores — the two arms aren't on comparable scales |
| Tenant isolation | `store.py` | no code path in that file can read a chunk without a `tenant_id` predicate |
| Streaming answers | `pipeline.py`, `server.py` | sources → tokens → complete, with `context.is_active()` cancellation |
| Structured logs | `server.py` | one JSON line per event, always carrying Java's `trace_id` |

---

## Day 1 checkpoint

Answer these before moving to Day 2. Two of them have answers embedded in the code
comments; the other two you should work out yourself.

1. pgvector does **not** normalise vectors for you. `embeddings._normalise` does it
   before insert. Why does that matter for `<=>`, and what would break if you
   normalised documents but forgot to normalise queries?
2. `test_rrf_k_trades_agreement_against_head_position` shows k=60 and k=1 producing
   different winners on the same input. Which would you want for a corpus of API
   error codes, and why?
3. HNSW is approximate. At what corpus size does the recall loss start to matter,
   and which two build parameters trade recall against build time?
4. Your idempotency guarantee is a transaction. Name the failure mode it does *not*
   cover. (Hint: Java's job row and Python's transaction are not the same
   transaction.)

---

## Day 2 — the Java service

Spring Boot init with web, validation, data-jpa, flyway, redis, actuator, and a gRPC
client. Then, in order:

| Slot | Task |
| --- | --- |
| 0:45 | Flyway `V1__init.sql` (tenants, documents, jobs, query_logs). `V2` is already written. |
| 0:45 | `ApiKeyFilter` with constant-time hash compare + a per-request `TenantContext`. |
| 1:15 | `POST /v1/documents`: checksum → dedupe → store blob → create job → `202`. `Idempotency-Key` upserts against the unique constraint. |
| 1:00 | Ingestion worker: `select … for update skip locked limit 10`, calls `IndexDocument`, exponential backoff, max 3 attempts. |
| 1:00 | Resilience4j around the gRPC client: 30s timeout on index, 10s on retrieve, circuit breaker on retrieve, bulkhead capping concurrency. |
| 1:00 | `/v1/search` and `/v1/answer` + Redis cache keyed `sha256(tenant, normalised_query, strategy, top_k, corpus_version)`. Bump `corpus_version` on every successful ingest — that is the whole invalidation strategy. |
| 1:00 | `/v1/answer/stream`: `SseEmitter` over the gRPC stream, `AnswerChunk` frames → named SSE events, cancel the gRPC call on client disconnect. |
| 0:30 | `@SpringBootTest` with Testcontainers + an in-process gRPC test server. |

**End state:** `curl -N -H 'X-API-Key: …' localhost:8080/v1/answer/stream?q=…` streams
sources then tokens.

**The failure mode to watch for:** spending Day 2 tuning retrieval instead of writing
Java, because retrieval is the familiar part. Retrieval is already good enough. The
Java layer is what the job posting is asking about.

---

## Day 3 — prove it, then package it

Micrometer timers per stage with the trace id propagated across the boundary; port
the eval harness to hit the *Java* endpoint and record hit@5 / MRR / faithfulness for
all four strategies; `k6` at 50 concurrent users, cold vs warm cache, p50/p95/p99;
`docker compose up` from a fresh clone; GitHub Actions building both services; README
with the numbers and a demo GIF.

Those measured numbers are what goes on the resume. Leave the bullets blank until
Day 3 fills them in.

---

## On "the training part"

Classical RAG has no training step. Indexing is a one-shot transform — no loss
function, nothing converges. Writing "trained a RAG model" on a resume will cost you
credibility with anyone who knows the field.

What this project has is an offline pipeline and a tuning loop. If you want a real
training component, `retrieval.rerank_lexical_overlap` is deliberately a placeholder:
mine hard negatives from the top-20 dense results that aren't the gold chunk, fine-tune
a MiniLM cross-encoder on your golden set for a few epochs on CPU, and report the
hit@5 delta. That is an honest "trained X, improved Y by Z%" line. Do it after the
three days ship, not during.
