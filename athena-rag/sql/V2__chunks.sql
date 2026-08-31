-- Owned by Java's Flyway in the real repo (V2, after V1 creates tenants/documents/jobs).
-- Kept here too so athena-rag can bootstrap a standalone dev database.
--
-- One table, two retrieval paths: an HNSW index for dense search and a GIN index
-- over a generated tsvector for lexical search. That is why there is no separate
-- BM25 process to run or keep in sync.

create extension if not exists vector;

create table if not exists chunks (
  id          uuid primary key,
  tenant_id   uuid not null,
  document_id uuid not null,
  ordinal     int  not null,
  text        text not null,
  metadata    jsonb not null default '{}',
  embedding   vector(768),
  tsv tsvector generated always as (to_tsvector('english', text)) stored,
  created_at  timestamptz not null default now(),
  unique (tenant_id, document_id, ordinal)
);

-- vector_cosine_ops because we normalise embeddings and compare with <=>.
create index if not exists chunks_embedding_hnsw
  on chunks using hnsw (embedding vector_cosine_ops);

create index if not exists chunks_tsv_gin on chunks using gin (tsv);

-- Every query filters on tenant_id first. This index is the security boundary's
-- performance backstop, not its correctness guarantee — that lives in the SQL.
create index if not exists chunks_tenant_doc on chunks (tenant_id, document_id);
