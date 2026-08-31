# Hybrid Retrieval

## Dense retrieval
Dense retrieval embeds the query and every chunk into the same vector space and
compares them with cosine similarity. It captures paraphrase: a query about
"how do I cancel a running job" can match a passage that only says "aborting an
in-flight task". Its weakness is exact tokens. Product codes, error identifiers
and rare proper nouns are frequently missed because the embedding smooths them away.

## Lexical retrieval
Lexical retrieval matches literal terms, weighting rare terms more heavily than
common ones. In Postgres this is a tsvector column with a GIN index and ts_rank
for ordering. It nails exact identifiers and fails completely on paraphrase.

## Reciprocal rank fusion
Reciprocal rank fusion combines two ranked lists by summing one divided by a
damping constant plus the rank of each document in each list. The damping constant
is normally sixty. Fusion happens on ranks rather than raw scores because cosine
similarity and ts_rank live on incomparable scales.

## Reranking
A cross encoder reranker reads the query and the passage together in one forward
pass and outputs a single relevance score. It is far more accurate than comparing
two independently produced embeddings, and far too slow to run over an entire
corpus, so it is applied only to the top twenty or so fused candidates.
