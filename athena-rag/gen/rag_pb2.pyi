from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RetrievalStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STRATEGY_UNSPECIFIED: _ClassVar[RetrievalStrategy]
    DENSE: _ClassVar[RetrievalStrategy]
    LEXICAL: _ClassVar[RetrievalStrategy]
    HYBRID_RRF: _ClassVar[RetrievalStrategy]
    HYBRID_RRF_RERANK: _ClassVar[RetrievalStrategy]
STRATEGY_UNSPECIFIED: RetrievalStrategy
DENSE: RetrievalStrategy
LEXICAL: RetrievalStrategy
HYBRID_RRF: RetrievalStrategy
HYBRID_RRF_RERANK: RetrievalStrategy

class IndexDocumentRequest(_message.Message):
    __slots__ = ("tenant_id", "document_id", "uri", "content_type", "options", "trace_id")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    document_id: str
    uri: str
    content_type: str
    options: IndexOptions
    trace_id: str
    def __init__(self, tenant_id: _Optional[str] = ..., document_id: _Optional[str] = ..., uri: _Optional[str] = ..., content_type: _Optional[str] = ..., options: _Optional[_Union[IndexOptions, _Mapping]] = ..., trace_id: _Optional[str] = ...) -> None: ...

class IndexOptions(_message.Message):
    __slots__ = ("chunk_tokens", "chunk_overlap", "embedding_model")
    CHUNK_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_OVERLAP_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    chunk_tokens: int
    chunk_overlap: int
    embedding_model: str
    def __init__(self, chunk_tokens: _Optional[int] = ..., chunk_overlap: _Optional[int] = ..., embedding_model: _Optional[str] = ...) -> None: ...

class IndexDocumentResponse(_message.Message):
    __slots__ = ("chunks_indexed", "took_ms", "replaced_existing")
    CHUNKS_INDEXED_FIELD_NUMBER: _ClassVar[int]
    TOOK_MS_FIELD_NUMBER: _ClassVar[int]
    REPLACED_EXISTING_FIELD_NUMBER: _ClassVar[int]
    chunks_indexed: int
    took_ms: int
    replaced_existing: bool
    def __init__(self, chunks_indexed: _Optional[int] = ..., took_ms: _Optional[int] = ..., replaced_existing: _Optional[bool] = ...) -> None: ...

class RetrieveRequest(_message.Message):
    __slots__ = ("tenant_id", "query", "top_k", "strategy", "document_ids", "trace_id")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    query: str
    top_k: int
    strategy: RetrievalStrategy
    document_ids: _containers.RepeatedScalarFieldContainer[str]
    trace_id: str
    def __init__(self, tenant_id: _Optional[str] = ..., query: _Optional[str] = ..., top_k: _Optional[int] = ..., strategy: _Optional[_Union[RetrievalStrategy, str]] = ..., document_ids: _Optional[_Iterable[str]] = ..., trace_id: _Optional[str] = ...) -> None: ...

class Passage(_message.Message):
    __slots__ = ("chunk_id", "document_id", "text", "score", "rank", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, chunk_id: _Optional[str] = ..., document_id: _Optional[str] = ..., text: _Optional[str] = ..., score: _Optional[float] = ..., rank: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Timings(_message.Message):
    __slots__ = ("embed_ms", "vector_ms", "lexical_ms", "rerank_ms", "generate_ms")
    EMBED_MS_FIELD_NUMBER: _ClassVar[int]
    VECTOR_MS_FIELD_NUMBER: _ClassVar[int]
    LEXICAL_MS_FIELD_NUMBER: _ClassVar[int]
    RERANK_MS_FIELD_NUMBER: _ClassVar[int]
    GENERATE_MS_FIELD_NUMBER: _ClassVar[int]
    embed_ms: int
    vector_ms: int
    lexical_ms: int
    rerank_ms: int
    generate_ms: int
    def __init__(self, embed_ms: _Optional[int] = ..., vector_ms: _Optional[int] = ..., lexical_ms: _Optional[int] = ..., rerank_ms: _Optional[int] = ..., generate_ms: _Optional[int] = ...) -> None: ...

class RetrieveResponse(_message.Message):
    __slots__ = ("passages", "took_ms", "timings")
    PASSAGES_FIELD_NUMBER: _ClassVar[int]
    TOOK_MS_FIELD_NUMBER: _ClassVar[int]
    TIMINGS_FIELD_NUMBER: _ClassVar[int]
    passages: _containers.RepeatedCompositeFieldContainer[Passage]
    took_ms: int
    timings: Timings
    def __init__(self, passages: _Optional[_Iterable[_Union[Passage, _Mapping]]] = ..., took_ms: _Optional[int] = ..., timings: _Optional[_Union[Timings, _Mapping]] = ...) -> None: ...

class Turn(_message.Message):
    __slots__ = ("role", "content")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class AnswerRequest(_message.Message):
    __slots__ = ("tenant_id", "query", "top_k", "strategy", "history", "trace_id")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    query: str
    top_k: int
    strategy: RetrievalStrategy
    history: _containers.RepeatedCompositeFieldContainer[Turn]
    trace_id: str
    def __init__(self, tenant_id: _Optional[str] = ..., query: _Optional[str] = ..., top_k: _Optional[int] = ..., strategy: _Optional[_Union[RetrievalStrategy, str]] = ..., history: _Optional[_Iterable[_Union[Turn, _Mapping]]] = ..., trace_id: _Optional[str] = ...) -> None: ...

class AnswerComplete(_message.Message):
    __slots__ = ("prompt_chars", "completion_chars", "timings", "grounded")
    PROMPT_CHARS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_CHARS_FIELD_NUMBER: _ClassVar[int]
    TIMINGS_FIELD_NUMBER: _ClassVar[int]
    GROUNDED_FIELD_NUMBER: _ClassVar[int]
    prompt_chars: int
    completion_chars: int
    timings: Timings
    grounded: bool
    def __init__(self, prompt_chars: _Optional[int] = ..., completion_chars: _Optional[int] = ..., timings: _Optional[_Union[Timings, _Mapping]] = ..., grounded: _Optional[bool] = ...) -> None: ...

class AnswerChunk(_message.Message):
    __slots__ = ("retrieval", "token", "complete")
    RETRIEVAL_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    retrieval: RetrieveResponse
    token: str
    complete: AnswerComplete
    def __init__(self, retrieval: _Optional[_Union[RetrieveResponse, _Mapping]] = ..., token: _Optional[str] = ..., complete: _Optional[_Union[AnswerComplete, _Mapping]] = ...) -> None: ...

class DeleteDocumentRequest(_message.Message):
    __slots__ = ("tenant_id", "document_id")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    document_id: str
    def __init__(self, tenant_id: _Optional[str] = ..., document_id: _Optional[str] = ...) -> None: ...

class DeleteDocumentResponse(_message.Message):
    __slots__ = ("chunks_deleted",)
    CHUNKS_DELETED_FIELD_NUMBER: _ClassVar[int]
    chunks_deleted: int
    def __init__(self, chunks_deleted: _Optional[int] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("ready", "embedding_model", "dimensions", "indexed_chunks")
    READY_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    INDEXED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    embedding_model: str
    dimensions: int
    indexed_chunks: int
    def __init__(self, ready: _Optional[bool] = ..., embedding_model: _Optional[str] = ..., dimensions: _Optional[int] = ..., indexed_chunks: _Optional[int] = ...) -> None: ...
