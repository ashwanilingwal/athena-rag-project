"""athena-rag: the retrieval engine behind the Athena document Q&A platform.

Read order: config -> chunking -> embeddings -> store -> retrieval -> generate -> pipeline
The gRPC adapter lives in ../server.py and contains no business logic.
"""
from .pipeline import Athena, IndexResult

__all__ = ["Athena", "IndexResult"]
