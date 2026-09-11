"""Vector store subsystem: embedding generation and semantic retrieval.

This subsystem provides:
- Embedder: Wrapper around sentence-transformers for generating code embeddings
- VectorStore: ChromaDB wrapper for storing and querying code chunks by semantic similarity
- RetrievalResult: Data model for retrieval results with similarity scores
"""

from repomind.vectorstore.models import RetrievalResult
from repomind.vectorstore.embedder import Embedder
from repomind.vectorstore.store import VectorStore

__all__ = [
    "RetrievalResult",
    "Embedder",
    "VectorStore",
]
