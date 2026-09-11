"""Chunking subsystem: code-aware segmentation.

This subsystem converts parsed code elements into semantically meaningful
chunks with rich metadata, following syntactic boundaries rather than
arbitrary token windows.
"""

from repomind.chunking.models import CodeChunk
from repomind.chunking.chunker import CodeChunker, chunk_file

__all__ = [
    "CodeChunk",
    "CodeChunker",
    "chunk_file",
]
