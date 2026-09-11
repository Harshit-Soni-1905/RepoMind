"""Data models for vector store retrieval results."""

from dataclasses import dataclass
from typing import Optional

from repomind.chunking.models import CodeChunk


@dataclass(frozen=True)
class RetrievalResult:
    """Represents a code chunk retrieved from the vector store.

    Attributes:
        chunk: The retrieved CodeChunk
        score: Similarity score (higher = more similar, typically cosine similarity)
        distance: Distance metric (lower = more similar, depends on metric used)
        rank: Position in the ranked result list (1-indexed)
    """
    chunk: CodeChunk
    score: float
    distance: float
    rank: int

    def __post_init__(self):
        """Validate rank is positive."""
        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")
