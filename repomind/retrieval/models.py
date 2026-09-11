"""Data models for unified hybrid retrieval results."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Set


class RetrievalSource(Enum):
    """Origin of a retrieved result item in the hybrid retrieval pipeline."""
    VECTOR = "vector"  # Retrieved exclusively via vector similarity search
    GRAPH = "graph"    # Discovered exclusively via graph expansion
    BOTH = "both"      # Found by vector search AND graph expansion


@dataclass(frozen=True)
class HybridResultNode:
    """Unified result node combining semantic and graph retrieval context.

    Preserves full provenance, line numbers, scores, and relational history.
    """
    node_id: str
    filepath: Path
    symbol_name: Optional[str]
    symbol_type: str  # e.g., 'file', 'class', 'function', 'method'
    source_code: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    
    # Retrieval provenance & metrics
    source: RetrievalSource = RetrievalSource.VECTOR
    semantic_score: Optional[float] = None  # Similarity score [ -1.0 to 1.0 ] from vector search
    semantic_distance: Optional[float] = None  # Cosine distance from vector search
    graph_distance: Optional[int] = None  # Shortest BFS distance from another vector seed (0 = no external corroboration)
    graph_support: float = 0.0  # Distance-decayed structural corroboration from other seeds [0.0, 1.0)
    related_via: List[str] = field(default_factory=list)  # Relationship paths/edge types (e.g. ['IMPORTS', 'DEFINES'])
    combined_score: float = 0.0  # Unified fusion score used for final ranking


@dataclass(frozen=True)
class HybridQueryResult:
    """Container for complete hybrid retrieval operation output."""
    query: str
    nodes: List[HybridResultNode]
    total_results: int
    vector_count: int
    graph_count: int
    both_count: int
    top_k: int
    traversal_depth: int
