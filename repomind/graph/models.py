"""Data models for code graph representation."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class NodeType(Enum):
    """Types of nodes in the code graph."""
    FILE = "file"           # Python module/file
    CLASS = "class"         # Class definition
    FUNCTION = "function"   # Top-level function
    METHOD = "method"       # Class method


class EdgeType(Enum):
    """Types of edges in the code graph."""
    IMPORTS = "imports"         # Module A imports module B
    DEFINES = "defines"         # File defines a class/function
    CONTAINS = "contains"       # Class contains a method
    CALLS = "calls"            # Function/method calls another (optional, conservative)


@dataclass(frozen=True)
class GraphNode:
    """Represents a node in the code graph with metadata.

    Node IDs follow the same format as CodeChunk IDs from Stage 3:
    - File: "path/to/file.py"
    - Function: "path/to/file.py::function_name"
    - Class: "path/to/file.py::ClassName"
    - Method: "path/to/file.py::ClassName.method_name"
    """
    node_id: str
    node_type: NodeType
    filepath: Path              # Original file path
    symbol_name: Optional[str]  # None for FILE nodes
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
