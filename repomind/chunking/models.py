"""Data models for code chunks.

This module defines the CodeChunk data structure representing a semantically
meaningful segment of code extracted along syntactic boundaries.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass(frozen=True)
class CodeChunk:
    """Represents a semantically meaningful chunk of code.

    Chunks are extracted along syntactic boundaries (functions, classes, methods)
    rather than arbitrary token windows, preserving semantic coherence.

    Attributes:
        chunk_id: Unique deterministic identifier (e.g., "path/to/file.py::function_name")
        filepath: Path to the source file (relative to repo root)
        chunk_type: Type of code element ("function", "method", "class", "class_header", "module_context")
        symbol_name: Name of the function/class/method, None for module_context
        source_code: The actual code text for this chunk
        start_line: Starting line number (1-indexed, inclusive)
        end_line: Ending line number (1-indexed, inclusive)
        docstring: Extracted docstring if present, None otherwise
        parent_class: Name of containing class for methods, None for functions/classes
        decorators: List of decorator names (e.g., ['@staticmethod', '@property'])
    """
    chunk_id: str
    filepath: Path
    chunk_type: str
    symbol_name: Optional[str]
    source_code: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    decorators: List[str] = None

    def __post_init__(self):
        """Provide default for mutable decorators field."""
        if self.decorators is None:
            object.__setattr__(self, 'decorators', [])
