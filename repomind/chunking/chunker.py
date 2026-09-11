"""Code-aware chunker for Python source files.

This module combines SourceFile (raw code text) and ParsedFile (AST metadata)
to chunk code along syntactic boundaries (functions, classes, methods).
"""

from pathlib import Path
from typing import List, Optional

from repomind.config import config
from repomind.ingestion.file_reader import SourceFile
from repomind.parsing.models import ParsedFile, FunctionInfo, ClassInfo
from repomind.chunking.models import CodeChunk


class CodeChunker:
    """Chunks Python source files based on AST syntactic boundaries.

    Chunking Strategy:
    1. Module Context Chunk: Included if module docstring or imports exist.
    2. Top-level Functions: Each function becomes 1 chunk.
    3. Small Classes (<= MAX_CHUNK_LINES): Entire class becomes 1 chunk.
    4. Large Classes (> MAX_CHUNK_LINES): Split into a class header chunk and
       individual method chunks.
    5. Fallback for files with syntax errors or no extracted symbols:
       Single file chunk.
    """

    def __init__(self, max_chunk_lines: Optional[int] = None):
        """Initialize chunker.

        Args:
            max_chunk_lines: Threshold for splitting large classes.
                           Defaults to config.MAX_CHUNK_LINES.
        """
        self.max_chunk_lines = max_chunk_lines or config.MAX_CHUNK_LINES

    def chunk(self, source_file: SourceFile, parsed_file: ParsedFile) -> List[CodeChunk]:
        """Produce code chunks from a SourceFile and its ParsedFile AST metadata.

        Args:
            source_file: Raw source file loaded into memory
            parsed_file: Parsed file containing AST metadata

        Returns:
            List of CodeChunk objects extracted from the file
        """
        # Split source file content into lines for line-range extraction
        lines = source_file.content.splitlines(keepends=True)
        chunks: List[CodeChunk] = []
        rel_path = source_file.relative_path
        # Normalize path string for deterministic IDs across OS platforms
        path_str = rel_path.as_posix()

        # If file has syntax error, fallback to single file-level chunk
        if parsed_file.has_syntax_error:
            chunks.append(
                CodeChunk(
                    chunk_id=f"{path_str}::file",
                    filepath=rel_path,
                    chunk_type="file",
                    symbol_name=None,
                    source_code=source_file.content,
                    start_line=1,
                    end_line=source_file.line_count or 1,
                    docstring=None,
                    parent_class=None,
                    decorators=[],
                )
            )
            return chunks

        # 1. Module Context Chunk (docstring + imports)
        module_chunk = self._extract_module_context(source_file, parsed_file, lines)
        if module_chunk:
            chunks.append(module_chunk)

        element_chunks: List[CodeChunk] = []

        # 2. Top-level Functions
        for func_info in parsed_file.functions:
            func_chunk = self._create_function_chunk(func_info, rel_path, lines)
            element_chunks.append(func_chunk)

        # 3. Top-level Classes (entire class or split)
        for class_info in parsed_file.classes:
            class_line_count = class_info.end_line - class_info.start_line + 1

            if class_line_count <= self.max_chunk_lines:
                # Small class: single chunk containing full class definition
                class_chunk = self._create_class_chunk(class_info, rel_path, lines)
                element_chunks.append(class_chunk)
            else:
                # Large class: split into header + method chunks
                split_chunks = self._split_large_class(class_info, rel_path, lines)
                element_chunks.extend(split_chunks)

        # Sort element chunks by start_line to match source code order
        element_chunks.sort(key=lambda c: (c.start_line, c.end_line))
        chunks.extend(element_chunks)

        # If file is valid but yielded no chunks (e.g. only comments or empty),
        # return a single file-level chunk so the file content isn't lost
        if not chunks:
            chunks.append(
                CodeChunk(
                    chunk_id=f"{path_str}::file",
                    filepath=rel_path,
                    chunk_type="file",
                    symbol_name=None,
                    source_code=source_file.content,
                    start_line=1,
                    end_line=source_file.line_count or 1,
                    docstring=parsed_file.module_docstring,
                    parent_class=None,
                    decorators=[],
                )
            )

        return chunks

    def _extract_module_context(
        self, source_file: SourceFile, parsed_file: ParsedFile, lines: List[str]
    ) -> Optional[CodeChunk]:
        """Extract module-level docstring and imports into a module context chunk."""
        has_docstring = parsed_file.module_docstring is not None
        has_imports = len(parsed_file.imports) > 0

        if not (has_docstring or has_imports):
            return None

        rel_path = source_file.relative_path
        path_str = rel_path.as_posix()
        context_lines: List[str] = []
        end_line = 1

        # Collect import line ranges and docstring lines
        if parsed_file.imports:
            max_import_line = max(imp.line for imp in parsed_file.imports)
            end_line = max(end_line, max_import_line)

        # Extract source text up to the last import (or docstring)
        end_line = min(end_line, len(lines))
        if end_line > 0:
            extracted_code = "".join(lines[:end_line])
        else:
            extracted_code = ""

        if not extracted_code.strip():
            return None

        return CodeChunk(
            chunk_id=f"{path_str}::module_context",
            filepath=rel_path,
            chunk_type="module_context",
            symbol_name=None,
            source_code=extracted_code,
            start_line=1,
            end_line=end_line,
            docstring=parsed_file.module_docstring,
            parent_class=None,
            decorators=[],
        )

    def _create_function_chunk(
        self, func_info: FunctionInfo, filepath: Path, lines: List[str]
    ) -> CodeChunk:
        """Create a CodeChunk from a FunctionInfo."""
        source_code = self._extract_lines(lines, func_info.start_line, func_info.end_line)
        chunk_type = "method" if func_info.is_method else "function"
        path_str = filepath.as_posix()

        # Include line range in ID to handle overloaded methods (@overload decorators)
        # which can have multiple definitions with the same name in the same class
        if func_info.is_method and func_info.parent_class:
            chunk_id = f"{path_str}::{func_info.parent_class}.{func_info.name}@{func_info.start_line}"
        else:
            chunk_id = f"{path_str}::{func_info.name}@{func_info.start_line}"

        return CodeChunk(
            chunk_id=chunk_id,
            filepath=filepath,
            chunk_type=chunk_type,
            symbol_name=func_info.name,
            source_code=source_code,
            start_line=func_info.start_line,
            end_line=func_info.end_line,
            docstring=func_info.docstring,
            parent_class=func_info.parent_class,
            decorators=func_info.decorators,
        )

    def _create_class_chunk(
        self, class_info: ClassInfo, filepath: Path, lines: List[str]
    ) -> CodeChunk:
        """Create a single CodeChunk for an entire class definition."""
        source_code = self._extract_lines(lines, class_info.start_line, class_info.end_line)
        path_str = filepath.as_posix()

        return CodeChunk(
            chunk_id=f"{path_str}::{class_info.name}",
            filepath=filepath,
            chunk_type="class",
            symbol_name=class_info.name,
            source_code=source_code,
            start_line=class_info.start_line,
            end_line=class_info.end_line,
            docstring=class_info.docstring,
            parent_class=None,
            decorators=class_info.decorators,
        )

    def _split_large_class(
        self, class_info: ClassInfo, filepath: Path, lines: List[str]
    ) -> List[CodeChunk]:
        """Split a large class into a header chunk and individual method chunks.

        Header includes line 1 of class down to start of first method (or docstring/class vars).
        """
        chunks: List[CodeChunk] = []
        path_str = filepath.as_posix()

        # Determine header end line: up to line before first method, or class end if no methods
        if class_info.methods:
            first_method_start = min(m.start_line for m in class_info.methods)
            header_end_line = max(class_info.start_line, first_method_start - 1)
        else:
            header_end_line = class_info.end_line

        header_code = self._extract_lines(lines, class_info.start_line, header_end_line)

        header_chunk = CodeChunk(
            chunk_id=f"{path_str}::{class_info.name}__header",
            filepath=filepath,
            chunk_type="class_header",
            symbol_name=class_info.name,
            source_code=header_code,
            start_line=class_info.start_line,
            end_line=header_end_line,
            docstring=class_info.docstring,
            parent_class=None,
            decorators=class_info.decorators,
        )
        chunks.append(header_chunk)

        # Create method chunks
        for method_info in class_info.methods:
            method_chunk = self._create_function_chunk(method_info, filepath, lines)
            chunks.append(method_chunk)

        return chunks

    def _extract_lines(self, lines: List[str], start_line: int, end_line: int) -> str:
        """Extract source code text for 1-indexed line range [start_line, end_line]."""
        # Convert 1-indexed line numbers to 0-indexed slice indices
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        return "".join(lines[start_idx:end_idx])


def chunk_file(source_file: SourceFile, parsed_file: ParsedFile) -> List[CodeChunk]:
    """Convenience function to chunk a file given its SourceFile and ParsedFile.

    Args:
        source_file: SourceFile instance
        parsed_file: ParsedFile instance

    Returns:
        List of CodeChunk objects
    """
    chunker = CodeChunker()
    return chunker.chunk(source_file, parsed_file)
