"""Jupyter Notebook (.ipynb) parser and chunker.

Parses Jupyter Notebooks using the standard library json module, extracts code
and markdown cells, sanitizes IPython magics, extracts AST metadata (functions,
classes, imports, calls) from Python code cells, and produces semantic CodeChunks
and a ParsedFile model compatible with the RepoMind graph and vector store pipeline.
"""

import ast
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from repomind.config import config
from repomind.parsing.models import (
    ParsedFile,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    CallInfo,
)
from repomind.chunking.models import CodeChunk
from repomind.parsing.ast_parser import CodeElementExtractor


def _sanitize_ipython_magics(source: str) -> str:
    """Sanitize IPython magic commands and shell escapes into Python comments.

    Replaces %, %%, !, and ? lines with # <line> to preserve exact line counts
    and line lengths so ast.parse can parse without syntax errors.

    Args:
        source: Raw code cell source text

    Returns:
        Sanitized Python code string with magics converted to comments
    """
    sanitized_lines = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        leading_ws = line[: len(line) - len(stripped)]

        # Check if line starts with magic characters: %, %%, !, ?
        if stripped.startswith(("%", "!", "?")):
            sanitized_lines.append(f"{leading_ws}# {stripped}")
        # Check if line ends with help query ? (e.g., obj? or obj??)
        elif stripped.rstrip("\r\n").endswith("?"):
            sanitized_lines.append(f"{leading_ws}# {stripped}")
        else:
            sanitized_lines.append(line)
    return "".join(sanitized_lines)


def _extract_cell_source(cell: Dict[str, Any]) -> str:
    """Extract source code string from a notebook cell dictionary.

    Handles both string source and list-of-strings source representation.

    Args:
        cell: Notebook cell dictionary

    Returns:
        Source code as a single string
    """
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    elif isinstance(source, str):
        return source
    return ""


def _is_python_kernel(metadata: Dict[str, Any]) -> bool:
    """Determine if a notebook targets a Python kernel.

    Args:
        metadata: Top-level notebook metadata dictionary

    Returns:
        True if Python kernel, False otherwise
    """
    # Check language_info
    lang_info = metadata.get("language_info", {})
    if isinstance(lang_info, dict) and "name" in lang_info:
        lang_name = str(lang_info["name"]).lower().strip()
        if lang_name:
            return lang_name.startswith("py")

    # Check kernelspec
    kernelspec = metadata.get("kernelspec", {})
    if isinstance(kernelspec, dict):
        lang = str(kernelspec.get("language", "")).lower().strip()
        if lang:
            return lang.startswith("py")
        kname = str(kernelspec.get("name", "")).lower().strip()
        if kname:
            return "python" in kname or kname.startswith("py")

    # Default to Python if unspecified
    return True


class NotebookParser:
    """Parses Jupyter Notebooks (.ipynb) into ParsedFile and CodeChunks.

    Extracts functions, classes, imports, and calls from code cells while ignoring
    outputs, execution counts, and binary attachments. Handles IPython magics,
    markdown cells, non-Python kernels, and syntax errors gracefully.
    """

    def __init__(self, max_chunk_lines: Optional[int] = None):
        """Initialize notebook parser.

        Args:
            max_chunk_lines: Threshold for splitting large classes into header + methods.
        """
        self.max_chunk_lines = max_chunk_lines or config.MAX_CHUNK_LINES

    def parse(
        self, content: str, filepath: Path
    ) -> Tuple[ParsedFile, List[CodeChunk]]:
        """Parse notebook JSON content into ParsedFile metadata and CodeChunks.

        Args:
            content: Raw JSON string of the .ipynb file
            filepath: Path to the notebook file (relative or absolute)

        Returns:
            Tuple of (ParsedFile, List[CodeChunk])
        """
        path_str = filepath.as_posix()

        # Step 1: Parse JSON
        try:
            notebook_data = json.loads(content)
            if not isinstance(notebook_data, dict):
                raise ValueError("Notebook root JSON must be an object")
        except Exception as e:
            # Fallback for malformed JSON
            line_count = max(1, len(content.splitlines()))
            fallback_chunk = CodeChunk(
                chunk_id=f"{path_str}::file",
                filepath=filepath,
                chunk_type="file",
                symbol_name=None,
                source_code=content,
                start_line=1,
                end_line=line_count,
                docstring=None,
                parent_class=None,
                decorators=[],
            )
            parsed_file = ParsedFile(
                filepath=filepath,
                functions=[],
                classes=[],
                imports=[],
                calls=[],
                module_docstring=None,
                has_syntax_error=True,
                syntax_error_message=f"Invalid notebook JSON: {str(e)}",
            )
            return parsed_file, [fallback_chunk]

        cells = notebook_data.get("cells", [])
        metadata = notebook_data.get("metadata", {})
        is_python = _is_python_kernel(metadata)

        all_chunks: List[CodeChunk] = []
        all_functions: List[FunctionInfo] = []
        all_classes: List[ClassInfo] = []
        all_imports: List[ImportInfo] = []
        all_calls: List[CallInfo] = []
        module_docstring: Optional[str] = None

        # Shared extractor across all code cells in the notebook to maintain import and symbol context
        extractor = CodeElementExtractor()

        current_line = 1

        for cell_idx, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue

            cell_type = cell.get("cell_type", "")
            raw_source = _extract_cell_source(cell)
            cell_lines = raw_source.splitlines(keepends=True)
            cell_line_count = len(cell_lines) if cell_lines else 1

            cell_start_line = current_line
            cell_end_line = current_line + max(0, cell_line_count - 1)
            current_line = cell_end_line + 1

            # Skip completely empty cells
            if not raw_source.strip():
                continue

            # Case A: Markdown Cell
            if cell_type == "markdown":
                if module_docstring is None:
                    module_docstring = raw_source.strip()

                chunk = CodeChunk(
                    chunk_id=f"{path_str}::cell_{cell_idx}::markdown",
                    filepath=filepath,
                    chunk_type="notebook_markdown",
                    symbol_name=None,
                    source_code=raw_source,
                    start_line=cell_start_line,
                    end_line=cell_end_line,
                    docstring=None,
                    parent_class=None,
                    decorators=[],
                )
                all_chunks.append(chunk)

            # Case B: Code Cell
            elif cell_type == "code":
                if not is_python:
                    # Non-Python kernel (e.g. R, Julia): emit raw notebook_code chunk
                    chunk = CodeChunk(
                        chunk_id=f"{path_str}::cell_{cell_idx}::code",
                        filepath=filepath,
                        chunk_type="notebook_code",
                        symbol_name=None,
                        source_code=raw_source,
                        start_line=cell_start_line,
                        end_line=cell_end_line,
                        docstring=None,
                        parent_class=None,
                        decorators=[],
                    )
                    all_chunks.append(chunk)
                    continue

                # Python code cell: sanitize magics and attempt AST parsing
                sanitized_code = _sanitize_ipython_magics(raw_source)

                try:
                    tree = ast.parse(sanitized_code, filename=f"{path_str}#cell_{cell_idx}")

                    # Track counts before visiting this cell
                    prev_func_count = len(extractor.functions)
                    prev_class_count = len(extractor.classes)
                    prev_import_count = len(extractor.imports)
                    prev_call_count = len(extractor.calls)

                    extractor.visit(tree)

                    # Newly discovered items in this cell
                    cell_funcs = extractor.functions[prev_func_count:]
                    cell_classes = extractor.classes[prev_class_count:]
                    cell_imports = extractor.imports[prev_import_count:]
                    cell_calls = extractor.calls[prev_call_count:]

                    # Line offset for this cell: line 1 in cell -> cell_start_line
                    line_offset = cell_start_line - 1

                    # Adjust line numbers for functions
                    adjusted_funcs: List[FunctionInfo] = []
                    for f in cell_funcs:
                        adj_f = FunctionInfo(
                            name=f.name,
                            start_line=f.start_line + line_offset,
                            end_line=f.end_line + line_offset,
                            docstring=f.docstring,
                            parameters=f.parameters,
                            decorators=f.decorators,
                            is_async=f.is_async,
                            is_method=f.is_method,
                            parent_class=f.parent_class,
                        )
                        adjusted_funcs.append(adj_f)
                        all_functions.append(adj_f)

                    # Adjust line numbers for classes and their methods
                    adjusted_classes: List[ClassInfo] = []
                    for c in cell_classes:
                        adj_methods = [
                            FunctionInfo(
                                name=m.name,
                                start_line=m.start_line + line_offset,
                                end_line=m.end_line + line_offset,
                                docstring=m.docstring,
                                parameters=m.parameters,
                                decorators=m.decorators,
                                is_async=m.is_async,
                                is_method=True,
                                parent_class=c.name,
                            )
                            for m in c.methods
                        ]
                        adj_c = ClassInfo(
                            name=c.name,
                            start_line=c.start_line + line_offset,
                            end_line=c.end_line + line_offset,
                            docstring=c.docstring,
                            bases=c.bases,
                            decorators=c.decorators,
                            methods=adj_methods,
                        )
                        adjusted_classes.append(adj_c)
                        all_classes.append(adj_c)

                    # Adjust line numbers for imports
                    for imp in cell_imports:
                        adj_imp = ImportInfo(
                            module=imp.module,
                            line=imp.line + line_offset,
                            is_from_import=imp.is_from_import,
                            names=imp.names,
                            aliases=imp.aliases,
                        )
                        all_imports.append(adj_imp)

                    # Adjust line numbers for calls
                    module_name = filepath.with_suffix("").as_posix().replace("/", ".")
                    for call in cell_calls:
                        qualname = call.callee_qualname
                        if qualname and qualname.startswith("<module>."):
                            qualname = qualname.replace("<module>.", f"{module_name}.")

                        adj_call = CallInfo(
                            caller_name=call.caller_name,
                            caller_type=call.caller_type,
                            caller_class=call.caller_class,
                            caller_start_line=call.caller_start_line + line_offset,
                            callee_name=call.callee_name,
                            callee_qualname=qualname,
                            call_line=call.call_line + line_offset,
                            is_resolved=call.is_resolved,
                            resolution_type=call.resolution_type,
                        )
                        all_calls.append(adj_call)

                    # Chunk generation for the code cell
                    has_symbols = bool(adjusted_funcs or adjusted_classes)

                    if not has_symbols:
                        # Code cell without functions/classes -> notebook_code chunk
                        chunk = CodeChunk(
                            chunk_id=f"{path_str}::cell_{cell_idx}::code",
                            filepath=filepath,
                            chunk_type="notebook_code",
                            symbol_name=None,
                            source_code=raw_source,
                            start_line=cell_start_line,
                            end_line=cell_end_line,
                            docstring=None,
                            parent_class=None,
                            decorators=[],
                        )
                        all_chunks.append(chunk)
                    else:
                        # Extract chunks for functions
                        for func_info in adjusted_funcs:
                            # Extract function source text from cell_lines (1-indexed relative to cell)
                            f_cell_start = func_info.start_line - line_offset
                            f_cell_end = func_info.end_line - line_offset
                            func_code = "".join(
                                cell_lines[max(0, f_cell_start - 1) : min(len(cell_lines), f_cell_end)]
                            )

                            func_chunk = CodeChunk(
                                chunk_id=f"{path_str}::cell_{cell_idx}::{func_info.name}@{func_info.start_line}",
                                filepath=filepath,
                                chunk_type="function",
                                symbol_name=func_info.name,
                                source_code=func_code,
                                start_line=func_info.start_line,
                                end_line=func_info.end_line,
                                docstring=func_info.docstring,
                                parent_class=None,
                                decorators=func_info.decorators,
                            )
                            all_chunks.append(func_chunk)

                        # Extract chunks for classes
                        for class_info in adjusted_classes:
                            c_line_count = class_info.end_line - class_info.start_line + 1
                            c_cell_start = class_info.start_line - line_offset
                            c_cell_end = class_info.end_line - line_offset

                            if c_line_count <= self.max_chunk_lines:
                                class_code = "".join(
                                    cell_lines[max(0, c_cell_start - 1) : min(len(cell_lines), c_cell_end)]
                                )
                                class_chunk = CodeChunk(
                                    chunk_id=f"{path_str}::cell_{cell_idx}::{class_info.name}@{class_info.start_line}",
                                    filepath=filepath,
                                    chunk_type="class",
                                    symbol_name=class_info.name,
                                    source_code=class_code,
                                    start_line=class_info.start_line,
                                    end_line=class_info.end_line,
                                    docstring=class_info.docstring,
                                    parent_class=None,
                                    decorators=class_info.decorators,
                                )
                                all_chunks.append(class_chunk)
                            else:
                                # Large class: split into header + method chunks
                                first_method_start = (
                                    min(m.start_line for m in class_info.methods)
                                    if class_info.methods
                                    else class_info.end_line + 1
                                )
                                header_end_line = max(class_info.start_line, first_method_start - 1)
                                h_cell_end = header_end_line - line_offset

                                header_code = "".join(
                                    cell_lines[max(0, c_cell_start - 1) : min(len(cell_lines), h_cell_end)]
                                )
                                header_chunk = CodeChunk(
                                    chunk_id=f"{path_str}::cell_{cell_idx}::{class_info.name}__header",
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
                                all_chunks.append(header_chunk)

                                for method_info in class_info.methods:
                                    m_cell_start = method_info.start_line - line_offset
                                    m_cell_end = method_info.end_line - line_offset
                                    method_code = "".join(
                                        cell_lines[max(0, m_cell_start - 1) : min(len(cell_lines), m_cell_end)]
                                    )
                                    method_chunk = CodeChunk(
                                        chunk_id=f"{path_str}::cell_{cell_idx}::{class_info.name}.{method_info.name}@{method_info.start_line}",
                                        filepath=filepath,
                                        chunk_type="method",
                                        symbol_name=method_info.name,
                                        source_code=method_code,
                                        start_line=method_info.start_line,
                                        end_line=method_info.end_line,
                                        docstring=method_info.docstring,
                                        parent_class=class_info.name,
                                        decorators=method_info.decorators,
                                    )
                                    all_chunks.append(method_chunk)

                except SyntaxError:
                    # Cell has syntax error: fallback to raw notebook_code chunk for this cell
                    chunk = CodeChunk(
                        chunk_id=f"{path_str}::cell_{cell_idx}::code",
                        filepath=filepath,
                        chunk_type="notebook_code",
                        symbol_name=None,
                        source_code=raw_source,
                        start_line=cell_start_line,
                        end_line=cell_end_line,
                        docstring=None,
                        parent_class=None,
                        decorators=[],
                    )
                    all_chunks.append(chunk)

            # Case C: Raw / other cell type
            else:
                chunk = CodeChunk(
                    chunk_id=f"{path_str}::cell_{cell_idx}::code",
                    filepath=filepath,
                    chunk_type="notebook_code",
                    symbol_name=None,
                    source_code=raw_source,
                    start_line=cell_start_line,
                    end_line=cell_end_line,
                    docstring=None,
                    parent_class=None,
                    decorators=[],
                )
                all_chunks.append(chunk)

        # If notebook yielded no chunks at all (e.g. empty notebook), create a single file chunk
        if not all_chunks:
            all_chunks.append(
                CodeChunk(
                    chunk_id=f"{path_str}::file",
                    filepath=filepath,
                    chunk_type="file",
                    symbol_name=None,
                    source_code=content,
                    start_line=1,
                    end_line=1,
                    docstring=module_docstring,
                    parent_class=None,
                    decorators=[],
                )
            )

        parsed_file = ParsedFile(
            filepath=filepath,
            functions=all_functions,
            classes=all_classes,
            imports=all_imports,
            calls=all_calls,
            module_docstring=module_docstring,
            has_syntax_error=False,
            syntax_error_message=None,
        )

        return parsed_file, all_chunks


def parse_notebook(
    content: str, filepath: Path, max_chunk_lines: Optional[int] = None
) -> Tuple[ParsedFile, List[CodeChunk]]:
    """Convenience function to parse a Jupyter notebook (.ipynb).

    Args:
        content: Raw JSON text of the notebook
        filepath: Path to the notebook
        max_chunk_lines: Optional line limit for class chunk splitting

    Returns:
        Tuple of (ParsedFile, List[CodeChunk])
    """
    parser = NotebookParser(max_chunk_lines=max_chunk_lines)
    return parser.parse(content, filepath)
