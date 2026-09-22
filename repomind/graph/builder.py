"""Graph builder for constructing code dependency graph from parsed files."""

from pathlib import Path
from typing import Dict, List, Optional, Set
import networkx as nx

from repomind.parsing.models import ParsedFile, FunctionInfo, ClassInfo, ImportInfo, CallInfo
from repomind.graph.models import NodeType, EdgeType, GraphNode


class CodeGraphBuilder:
    """Builds a directed graph representing code structure and dependencies.

    The graph captures:
    - File/module nodes
    - Class and function/method nodes
    - Import relationships (IMPORTS edges)
    - Containment relationships (DEFINES, CONTAINS edges)
    """

    def __init__(self):
        """Initialize the graph builder."""
        self.graph = nx.DiGraph()
        self._repo_root: Optional[Path] = None
        self._file_nodes: Set[str] = set()  # Track created file nodes

    def build_graph(
        self,
        parsed_files: List[ParsedFile],
        repo_root: Optional[Path] = None
    ) -> nx.DiGraph:
        """Build a code graph from a list of parsed files.

        Args:
            parsed_files: List of ParsedFile objects from Stage 2
            repo_root: Optional repository root for resolving relative imports

        Returns:
            NetworkX directed graph with nodes and edges
        """
        self.graph = nx.DiGraph()
        self._file_nodes = set()
        self._repo_root = repo_root

        # Build a mapping of module paths for import resolution
        self._module_map: Dict[str, str] = {}
        for pf in parsed_files:
            module_name = self._filepath_to_module_name(pf.filepath)
            file_id = pf.filepath.as_posix()
            self._module_map[module_name] = file_id

        # Build a mapping of symbol qualnames to node IDs for call resolution
        self._symbol_map: Dict[str, str] = {}
        for pf in parsed_files:
            self._build_symbol_map(pf)

        # Process each file
        for parsed_file in parsed_files:
            self._add_file_nodes(parsed_file)

        # Add import edges after all nodes are created
        for parsed_file in parsed_files:
            self._add_import_edges(parsed_file)

        # Add CALLS edges from call information
        for parsed_file in parsed_files:
            self._add_call_edges(parsed_file)

        # Clear temporary lookup dictionaries to free memory after graph build completes
        self._module_map.clear()
        self._symbol_map.clear()
        self._file_nodes.clear()

        return self.graph

    def _build_symbol_map(self, parsed_file: ParsedFile) -> None:
        """Build mapping from symbol qualnames to node IDs for call resolution.

        This creates a lookup for resolving callee names to graph node IDs.
        """
        file_id = parsed_file.filepath.as_posix()
        module_name = self._filepath_to_module_name(parsed_file.filepath)

        # Map top-level functions
        for func in parsed_file.functions:
            qualname = f"{module_name}.{func.name}"
            node_id = f"{file_id}::{func.name}"
            self._symbol_map[qualname] = node_id

        # Map classes and methods
        for cls in parsed_file.classes:
            class_qualname = f"{module_name}.{cls.name}"
            class_node_id = f"{file_id}::{cls.name}"
            self._symbol_map[class_qualname] = class_node_id

            for method in cls.methods:
                method_qualname = f"{module_name}.{cls.name}.{method.name}"
                method_node_id = f"{file_id}::{cls.name}.{method.name}"
                self._symbol_map[method_qualname] = method_node_id

    def _add_file_nodes(self, parsed_file: ParsedFile) -> None:
        """Add nodes and edges for a single parsed file."""
        filepath = parsed_file.filepath
        file_id = filepath.as_posix()

        # Create file node
        if file_id not in self._file_nodes:
            self._add_node(
                node_id=file_id,
                node_type=NodeType.FILE,
                filepath=filepath,
                symbol_name=None,
                docstring=parsed_file.module_docstring,
            )
            self._file_nodes.add(file_id)

        # Add top-level functions
        for func_info in parsed_file.functions:
            if not func_info.is_method:  # Methods handled with classes
                self._add_function_node(func_info, filepath, file_id)

        # Add classes and their methods
        for class_info in parsed_file.classes:
            self._add_class_node(class_info, filepath, file_id)

    def _add_function_node(
        self,
        func_info: FunctionInfo,
        filepath: Path,
        file_id: str
    ) -> None:
        """Add a function node and link it to its file."""
        path_str = filepath.as_posix()
        func_id = f"{path_str}::{func_info.name}"

        self._add_node(
            node_id=func_id,
            node_type=NodeType.FUNCTION,
            filepath=filepath,
            symbol_name=func_info.name,
            start_line=func_info.start_line,
            end_line=func_info.end_line,
            docstring=func_info.docstring,
        )

        # File DEFINES function
        self.graph.add_edge(file_id, func_id, edge_type=EdgeType.DEFINES)

    def _add_class_node(
        self,
        class_info: ClassInfo,
        filepath: Path,
        file_id: str
    ) -> None:
        """Add a class node and its methods, link to file."""
        path_str = filepath.as_posix()
        class_id = f"{path_str}::{class_info.name}"

        self._add_node(
            node_id=class_id,
            node_type=NodeType.CLASS,
            filepath=filepath,
            symbol_name=class_info.name,
            start_line=class_info.start_line,
            end_line=class_info.end_line,
            docstring=class_info.docstring,
        )

        # File DEFINES class
        self.graph.add_edge(file_id, class_id, edge_type=EdgeType.DEFINES)

        # Add methods
        for method_info in class_info.methods:
            self._add_method_node(method_info, filepath, class_id)

    def _add_method_node(
        self,
        method_info: FunctionInfo,
        filepath: Path,
        class_id: str
    ) -> None:
        """Add a method node and link it to its class."""
        path_str = filepath.as_posix()
        parent_class = method_info.parent_class or ""
        method_id = f"{path_str}::{parent_class}.{method_info.name}"

        self._add_node(
            node_id=method_id,
            node_type=NodeType.METHOD,
            filepath=filepath,
            symbol_name=method_info.name,
            start_line=method_info.start_line,
            end_line=method_info.end_line,
            docstring=method_info.docstring,
        )

        # Class CONTAINS method
        self.graph.add_edge(class_id, method_id, edge_type=EdgeType.CONTAINS)

    def _add_node(
        self,
        node_id: str,
        node_type: NodeType,
        filepath: Path,
        symbol_name: Optional[str],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        docstring: Optional[str] = None,
    ) -> None:
        """Add a node to the graph with metadata."""
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            filepath=filepath,
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
        )

        self.graph.add_node(
            node_id,
            node_type=node_type.value,
            filepath=str(filepath.as_posix()),
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            node_obj=node,  # Store the GraphNode object
        )

    def _add_import_edges(self, parsed_file: ParsedFile) -> None:
        """Add IMPORTS edges based on import statements."""
        source_file_id = parsed_file.filepath.as_posix()

        for import_info in parsed_file.imports:
            target_file_id = self._resolve_import(import_info, parsed_file.filepath)

            if target_file_id and target_file_id in self._file_nodes:
                # Create IMPORTS edge
                self.graph.add_edge(
                    source_file_id,
                    target_file_id,
                    edge_type=EdgeType.IMPORTS
                )

    def _add_call_edges(self, parsed_file: ParsedFile) -> None:
        """Add CALLS edges based on call information from parsing.

        Resolves callee qualnames to existing graph nodes and creates CALLS edges.
        Only adds edges for confidently resolved calls (local, imported, method).
        """
        source_file_id = parsed_file.filepath.as_posix()

        for call_info in parsed_file.calls:
            if not call_info.is_resolved:
                continue  # Skip unresolved calls - conservative approach

            # Determine caller node ID
            caller_node_id = self._get_caller_node_id(call_info, source_file_id)
            if not caller_node_id or caller_node_id not in self.graph:
                continue

            # Determine callee node ID from qualname
            callee_node_id = self._symbol_map.get(call_info.callee_qualname)
            if not callee_node_id or callee_node_id not in self.graph:
                continue

            # Don't add self-loops
            if caller_node_id == callee_node_id:
                continue

            # Add CALLS edge
            self.graph.add_edge(
                caller_node_id,
                callee_node_id,
                edge_type=EdgeType.CALLS
            )

    def _get_caller_node_id(self, call_info: CallInfo, source_file_id: str) -> Optional[str]:
        """Get the node ID for the caller based on call info."""
        module_name = self._filepath_to_module_name(Path(source_file_id))

        if call_info.caller_type == "function":
            caller_qualname = f"{module_name}.{call_info.caller_name}"
        elif call_info.caller_type == "method" and call_info.caller_class:
            caller_qualname = f"{module_name}.{call_info.caller_class}.{call_info.caller_name}"
        else:
            return None

        return self._symbol_map.get(caller_qualname)

    def _resolve_import(
        self,
        import_info: ImportInfo,
        current_file: Path
    ) -> Optional[str]:
        """Resolve an import to a file ID.

        Handles:
        - Absolute imports: import os, from pathlib import Path
        - Relative imports: from .module import X, from ..package import Y

        Returns:
            File ID (POSIX path string) if resolved, None otherwise
        """
        module = import_info.module

        # Handle relative imports
        if module.startswith('.'):
            return self._resolve_relative_import(module, current_file)

        # Try direct module name lookup
        if module in self._module_map:
            return self._module_map[module]

        # Try as a file path (e.g., "subdir.module" -> "subdir/module.py")
        potential_path = module.replace('.', '/')

        # Check with .py extension
        potential_file_id = f"{potential_path}.py"
        if potential_file_id in self._file_nodes:
            return potential_file_id

        # Check as package __init__.py
        potential_init = f"{potential_path}/__init__.py"
        if potential_init in self._file_nodes:
            return potential_init

        # Unresolved import (stdlib, external package, or missing file)
        return None

    def _resolve_relative_import(
        self,
        module: str,
        current_file: Path
    ) -> Optional[str]:
        """Resolve a relative import like 'from .module import X'.

        Args:
            module: Import module string starting with dots
            current_file: Path of the file containing the import

        Returns:
            Resolved file ID or None if unresolved
        """
        # Count leading dots
        level = 0
        for char in module:
            if char == '.':
                level += 1
            else:
                break

        # Remaining module name after dots
        relative_module = module[level:] if level < len(module) else ""

        # Navigate up 'level' directories from current file
        current_dir = current_file.parent
        for _ in range(level - 1):  # level=1 means same dir, level=2 means parent, etc.
            current_dir = current_dir.parent
            if current_dir == current_dir.parent:  # Reached root
                return None

        # Build target path
        if relative_module:
            target_parts = relative_module.split('.')
            target_path = current_dir / Path(*target_parts)
        else:
            target_path = current_dir

        # Try as module file
        target_file = target_path.with_suffix('.py')
        file_id = target_file.as_posix()
        if file_id in self._file_nodes:
            return file_id

        # Try as package __init__.py
        target_init = target_path / '__init__.py'
        file_id = target_init.as_posix()
        if file_id in self._file_nodes:
            return file_id

        return None

    def _filepath_to_module_name(self, filepath: Path) -> str:
        """Convert a filepath to a Python module name.

        Examples:
            "repomind/parsing/ast_parser.py" -> "repomind.parsing.ast_parser"
            "utils.py" -> "utils"
        """
        # Remove .py extension
        parts = filepath.with_suffix('').parts

        # Join with dots
        module_name = '.'.join(parts)

        return module_name
