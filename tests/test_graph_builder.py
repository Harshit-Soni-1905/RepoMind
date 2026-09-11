"""Tests for CodeGraphBuilder (Stage 5)."""

from pathlib import Path
import pytest
import networkx as nx

from repomind.parsing.models import ParsedFile, FunctionInfo, ClassInfo, ImportInfo, CallInfo
from repomind.graph.models import NodeType, EdgeType, GraphNode
from repomind.graph.builder import CodeGraphBuilder


@pytest.fixture
def sample_parsed_files():
    """Create sample ParsedFile objects representing a multi-file Python project."""
    # file_a.py imports file_b and defines a function
    file_a = ParsedFile(
        filepath=Path("pkg/file_a.py"),
        module_docstring="File A module docstring.",
        imports=[
            ImportInfo(module="pkg.file_b", names=["func_b"], is_from_import=True, line=1),
            ImportInfo(module="os", names=[], is_from_import=False, line=2),
        ],
        functions=[
            FunctionInfo(
                name="func_a",
                start_line=5,
                end_line=10,
                docstring="Docstring func_a",
                is_method=False,
            )
        ],
        classes=[],
    )

    # file_b.py defines a class with a method and a function
    file_b = ParsedFile(
        filepath=Path("pkg/file_b.py"),
        module_docstring="File B module docstring.",
        imports=[],
        functions=[
            FunctionInfo(
                name="func_b",
                start_line=3,
                end_line=8,
                docstring="Docstring func_b",
                is_method=False,
            )
        ],
        classes=[
            ClassInfo(
                name="ClassB",
                start_line=10,
                end_line=20,
                docstring="Docstring ClassB",
                methods=[
                    FunctionInfo(
                        name="method_b",
                        start_line=12,
                        end_line=15,
                        docstring="Docstring method_b",
                        is_method=True,
                        parent_class="ClassB",
                    )
                ],
            )
        ],
    )

    # sub/file_c.py uses relative import to import file_a
    file_c = ParsedFile(
        filepath=Path("pkg/sub/file_c.py"),
        module_docstring="File C module docstring.",
        imports=[
            ImportInfo(module="..file_a", names=["func_a"], is_from_import=True, line=1),
        ],
        functions=[],
        classes=[],
    )

    return [file_a, file_b, file_c]


def test_build_graph_node_creation(sample_parsed_files):
    """Test that all files, classes, functions, and methods are created as nodes."""
    builder = CodeGraphBuilder()
    graph = builder.build_graph(sample_parsed_files)

    # Expected node IDs
    file_a_id = "pkg/file_a.py"
    func_a_id = "pkg/file_a.py::func_a"
    file_b_id = "pkg/file_b.py"
    func_b_id = "pkg/file_b.py::func_b"
    class_b_id = "pkg/file_b.py::ClassB"
    method_b_id = "pkg/file_b.py::ClassB.method_b"
    file_c_id = "pkg/sub/file_c.py"

    assert file_a_id in graph
    assert func_a_id in graph
    assert file_b_id in graph
    assert func_b_id in graph
    assert class_b_id in graph
    assert method_b_id in graph
    assert file_c_id in graph

    # Verify node types and metadata
    node_a_data = graph.nodes[file_a_id]
    assert node_a_data["node_type"] == NodeType.FILE.value
    assert node_a_data["docstring"] == "File A module docstring."

    func_a_data = graph.nodes[func_a_id]
    assert func_a_data["node_type"] == NodeType.FUNCTION.value
    assert func_a_data["symbol_name"] == "func_a"
    assert func_a_data["start_line"] == 5

    class_b_data = graph.nodes[class_b_id]
    assert class_b_data["node_type"] == NodeType.CLASS.value
    assert class_b_data["symbol_name"] == "ClassB"

    method_b_data = graph.nodes[method_b_id]
    assert method_b_data["node_type"] == NodeType.METHOD.value
    assert method_b_data["symbol_name"] == "method_b"


def test_build_graph_edges(sample_parsed_files):
    """Test DEFINES, CONTAINS, and IMPORTS edges in graph."""
    builder = CodeGraphBuilder()
    graph = builder.build_graph(sample_parsed_files)

    file_a_id = "pkg/file_a.py"
    func_a_id = "pkg/file_a.py::func_a"
    file_b_id = "pkg/file_b.py"
    func_b_id = "pkg/file_b.py::func_b"
    class_b_id = "pkg/file_b.py::ClassB"
    method_b_id = "pkg/file_b.py::ClassB.method_b"
    file_c_id = "pkg/sub/file_c.py"

    # File DEFINES function / class
    assert graph.has_edge(file_a_id, func_a_id)
    assert graph[file_a_id][func_a_id]["edge_type"] == EdgeType.DEFINES

    assert graph.has_edge(file_b_id, func_b_id)
    assert graph[file_b_id][func_b_id]["edge_type"] == EdgeType.DEFINES

    assert graph.has_edge(file_b_id, class_b_id)
    assert graph[file_b_id][class_b_id]["edge_type"] == EdgeType.DEFINES

    # Class CONTAINS method
    assert graph.has_edge(class_b_id, method_b_id)
    assert graph[class_b_id][method_b_id]["edge_type"] == EdgeType.CONTAINS

    # Absolute IMPORTS edge (file_a imports pkg.file_b)
    assert graph.has_edge(file_a_id, file_b_id)
    assert graph[file_a_id][file_b_id]["edge_type"] == EdgeType.IMPORTS

    # Relative IMPORTS edge (file_c imports ..file_a)
    assert graph.has_edge(file_c_id, file_a_id)
    assert graph[file_c_id][file_a_id]["edge_type"] == EdgeType.IMPORTS


def test_unresolved_and_stdlib_imports_ignored():
    """Test that stdlib or unindexed external imports are safely ignored."""
    parsed_file = ParsedFile(
        filepath=Path("main.py"),
        module_docstring=None,
        imports=[
            ImportInfo(module="sys", names=[], is_from_import=False, line=1),
            ImportInfo(module="non_existent_pkg", names=[], is_from_import=False, line=2),
        ],
        functions=[],
        classes=[],
    )
    builder = CodeGraphBuilder()
    graph = builder.build_graph([parsed_file])

    assert "main.py" in graph
    assert len(graph.edges) == 0


def test_build_graph_call_edges_local_function():
    """Test CALLS edges for local function calls within same file."""
    file_a = ParsedFile(
        filepath=Path("pkg/file_a.py"),
        module_docstring="File A module docstring.",
        imports=[],
        functions=[
            FunctionInfo(
                name="func_a",
                start_line=5,
                end_line=10,
                docstring="Docstring func_a",
                is_method=False,
            ),
            FunctionInfo(
                name="func_b",
                start_line=12,
                end_line=15,
                docstring="Docstring func_b",
                is_method=False,
            ),
        ],
        classes=[],
        calls=[
            CallInfo(
                caller_name="func_a",
                caller_type="function",
                caller_class=None,
                caller_start_line=5,
                callee_name="func_b",
                callee_qualname="pkg.file_a.func_b",
                call_line=7,
                is_resolved=True,
                resolution_type="local",
            ),
        ],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_a])

    file_a_id = "pkg/file_a.py"
    func_a_id = "pkg/file_a.py::func_a"
    func_b_id = "pkg/file_a.py::func_b"

    # CALLS edge from func_a to func_b
    assert graph.has_edge(func_a_id, func_b_id)
    assert graph[func_a_id][func_b_id]["edge_type"] == EdgeType.CALLS


def test_build_graph_call_edges_method_call():
    """Test CALLS edges for method calls on self."""
    file_b = ParsedFile(
        filepath=Path("pkg/file_b.py"),
        module_docstring="File B module docstring.",
        imports=[],
        functions=[],
        classes=[
            ClassInfo(
                name="ClassB",
                start_line=10,
                end_line=20,
                docstring="Docstring ClassB",
                methods=[
                    FunctionInfo(
                        name="method_a",
                        start_line=12,
                        end_line=15,
                        docstring="Docstring method_a",
                        is_method=True,
                        parent_class="ClassB",
                    ),
                    FunctionInfo(
                        name="method_b",
                        start_line=17,
                        end_line=19,
                        docstring="Docstring method_b",
                        is_method=True,
                        parent_class="ClassB",
                    ),
                ],
            )
        ],
        calls=[
            CallInfo(
                caller_name="method_a",
                caller_type="method",
                caller_class="ClassB",
                caller_start_line=12,
                callee_name="method_b",
                callee_qualname="pkg.file_b.ClassB.method_b",
                call_line=14,
                is_resolved=True,
                resolution_type="method",
            ),
        ],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_b])

    class_b_id = "pkg/file_b.py::ClassB"
    method_a_id = "pkg/file_b.py::ClassB.method_a"
    method_b_id = "pkg/file_b.py::ClassB.method_b"

    # CALLS edge from method_a to method_b
    assert graph.has_edge(method_a_id, method_b_id)
    assert graph[method_a_id][method_b_id]["edge_type"] == EdgeType.CALLS


def test_build_graph_call_edges_imported_function():
    """Test CALLS edges for calls to imported functions."""
    file_a = ParsedFile(
        filepath=Path("pkg/file_a.py"),
        module_docstring="File A module docstring.",
        imports=[
            ImportInfo(module="pkg.helpers", names=["helper_func"], is_from_import=True, line=1),
        ],
        functions=[
            FunctionInfo(
                name="func_a",
                start_line=5,
                end_line=10,
                docstring="Docstring func_a",
                is_method=False,
            ),
        ],
        classes=[],
        calls=[
            CallInfo(
                caller_name="func_a",
                caller_type="function",
                caller_class=None,
                caller_start_line=5,
                callee_name="helper_func",
                callee_qualname="pkg.helpers.helper_func",
                call_line=7,
                is_resolved=True,
                resolution_type="imported",
            ),
        ],
    )

    file_b = ParsedFile(
        filepath=Path("pkg/helpers.py"),
        module_docstring="Helpers module.",
        imports=[],
        functions=[
            FunctionInfo(
                name="helper_func",
                start_line=3,
                end_line=8,
                docstring="Docstring helper_func",
                is_method=False,
            ),
        ],
        classes=[],
        calls=[],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_a, file_b])

    func_a_id = "pkg/file_a.py::func_a"
    helper_func_id = "pkg/helpers.py::helper_func"

    # CALLS edge from func_a to helper_func (imported)
    assert graph.has_edge(func_a_id, helper_func_id)
    assert graph[func_a_id][helper_func_id]["edge_type"] == EdgeType.CALLS


def test_build_graph_call_edges_unresolved_skipped():
    """Test that unresolved calls do not create CALLS edges (conservative approach)."""
    file_a = ParsedFile(
        filepath=Path("pkg/file_a.py"),
        module_docstring="File A module docstring.",
        imports=[],
        functions=[
            FunctionInfo(
                name="func_a",
                start_line=5,
                end_line=10,
                docstring="Docstring func_a",
                is_method=False,
            ),
        ],
        classes=[],
        calls=[
            CallInfo(
                caller_name="func_a",
                caller_type="function",
                caller_class=None,
                caller_start_line=5,
                callee_name="unknown_func",
                callee_qualname=None,
                call_line=7,
                is_resolved=False,
                resolution_type="unresolved",
            ),
        ],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_a])

    func_a_id = "pkg/file_a.py::func_a"

    # No CALLS edges should exist for unresolved calls
    outgoing_edges = list(graph.out_edges(func_a_id))
    calls_edges = [e for e in outgoing_edges if graph[e[0]][e[1]].get("edge_type") == EdgeType.CALLS]
    assert len(calls_edges) == 0


def test_build_graph_call_edges_no_self_loops():
    """Test that self-recursive calls don't create self-loop edges."""
    file_a = ParsedFile(
        filepath=Path("pkg/file_a.py"),
        module_docstring="File A module docstring.",
        imports=[],
        functions=[
            FunctionInfo(
                name="func_a",
                start_line=5,
                end_line=10,
                docstring="Docstring func_a",
                is_method=False,
            ),
        ],
        classes=[],
        calls=[
            CallInfo(
                caller_name="func_a",
                caller_type="function",
                caller_class=None,
                caller_start_line=5,
                callee_name="func_a",
                callee_qualname="pkg.file_a.func_a",
                call_line=7,
                is_resolved=True,
                resolution_type="local",
            ),
        ],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_a])

    func_a_id = "pkg/file_a.py::func_a"

    # No self-loop CALLS edges
    assert not graph.has_edge(func_a_id, func_a_id)
