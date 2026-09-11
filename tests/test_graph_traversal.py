"""Tests for GraphTraverser (Stage 5)."""

from pathlib import Path
import pytest

from repomind.parsing.models import ParsedFile, FunctionInfo, ClassInfo, ImportInfo, CallInfo
from repomind.graph.models import NodeType, EdgeType
from repomind.graph.builder import CodeGraphBuilder
from repomind.graph.traversal import GraphTraverser


@pytest.fixture
def populated_traverser():
    """Create a GraphTraverser initialized with a multi-file dependency graph."""
    # Topology:
    # utils.py
    # models.py imports utils.py
    # service.py imports models.py and utils.py
    # app.py imports service.py

    utils_file = ParsedFile(
        filepath=Path("app/utils.py"),
        module_docstring="Utils",
        imports=[],
        functions=[FunctionInfo(name="helper", start_line=1, end_line=5, is_method=False)],
        classes=[],
    )

    models_file = ParsedFile(
        filepath=Path("app/models.py"),
        module_docstring="Models",
        imports=[ImportInfo(module="app.utils", names=["helper"], is_from_import=True, line=1)],
        functions=[],
        classes=[ClassInfo(name="User", start_line=5, end_line=20, methods=[
            FunctionInfo(name="save", start_line=10, end_line=15, is_method=True, parent_class="User")
        ])],
    )

    service_file = ParsedFile(
        filepath=Path("app/service.py"),
        module_docstring="Service",
        imports=[
            ImportInfo(module="app.models", names=["User"], is_from_import=True, line=1),
            ImportInfo(module="app.utils", names=["helper"], is_from_import=True, line=2),
        ],
        functions=[FunctionInfo(name="process_user", start_line=5, end_line=15, is_method=False)],
        classes=[],
    )

    app_file = ParsedFile(
        filepath=Path("app/main.py"),
        module_docstring="Main app",
        imports=[ImportInfo(module="app.service", names=["process_user"], is_from_import=True, line=1)],
        functions=[],
        classes=[],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([utils_file, models_file, service_file, app_file])
    return GraphTraverser(graph)


def test_get_node(populated_traverser):
    """Test retrieving a single GraphNode object by ID."""
    node = populated_traverser.get_node("app/utils.py::helper")
    assert node is not None
    assert node.node_id == "app/utils.py::helper"
    assert node.node_type == NodeType.FUNCTION
    assert node.symbol_name == "helper"

    assert populated_traverser.get_node("non_existent") is None


def test_get_dependencies_depth_1(populated_traverser):
    """Test downstream dependencies at depth=1."""
    # service.py imports models.py, utils.py and DEFINES process_user
    # Filter by IMPORTS edge type to check module dependencies
    deps = populated_traverser.get_dependencies(
        "app/service.py",
        depth=1,
        edge_types=[EdgeType.IMPORTS]
    )
    dep_ids = {n.node_id for n in deps}
    assert dep_ids == {"app/models.py", "app/utils.py"}


def test_get_dependencies_transitive(populated_traverser):
    """Test transitive downstream dependencies with depth=None."""
    # app/main.py -> app/service.py -> app/models.py, app/utils.py
    deps = populated_traverser.get_dependencies(
        "app/main.py",
        depth=None,
        edge_types=[EdgeType.IMPORTS]
    )
    dep_ids = {n.node_id for n in deps}
    assert dep_ids == {"app/service.py", "app/models.py", "app/utils.py"}


def test_get_dependents_depth_1(populated_traverser):
    """Test upstream dependents at depth=1."""
    # utils.py is imported by models.py and service.py
    dependents = populated_traverser.get_dependents(
        "app/utils.py",
        depth=1,
        edge_types=[EdgeType.IMPORTS]
    )
    dependent_ids = {n.node_id for n in dependents}
    assert dependent_ids == {"app/models.py", "app/service.py"}


def test_get_dependents_transitive(populated_traverser):
    """Test transitive upstream dependents with depth=None."""
    # utils.py is depended on by models.py, service.py, and main.py
    dependents = populated_traverser.get_dependents(
        "app/utils.py",
        depth=None,
        edge_types=[EdgeType.IMPORTS]
    )
    dependent_ids = {n.node_id for n in dependents}
    assert dependent_ids == {"app/models.py", "app/service.py", "app/main.py"}


def test_get_neighbors(populated_traverser):
    """Test retrieving both incoming and outgoing direct neighbors."""
    # models.py imports utils.py, is imported by service.py, and defines User class
    neighbors = populated_traverser.get_neighbors("app/models.py")
    neighbor_ids = {n.node_id for n in neighbors}

    assert "app/utils.py" in neighbor_ids      # outgoing IMPORTS edge
    assert "app/service.py" in neighbor_ids    # incoming IMPORTS edge
    assert "app/models.py::User" in neighbor_ids # outgoing DEFINES edge


def test_get_dependencies_with_calls_edge(populated_traverser):
    """Test that dependencies can traverse CALLS edges."""
    # Create a graph with CALLS edges for testing
    file_a = ParsedFile(
        filepath=Path("pkg/module_a.py"),
        module_docstring="Module A",
        imports=[],
        functions=[
            FunctionInfo(name="func_a", start_line=1, end_line=10, is_method=False),
            FunctionInfo(name="func_b", start_line=12, end_line=15, is_method=False),
        ],
        classes=[],
        calls=[
            CallInfo(
                caller_name="func_a",
                caller_type="function",
                caller_class=None,
                caller_start_line=1,
                callee_name="func_b",
                callee_qualname="pkg.module_a.func_b",
                call_line=5,
                is_resolved=True,
                resolution_type="local",
            ),
        ],
    )

    file_b = ParsedFile(
        filepath=Path("pkg/module_b.py"),
        module_docstring="Module B",
        imports=[],
        functions=[],
        classes=[],
        calls=[],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_a, file_b])
    traverser = GraphTraverser(graph)

    func_a_id = "pkg/module_a.py::func_a"
    func_b_id = "pkg/module_a.py::func_b"

    # Get dependencies via CALLS edge
    deps = traverser.get_dependencies(
        func_a_id,
        depth=1,
        edge_types=[EdgeType.CALLS]
    )
    dep_ids = {n.node_id for n in deps}
    assert dep_ids == {func_b_id}


def test_get_dependents_with_calls_edge(populated_traverser):
    """Test that dependents can traverse CALLS edges in reverse."""
    # Create a graph with CALLS edges for testing
    file_a = ParsedFile(
        filepath=Path("pkg/module_a.py"),
        module_docstring="Module A",
        imports=[],
        functions=[
            FunctionInfo(name="func_a", start_line=1, end_line=10, is_method=False),
            FunctionInfo(name="func_b", start_line=12, end_line=15, is_method=False),
        ],
        classes=[],
        calls=[
            CallInfo(
                caller_name="func_a",
                caller_type="function",
                caller_class=None,
                caller_start_line=1,
                callee_name="func_b",
                callee_qualname="pkg.module_a.func_b",
                call_line=5,
                is_resolved=True,
                resolution_type="local",
            ),
        ],
    )

    file_b = ParsedFile(
        filepath=Path("pkg/module_b.py"),
        module_docstring="Module B",
        imports=[],
        functions=[],
        classes=[],
        calls=[],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_a, file_b])
    traverser = GraphTraverser(graph)

    func_a_id = "pkg/module_a.py::func_a"
    func_b_id = "pkg/module_a.py::func_b"

    # Get dependents via CALLS edge (reverse direction)
    dependents = traverser.get_dependents(
        func_b_id,
        depth=1,
        edge_types=[EdgeType.CALLS]
    )
    dependent_ids = {n.node_id for n in dependents}
    assert dependent_ids == {func_a_id}


def test_get_neighbors_with_calls_edge(populated_traverser):
    """Test that neighbors includes CALLS edge connections."""
    # Create a graph with CALLS edges for testing
    file_a = ParsedFile(
        filepath=Path("pkg/module_a.py"),
        module_docstring="Module A",
        imports=[],
        functions=[
            FunctionInfo(name="func_a", start_line=1, end_line=10, is_method=False),
            FunctionInfo(name="func_b", start_line=12, end_line=15, is_method=False),
        ],
        classes=[],
        calls=[
            CallInfo(
                caller_name="func_a",
                caller_type="function",
                caller_class=None,
                caller_start_line=1,
                callee_name="func_b",
                callee_qualname="pkg.module_a.func_b",
                call_line=5,
                is_resolved=True,
                resolution_type="local",
            ),
        ],
    )

    builder = CodeGraphBuilder()
    graph = builder.build_graph([file_a])
    traverser = GraphTraverser(graph)

    func_a_id = "pkg/module_a.py::func_a"
    func_b_id = "pkg/module_a.py::func_b"

    # Get neighbors - should include CALLS connections
    neighbors = traverser.get_neighbors(func_a_id)
    neighbor_ids = {n.node_id for n in neighbors}
    assert func_b_id in neighbor_ids

    neighbors_b = traverser.get_neighbors(func_b_id)
    neighbor_ids_b = {n.node_id for n in neighbors_b}
    assert func_a_id in neighbor_ids_b
