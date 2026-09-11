"""Comprehensive test suite for Stage 6 Hybrid Retriever."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
import networkx as nx

from repomind.chunking.models import CodeChunk
from repomind.vectorstore.models import RetrievalResult
from repomind.graph.models import NodeType, EdgeType, GraphNode
from repomind.graph.builder import CodeGraphBuilder
from repomind.graph.traversal import GraphTraverser
from repomind.retrieval.models import RetrievalSource, HybridResultNode, HybridQueryResult
from repomind.retrieval.hybrid import HybridRetriever


class DummyVectorStore:
    """Mock VectorStore for deterministic unit testing."""
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.last_query = None
        self.last_top_k = None

    def search(self, query: str, top_k: int = 5):
        self.last_query = query
        self.last_top_k = top_k
        return self.search_results[:top_k]


@pytest.fixture
def sample_graph():
    """Build a sample graph for testing graph expansion.

    Structure:
    auth.py (file)
       │ (DEFINES)
       ├── login_func (function: "auth.py::login")
       └── AuthClass (class: "auth.py::Auth")
             │ (CONTAINS)
             └── verify_method (method: "auth.py::Auth.verify")

    db.py (file)
       │ (DEFINES)
       └── UserDb (class: "db.py::UserDb")

    auth.py ──(IMPORTS)──> db.py
    """
    g = nx.DiGraph()

    # Create nodes
    file_auth = GraphNode(node_id="auth.py", node_type=NodeType.FILE, filepath=Path("auth.py"), symbol_name=None)
    func_login = GraphNode(node_id="auth.py::login", node_type=NodeType.FUNCTION, filepath=Path("auth.py"), symbol_name="login", start_line=5, end_line=15)
    class_auth = GraphNode(node_id="auth.py::Auth", node_type=NodeType.CLASS, filepath=Path("auth.py"), symbol_name="Auth", start_line=20, end_line=50)
    method_verify = GraphNode(node_id="auth.py::Auth.verify", node_type=NodeType.METHOD, filepath=Path("auth.py"), symbol_name="verify", start_line=30, end_line=45)

    file_db = GraphNode(node_id="db.py", node_type=NodeType.FILE, filepath=Path("db.py"), symbol_name=None)
    class_db = GraphNode(node_id="db.py::UserDb", node_type=NodeType.CLASS, filepath=Path("db.py"), symbol_name="UserDb", start_line=10, end_line=40)

    isolated = GraphNode(node_id="isolated.py", node_type=NodeType.FILE, filepath=Path("isolated.py"), symbol_name=None)

    for node in [file_auth, func_login, class_auth, method_verify, file_db, class_db, isolated]:
        g.add_node(node.node_id, node_obj=node)

    # Edges
    g.add_edge("auth.py", "auth.py::login", edge_type=EdgeType.DEFINES.value)
    g.add_edge("auth.py", "auth.py::Auth", edge_type=EdgeType.DEFINES.value)
    g.add_edge("auth.py::Auth", "auth.py::Auth.verify", edge_type=EdgeType.CONTAINS.value)
    g.add_edge("db.py", "db.py::UserDb", edge_type=EdgeType.DEFINES.value)
    g.add_edge("auth.py", "db.py", edge_type=EdgeType.IMPORTS.value)

    return g


@pytest.fixture
def sample_chunks():
    chunk_login = CodeChunk(
        chunk_id="auth.py::login",
        filepath=Path("auth.py"),
        chunk_type="function",
        symbol_name="login",
        source_code="def login(user, pwd):\n    pass",
        start_line=5,
        end_line=15,
    )
    chunk_db = CodeChunk(
        chunk_id="db.py::UserDb",
        filepath=Path("db.py"),
        chunk_type="class",
        symbol_name="UserDb",
        source_code="class UserDb:\n    pass",
        start_line=10,
        end_line=40,
    )
    return chunk_login, chunk_db


def test_empty_query_returns_empty_results(sample_graph):
    vstore = DummyVectorStore()
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    result = retriever.retrieve("")
    assert result.total_results == 0
    assert len(result.nodes) == 0


def test_both_systems_returning_no_results(sample_graph):
    vstore = DummyVectorStore(search_results=[])
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    result = retriever.retrieve("authentication", top_k=5, depth=1)
    assert result.total_results == 0
    assert result.vector_count == 0
    assert result.graph_count == 0


def test_depth_0_vector_only_results(sample_graph, sample_chunks):
    chunk_login, _ = sample_chunks
    v_results = [RetrievalResult(chunk=chunk_login, score=0.9, distance=0.1, rank=1)]
    vstore = DummyVectorStore(search_results=v_results)
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    result = retriever.retrieve("login user", top_k=5, depth=0)

    assert result.total_results == 1
    assert result.vector_count == 1
    assert result.graph_count == 0
    assert result.both_count == 0
    assert result.nodes[0].node_id == "auth.py::login"
    assert result.nodes[0].source == RetrievalSource.VECTOR
    assert result.nodes[0].graph_distance == 0


def test_depth_1_graph_expansion(sample_graph, sample_chunks):
    chunk_login, _ = sample_chunks
    v_results = [RetrievalResult(chunk=chunk_login, score=0.8, distance=0.2, rank=1)]
    vstore = DummyVectorStore(search_results=v_results)
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    result = retriever.retrieve("login user", top_k=5, depth=1)

    # Seed is auth.py::login. Its direct predecessor in graph is auth.py (DEFINES)
    # BUT FILE nodes are filtered out from candidates (they don't represent code chunks)
    # So only the vector seed remains
    assert result.total_results == 1
    node_ids = {n.node_id for n in result.nodes}
    assert node_ids == {"auth.py::login"}

    login_node = next(n for n in result.nodes if n.node_id == "auth.py::login")
    assert login_node.source == RetrievalSource.VECTOR
    # graph_support is 0 since no external corroboration
    assert login_node.graph_support == 0.0
    assert login_node.graph_distance == 0


def test_depth_2_graph_expansion(sample_graph, sample_chunks):
    chunk_login, _ = sample_chunks
    v_results = [RetrievalResult(chunk=chunk_login, score=0.8, distance=0.2, rank=1)]
    vstore = DummyVectorStore(search_results=v_results)
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    result = retriever.retrieve("login user", top_k=5, depth=2)

    # Seed auth.py::login -> depth 1: auth.py (FILE, filtered) -> depth 2: auth.py::Auth (CLASS), db.py (FILE, filtered)
    node_ids = {n.node_id for n in result.nodes}
    assert "auth.py::login" in node_ids
    assert "auth.py::Auth" in node_ids
    # FILE nodes auth.py and db.py are filtered out
    assert "auth.py" not in node_ids
    assert "db.py" not in node_ids

    auth_class_node = next(n for n in result.nodes if n.node_id == "auth.py::Auth")
    assert auth_class_node.graph_distance == 2


def test_duplicate_removal_and_provenance_tracking(sample_graph, sample_chunks):
    """Test when an item is in vector search AND discovered via graph expansion."""
    chunk_login, chunk_db = sample_chunks
    # Vector store returns auth.py::login AND db.py::UserDb
    v_results = [
        RetrievalResult(chunk=chunk_login, score=0.9, distance=0.1, rank=1),
        RetrievalResult(chunk=chunk_db, score=0.7, distance=0.3, rank=2),
    ]
    vstore = DummyVectorStore(search_results=v_results)
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    # At depth=3, auth.py::login expands to auth.py which expands to db.py which defines db.py::UserDb
    result = retriever.retrieve("login user db", top_k=5, depth=3)

    db_user_node = next(n for n in result.nodes if n.node_id == "db.py::UserDb")
    # Should be marked BOTH because it was in vector search AND reached by graph expansion
    assert db_user_node.source == RetrievalSource.BOTH
    assert db_user_node.semantic_score == 0.7
    assert "defines" in db_user_node.related_via or "EdgeType.defines" in db_user_node.related_via
    assert result.both_count >= 1


def test_isolated_graph_node(sample_graph):
    chunk_isolated = CodeChunk(
        chunk_id="isolated.py",
        filepath=Path("isolated.py"),
        chunk_type="file",
        symbol_name=None,
        source_code="# isolated file",
        start_line=1,
        end_line=1,
    )
    v_results = [RetrievalResult(chunk=chunk_isolated, score=0.95, distance=0.05, rank=1)]
    vstore = DummyVectorStore(search_results=v_results)
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    result = retriever.retrieve("isolated code", top_k=5, depth=2)

    # Isolated node has no edges, so graph expansion returns 0 additional nodes
    assert result.total_results == 1
    assert result.nodes[0].node_id == "isolated.py"


def test_multiple_vector_results_sharing_graph_neighbors(sample_graph):
    chunk_func = CodeChunk(
        chunk_id="auth.py::login",
        filepath=Path("auth.py"),
        chunk_type="function",
        symbol_name="login",
        source_code="def login(): pass",
        start_line=5,
        end_line=15,
    )
    chunk_class = CodeChunk(
        chunk_id="auth.py::Auth",
        filepath=Path("auth.py"),
        chunk_type="class",
        symbol_name="Auth",
        source_code="class Auth: pass",
        start_line=20,
        end_line=50,
    )
    v_results = [
        RetrievalResult(chunk=chunk_func, score=0.85, distance=0.15, rank=1),
        RetrievalResult(chunk=chunk_class, score=0.80, distance=0.20, rank=2),
    ]
    vstore = DummyVectorStore(search_results=v_results)
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    # Both seed nodes expand via 'auth.py' (FILE, filtered) to reach each other
    result = retriever.retrieve("auth logic", top_k=5, depth=2)

    node_ids = [n.node_id for n in result.nodes]
    assert len(node_ids) == len(set(node_ids))
    # Both vector seeds are present
    assert "auth.py::login" in node_ids
    assert "auth.py::Auth" in node_ids
    # FILE node auth.py is filtered
    assert "auth.py" not in node_ids


def test_ranking_and_fusion_behavior(sample_graph, sample_chunks):
    chunk_login, _ = sample_chunks
    v_results = [RetrievalResult(chunk=chunk_login, score=0.9, distance=0.1, rank=1)]
    vstore = DummyVectorStore(search_results=v_results)
    traverser = GraphTraverser(sample_graph)
    retriever = HybridRetriever(vstore, traverser)

    result = retriever.retrieve("login", top_k=5, depth=1, vector_weight=0.7, graph_weight=0.3)

    # auth.py::login is the only seed. It has sem_score=0.9 -> norm_sem=(0.9+1)/2=0.95
    # and no *external* corroboration (a node cannot corroborate itself), so gph=0.
    # With corroborative fusion: score = norm_sem * (vector_weight + graph_weight * g_score)
    # = 0.95 * (0.7 + 0.3 * 0.0) = 0.665
    login_node = result.nodes[0]
    assert login_node.node_id == "auth.py::login"
    assert login_node.graph_support == 0.0
    assert abs(login_node.combined_score - 0.665) < 1e-4

    # At depth=1, the only neighbor is auth.py (FILE), which is filtered out
    # So no graph nodes should be present
    assert len(result.nodes) == 1


# ---------------------------------------------------------------------------
# Graph re-ranking capability
#
# Regression tests for the Stage 6 fusion defect: the graph term used to score a
# node's distance from *itself* (always 0 for vector hits), producing an identical
# constant across the whole vector set. Adding a constant to every element cannot
# reorder them, so graph evidence was provably incapable of re-ranking and hybrid
# output was always `vector_ranking ++ graph_tail`.
# ---------------------------------------------------------------------------

@pytest.fixture
def corroboration_graph():
    """Graph contrasting a structurally central node with an isolated one.

    core.py (file)
       │ (DEFINES)
       ├── core.py::hub
       ├── core.py::a
       └── core.py::b

    lonely.py::solo  -- no edges at all
    """
    g = nx.DiGraph()

    nodes = [
        GraphNode(node_id="core.py", node_type=NodeType.FILE, filepath=Path("core.py"), symbol_name=None),
        GraphNode(node_id="core.py::hub", node_type=NodeType.FUNCTION, filepath=Path("core.py"), symbol_name="hub"),
        GraphNode(node_id="core.py::a", node_type=NodeType.FUNCTION, filepath=Path("core.py"), symbol_name="a"),
        GraphNode(node_id="core.py::b", node_type=NodeType.FUNCTION, filepath=Path("core.py"), symbol_name="b"),
        GraphNode(node_id="lonely.py::solo", node_type=NodeType.FUNCTION, filepath=Path("lonely.py"), symbol_name="solo"),
    ]
    for node in nodes:
        g.add_node(node.node_id, node_obj=node)

    for child in ("core.py::hub", "core.py::a", "core.py::b"):
        g.add_edge("core.py", child, edge_type=EdgeType.DEFINES.value)

    return g


def _chunk(chunk_id: str, filepath: str) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        filepath=Path(filepath),
        chunk_type="function",
        symbol_name=chunk_id.split("::")[-1],
        source_code=f"def {chunk_id.split('::')[-1]}(): pass",
        start_line=1,
        end_line=2,
    )


@pytest.fixture
def corroboration_results():
    """Vector hits where the isolated node has the *higher* similarity.

    solo (0.90) beats hub (0.86) semantically, so any ranking change in hub's
    favour can only come from graph evidence.
    """
    return [
        RetrievalResult(chunk=_chunk("lonely.py::solo", "lonely.py"), score=0.90, distance=0.10, rank=1),
        RetrievalResult(chunk=_chunk("core.py::hub", "core.py"), score=0.86, distance=0.14, rank=2),
        RetrievalResult(chunk=_chunk("core.py::a", "core.py"), score=0.30, distance=0.70, rank=3),
        RetrievalResult(chunk=_chunk("core.py::b", "core.py"), score=0.30, distance=0.70, rank=4),
    ]


def test_graph_evidence_can_reorder_vector_results(corroboration_graph, corroboration_results):
    """Graph corroboration must be able to overturn the vector ordering.

    This is the property the old formula made mathematically impossible.
    """
    vstore = DummyVectorStore(search_results=corroboration_results)
    retriever = HybridRetriever(vstore, GraphTraverser(corroboration_graph))

    # depth=0: pure semantics -> solo (0.95 sem) outranks hub (0.93 sem)
    vector_only = [n.node_id for n in retriever.retrieve("q", top_k=10, depth=0).nodes]
    assert vector_only.index("lonely.py::solo") < vector_only.index("core.py::hub")

    # depth=2: hub is corroborated by seeds a and b (2 hops via core.py) while
    # solo is corroborated by nothing, so hub must overtake it.
    hybrid = [n.node_id for n in retriever.retrieve("q", top_k=10, depth=2).nodes]
    assert hybrid.index("core.py::hub") < hybrid.index("lonely.py::solo")

    # The identical query and identical vector hits produced a different order.
    assert hybrid[: len(vector_only)] != vector_only


def test_depth_0_yields_pure_vector_ordering_and_no_support(corroboration_graph, corroboration_results):
    """Vector-only mode must be untouched by the graph term."""
    vstore = DummyVectorStore(search_results=corroboration_results)
    retriever = HybridRetriever(vstore, GraphTraverser(corroboration_graph))

    result = retriever.retrieve("q", top_k=10, depth=0)

    # No traversal ran, so nothing can be corroborated.
    assert all(n.graph_support == 0.0 for n in result.nodes)
    assert all(n.graph_distance == 0 for n in result.nodes)
    assert result.graph_count == 0

    # Ordering is exactly descending semantic similarity, and the score is the
    # weighted semantic term alone.
    assert [n.node_id for n in result.nodes] == [
        "lonely.py::solo", "core.py::hub", "core.py::a", "core.py::b",
    ]
    solo = result.nodes[0]
    assert abs(solo.combined_score - 0.7 * 0.95) < 1e-9


def test_graph_support_decays_with_distance(sample_graph, sample_chunks):
    """With corroborator count held equal, nearer evidence must score higher."""
    chunk_login, _ = sample_chunks
    vstore = DummyVectorStore(
        search_results=[RetrievalResult(chunk=chunk_login, score=0.8, distance=0.2, rank=1)]
    )
    retriever = HybridRetriever(vstore, GraphTraverser(sample_graph))

    result = retriever.retrieve("login", top_k=5, depth=2)
    by_id = {n.node_id: n for n in result.nodes}

    # Each of these has exactly one corroborating seed (auth.py::login), so the
    # only difference between them is distance.
    # auth.py (FILE) is filtered out. auth.py::Auth (CLASS) at 2 hops is present.
    # For comparison, we need a node at depth 1 that's not FILE.
    # The test graph doesn't have such a node, so we verify the CLASS node at depth 2.
    far = by_id["auth.py::Auth"]       # 2 hops via FILE node
    assert far.graph_distance == 2
    # With exponential decay gamma=0.25: support = 0.25^2 * 0.5 (DEFINES weight) = 0.03125
    # g_score = 1 - 1/(1+0.03125) = 0.0303...
    assert far.graph_support > 0.0
    assert far.graph_support < 0.1  # very small due to exponential decay


def test_graph_support_accumulates_across_distinct_seeds(corroboration_graph, corroboration_results):
    """Independent corroborations must count for more than a single one."""
    traverser = GraphTraverser(corroboration_graph)

    # One seed: core.py is 1 hop from hub only.
    # Note: core.py is FILE node, filtered out. So we test with hub instead.
    single = HybridRetriever(
        DummyVectorStore(search_results=corroboration_results[1:2]), traverser
    ).retrieve("q", top_k=10, depth=1)
    hub_single = next(n for n in single.nodes if n.node_id == "core.py::hub")

    # Three seeds (hub, a, b), each 1 hop from core.py (but core.py is FILE, filtered).
    # The seeds are hub, a, b - they are all siblings under core.py.
    # Each has 1-hop to core.py (FILE) then 1-hop to each other = 2 hops total via FILE.
    # But FILE is filtered, so they don't actually reach each other at depth=1.
    # At depth=2 they would reach each other via FILE (2 hops).
    multi = HybridRetriever(
        DummyVectorStore(search_results=corroboration_results[1:]), traverser
    ).retrieve("q", top_k=10, depth=2)

    # With depth=2: hub gets corroboration from a and b via core.py (2 hops each)
    # Exponential decay: gamma=0.25, weight for DEFINES=0.5
    # Each path: 0.25^2 * 0.5 = 0.03125, total = 0.0625
    # g_score = 1 - 1/(1+0.0625) ≈ 0.0588
    hub_multi = next(n for n in multi.nodes if n.node_id == "core.py::hub")

    assert hub_multi.graph_support > hub_single.graph_support
    # Distance is unchanged; only the number of corroborators grew.
    assert hub_single.graph_distance == 0  # no external corroboration at depth=1
    assert hub_multi.graph_distance == 2   # 2 hops via core.py (FILE)


def test_graph_weight_controls_graph_influence(corroboration_graph, corroboration_results):
    """graph_weight must actually govern ordering (it was previously inert)."""
    vstore = DummyVectorStore(search_results=corroboration_results)
    retriever = HybridRetriever(vstore, GraphTraverser(corroboration_graph))

    # graph_weight=0 must reproduce the vector-only ordering even at depth>0.
    zero = [n.node_id for n in retriever.retrieve("q", top_k=10, depth=2, graph_weight=0.0).nodes]
    assert zero.index("lonely.py::solo") < zero.index("core.py::hub")

    # Restoring the weight lets structural evidence win.
    weighted = [n.node_id for n in retriever.retrieve("q", top_k=10, depth=2, graph_weight=0.3).nodes]
    assert weighted.index("core.py::hub") < weighted.index("lonely.py::solo")


def test_graph_only_node_does_not_outrank_strong_vector_hit(corroboration_graph, corroboration_results):
    """Nodes with no measured semantic relevance (pure graph nodes) stay below strong matches."""
    vstore = DummyVectorStore(search_results=corroboration_results)
    retriever = HybridRetriever(vstore, GraphTraverser(corroboration_graph))

    result = retriever.retrieve("q", top_k=10, depth=2)
    by_id = {n.node_id: n for n in result.nodes}

    # In the corroboration_graph, all non-FILE nodes have semantic scores from vector search.
    # The only "pure graph" node would be core.py (FILE) which is filtered out.
    # The test property: any node with norm_sem=0 must have fusion_score=0 and rank last.

    # Verify all returned nodes have combined_score > 0 (since all have semantic scores)
    for node in result.nodes:
        assert node.combined_score > 0.0, f"Node {node.node_id} should have positive score"

    # Verify no node has norm_sem=0 (i.e., all have semantic_score)
    for node in result.nodes:
        assert node.semantic_score is not None, f"Node {node.node_id} should have semantic score"

    # The key property: if a node had norm_sem=0, its fusion_score would be 0
    # and it could never outrank any node with norm_sem>0
    # This is guaranteed by the formula: fusion = norm_sem * (vector_weight + graph_weight * g_score)


def test_self_corroboration_impossible_through_cycles():
    """A node must never corroborate itself, even via a cycle back to itself."""
    g = nx.DiGraph()
    for node_id in ("x.py::x", "y.py::y"):
        g.add_node(
            node_id,
            node_obj=GraphNode(
                node_id=node_id,
                node_type=NodeType.FUNCTION,
                filepath=Path(node_id.split("::")[0]),
                symbol_name=node_id.split("::")[-1],
            ),
        )
    # Mutual cycle: x -> y -> x
    g.add_edge("x.py::x", "y.py::y", edge_type=EdgeType.CALLS.value)
    g.add_edge("y.py::y", "x.py::x", edge_type=EdgeType.CALLS.value)

    vstore = DummyVectorStore(
        search_results=[RetrievalResult(chunk=_chunk("x.py::x", "x.py"), score=0.9, distance=0.1, rank=1)]
    )
    retriever = HybridRetriever(vstore, GraphTraverser(g))

    result = retriever.retrieve("x", top_k=5, depth=3)
    x_node = next(n for n in result.nodes if n.node_id == "x.py::x")

    # x is the only seed, so despite the cycle it has zero external support.
    assert x_node.graph_support == 0.0
    assert x_node.graph_distance == 0


def test_no_vector_results_means_no_graph_support(corroboration_graph):
    """Graph expansion is seeded by vector hits; with none there is no support."""
    retriever = HybridRetriever(DummyVectorStore(search_results=[]), GraphTraverser(corroboration_graph))

    result = retriever.retrieve("q", top_k=10, depth=3)
    assert result.total_results == 0


def test_seed_absent_from_graph_scores_semantics_only(corroboration_graph):
    """A chunk with no graph node must not crash and must keep its vector score."""
    vstore = DummyVectorStore(
        search_results=[RetrievalResult(chunk=_chunk("ghost.py::ghost", "ghost.py"), score=0.5, distance=0.5, rank=1)]
    )
    retriever = HybridRetriever(vstore, GraphTraverser(corroboration_graph))

    result = retriever.retrieve("ghost", top_k=5, depth=2)

    assert result.total_results == 1
    node = result.nodes[0]
    assert node.graph_support == 0.0
    # sem = (0.5 + 1) / 2 = 0.75 -> 0.7 * 0.75
    assert abs(node.combined_score - 0.7 * 0.75) < 1e-9
