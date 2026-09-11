"""Hybrid retrieval component combining vector search with graph traversal."""

from typing import List, Optional, Dict, Set, Any
from pathlib import Path

from repomind.vectorstore.store import VectorStore
from repomind.graph.traversal import GraphTraverser
from repomind.graph.models import GraphNode, NodeType
from repomind.retrieval.models import (
    RetrievalSource,
    HybridResultNode,
    HybridQueryResult,
)


class HybridRetriever:
    """Orchestrates semantic vector search and structural graph traversal.

    Combines top-k semantic search results from VectorStore with structural
    neighbors discovered by GraphTraverser up to a specified depth limit.
    Deduplicates results and computes a deterministic fusion score for ranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_traverser: GraphTraverser,
        default_top_k: int = 5,
        default_depth: int = 1,
    ):
        """Initialize hybrid retriever with underlying storage/traversal components.

        Args:
            vector_store: Stage 4 VectorStore instance
            graph_traverser: Stage 5 GraphTraverser instance
            default_top_k: Default number of vector results to retrieve
            default_depth: Default graph traversal depth
        """
        self.vector_store = vector_store
        self.graph_traverser = graph_traverser
        self.default_top_k = default_top_k
        self.default_depth = default_depth

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
    ) -> HybridQueryResult:
        """Perform unified hybrid retrieval given a natural language query.

        Ranking fuses two components per candidate node:

            score = vector_weight * sem + graph_weight * gph

        where ``sem`` is the vector similarity mapped to [0, 1] (0 for nodes found
        only via the graph), and ``gph`` is distance-decayed structural
        corroboration from *other* vector seeds::

            support = sum over corroborating seeds s of 1 / (1 + distance(s, n))
            gph     = 1 - 1 / (1 + support)

        So a node referenced by several nearby seeds outranks an equally-similar
        but structurally isolated one, and ``gph`` is 0 for every node when
        depth=0 — making depth=0 exactly equivalent to vector-only ranking.

        Args:
            query: User's query string
            top_k: Maximum vector search results to retrieve (defaults to default_top_k)
            depth: Graph expansion depth (defaults to default_depth, 0 = vector only)
            vector_weight: Weight assigned to normalized vector similarity score [0.0-1.0]
            graph_weight: Weight assigned to graph corroboration score [0.0-1.0]

        Returns:
            HybridQueryResult containing deduplicated, ranked HybridResultNodes
        """
        k = top_k if top_k is not None else self.default_top_k
        d = depth if depth is not None else self.default_depth

        if not query or not query.strip():
            return HybridQueryResult(
                query=query,
                nodes=[],
                total_results=0,
                vector_count=0,
                graph_count=0,
                both_count=0,
                top_k=k,
                traversal_depth=d,
            )

        # 1. Execute vector search
        vector_results = self.vector_store.search(query=query, top_k=k)

        # Build intermediate map of result entries keyed by node_id
        # Structure: node_id -> dict storing accumulated vector and graph metadata
        candidates: Dict[str, Dict[str, Any]] = {}

        for vr in vector_results:
            chunk = vr.chunk
            node_id = chunk.chunk_id
            candidates[node_id] = {
                "node_id": node_id,
                "filepath": chunk.filepath,
                "symbol_name": chunk.symbol_name,
                "symbol_type": chunk.chunk_type,
                "source_code": chunk.source_code,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "docstring": chunk.docstring,
                "vector_found": True,
                "graph_found": False,
                "semantic_score": vr.score,
                "semantic_distance": vr.distance,
                # seed_id -> shortest distance from that seed. Populated only by
                # graph expansion, so a vector hit nobody else references stays empty.
                "support": {},
                "related_via": set(),
            }

        # 2. Execute graph expansion if depth > 0 and vector seed nodes exist
        if d > 0 and candidates:
            # Seed nodes from vector results
            seed_ids = list(candidates.keys())

            for seed_id in seed_ids:
                expanded_nodes = self._expand_graph_from_seed(seed_id, max_depth=d)

                for expanded_node, dist, rel_type in expanded_nodes:
                    exp_id = expanded_node.node_id

                    # A node must never corroborate itself
                    if exp_id == seed_id:
                        continue

                    # Filter out container FILE nodes from becoming candidate result chunks.
                    # FILE nodes represent module containers, not code chunks with source code.
                    if expanded_node.node_type == NodeType.FILE or expanded_node.node_type.value == "file":
                        continue

                    if exp_id in candidates:
                        # Item was already found by vector search or previous expansion
                        cand = candidates[exp_id]
                        cand["graph_found"] = True
                        if rel_type:
                            cand["related_via"].add(rel_type)
                    else:
                        # New node discovered exclusively via graph expansion
                        candidates[exp_id] = {
                            "node_id": exp_id,
                            "filepath": expanded_node.filepath,
                            "symbol_name": expanded_node.symbol_name,
                            "symbol_type": expanded_node.node_type.value,
                            "source_code": None,  # Graph node metadata doesn't embed full source by default
                            "start_line": expanded_node.start_line,
                            "end_line": expanded_node.end_line,
                            "docstring": expanded_node.docstring,
                            "vector_found": False,
                            "graph_found": True,
                            "semantic_score": None,
                            "semantic_distance": None,
                            "support": {},
                            "related_via": {rel_type} if rel_type else set(),
                        }
                        cand = candidates[exp_id]

                    # Store (distance, relation_type) per seed for fine-grained support calculation
                    prev = cand["support"].get(seed_id)
                    if prev is None or dist < prev[0]:
                        cand["support"][seed_id] = (dist, rel_type)

        # 3. Process candidate nodes: determine provenance, compute fusion score, sort
        result_nodes: List[HybridResultNode] = []
        vec_cnt = 0
        graph_cnt = 0
        both_cnt = 0

        # Constants for graph support calculation
        # Exponential decay gamma: 1-hop=0.25, 2-hop=0.0625, 3-hop=0.015625
        DECAY_GAMMA = 0.25

        # Edge-type relation weights
        # CALLS edges get highest weight as they represent direct function/method invocations
        RELATION_WEIGHTS = {
            "calls": 1.2,
            "EdgeType.CALLS": 1.2,
            "imports": 1.0,
            "EdgeType.IMPORTS": 1.0,
            "defines": 0.5,
            "EdgeType.DEFINES": 0.5,
            "contains": 0.5,
            "EdgeType.CONTAINS": 0.5,
        }

        for cand in candidates.values():
            vec_f = cand["vector_found"]
            graph_f = cand["graph_found"]

            if vec_f and graph_f:
                source = RetrievalSource.BOTH
                both_cnt += 1
            elif vec_f:
                source = RetrievalSource.VECTOR
                vec_cnt += 1
            else:
                source = RetrievalSource.GRAPH
                graph_cnt += 1

            # Compute normalized semantic component [0.0, 1.0]
            sem_score = cand["semantic_score"]
            if sem_score is not None:
                # Clamp similarity score [-1, 1] to [0, 1] for fusion
                norm_sem = max(0.0, (sem_score + 1.0) / 2.0)
            else:
                norm_sem = 0.0

            # Compute distance-decayed, edge-weighted structural corroboration.
            # Near relationships (1-hop imports) contribute strongly (0.25),
            # while multi-hop structural noise decays exponentially (0.0625 at 2-hop).
            support = cand["support"]
            support_total = 0.0
            min_dist = None

            for seed_id, supp_info in support.items():
                if isinstance(supp_info, tuple):
                    dist, rel_type = supp_info
                else:
                    dist = supp_info
                    rel_type = ""

                rel_weight = RELATION_WEIGHTS.get(str(rel_type), 0.5)
                decayed_val = (DECAY_GAMMA ** float(dist)) * rel_weight
                support_total += decayed_val

                if min_dist is None or dist < min_dist:
                    min_dist = dist

            # Monotonically squash unbounded support total into [0.0, 1.0)
            g_score = 1.0 - 1.0 / (1.0 + support_total)

            # Nearest external corroboration distance (0 if not reached from external seed)
            g_dist = min_dist if min_dist is not None else 0

            # Corroborative Score Fusion:
            # Graph evidence modulates semantic relevance as a corroborating factor
            # rather than acting as an independent additive term.
            #
            # fusion_score = norm_sem * (vector_weight + graph_weight * g_score)
            #
            # This guarantees:
            # 1. Weak graph evidence cannot overpower a strong vector result.
            # 2. Strong vector results with graph corroboration get a proportional boost.
            # 3. Pure graph candidates (norm_sem = 0.0) receive 0.0 and never displace vector hits.
            fusion_score = norm_sem * (vector_weight + graph_weight * g_score)

            node = HybridResultNode(
                node_id=cand["node_id"],
                filepath=cand["filepath"],
                symbol_name=cand["symbol_name"],
                symbol_type=cand["symbol_type"],
                source_code=cand["source_code"],
                start_line=cand["start_line"],
                end_line=cand["end_line"],
                docstring=cand["docstring"],
                source=source,
                semantic_score=sem_score,
                semantic_distance=cand["semantic_distance"],
                graph_distance=g_dist,
                graph_support=g_score,
                related_via=sorted(list(cand["related_via"])),
                combined_score=fusion_score,
            )
            result_nodes.append(node)

        # Sort results deterministically: by combined_score descending, then node_id ascending
        result_nodes.sort(key=lambda n: (-n.combined_score, n.node_id))

        return HybridQueryResult(
            query=query,
            nodes=result_nodes,
            total_results=len(result_nodes),
            vector_count=vec_cnt,
            graph_count=graph_cnt,
            both_count=both_cnt,
            top_k=k,
            traversal_depth=d,
        )

    def _expand_graph_from_seed(
        self,
        seed_id: str,
        max_depth: int
    ) -> List[tuple[GraphNode, int, str]]:
        """Perform BFS from seed node on graph, returning nodes with depth and edge type.

        Args:
            seed_id: Node ID to start expansion from
            max_depth: Max depth limit

        Returns:
            List of (GraphNode, distance_int, edge_type_str) tuples
        """
        if seed_id not in self.graph_traverser.graph:
            return []

        results: List[tuple[GraphNode, int, str]] = []
        visited: Set[str] = {seed_id}
        queue: List[tuple[str, int]] = [(seed_id, 0)]

        g = self.graph_traverser.graph

        while queue:
            curr_id, curr_dist = queue.pop(0)

            if curr_dist >= max_depth:
                continue

            # Check outgoing edges (successors)
            for successor in g.successors(curr_id):
                edge_data = g[curr_id][successor]
                edge_type = str(edge_data.get("edge_type", ""))
                
                if successor not in visited:
                    visited.add(successor)
                    node_obj = g.nodes[successor].get("node_obj")
                    if node_obj:
                        results.append((node_obj, curr_dist + 1, edge_type))
                    queue.append((successor, curr_dist + 1))

            # Check incoming edges (predecessors)
            for predecessor in g.predecessors(curr_id):
                edge_data = g[predecessor][curr_id]
                edge_type = str(edge_data.get("edge_type", ""))
                
                if predecessor not in visited:
                    visited.add(predecessor)
                    node_obj = g.nodes[predecessor].get("node_obj")
                    if node_obj:
                        results.append((node_obj, curr_dist + 1, edge_type))
                    queue.append((predecessor, curr_dist + 1))

        return results
