"""Graph traversal for querying code dependencies and relationships."""

from typing import List, Optional, Set
import networkx as nx

from repomind.graph.models import GraphNode, EdgeType


class GraphTraverser:
    """Traverses the code graph to answer structural queries.

    Supports queries for:
    - Dependencies (downstream): What does this symbol depend on?
    - Dependents (upstream): What depends on this symbol?
    - Neighbors: What is directly connected to this symbol?
    """

    def __init__(self, graph: nx.DiGraph):
        """Initialize traverser with a code graph.

        Args:
            graph: NetworkX directed graph from CodeGraphBuilder
        """
        self.graph = graph

    def get_dependencies(
        self,
        node_id: str,
        depth: Optional[int] = 1,
        edge_types: Optional[List[EdgeType]] = None
    ) -> List[GraphNode]:
        """Get nodes that the given node depends on (downstream).

        Follows edges in their forward direction.

        Args:
            node_id: Starting node ID
            depth: Maximum traversal depth. None = unlimited (transitive closure)
            edge_types: Optional list of edge types to follow. None = all edges

        Returns:
            List of GraphNode objects representing dependencies
        """
        if node_id not in self.graph:
            return []

        if edge_types is not None:
            # If edge_types is specified, sub-graph filter edges before reaching ancestors/descendants
            sub_edges = [
                (u, v) for u, v, d in self.graph.edges(data=True)
                if self._matches_edge_types(d, edge_types)
            ]
            sub_g = self.graph.edge_subgraph(sub_edges)
            if node_id not in sub_g:
                return []
            target_graph = sub_g
        else:
            target_graph = self.graph

        if depth is None:
            # Unlimited depth: get all descendants in target_graph
            descendant_ids = nx.descendants(target_graph, node_id)
            return self._ids_to_nodes(descendant_ids)
        else:
            # Limited depth: BFS traversal
            return self._traverse_bfs(node_id, depth, forward=True, edge_types=edge_types)

    def get_dependents(
        self,
        node_id: str,
        depth: Optional[int] = 1,
        edge_types: Optional[List[EdgeType]] = None
    ) -> List[GraphNode]:
        """Get nodes that depend on the given node (upstream).

        Follows edges in their reverse direction.

        Args:
            node_id: Starting node ID
            depth: Maximum traversal depth. None = unlimited (transitive closure)
            edge_types: Optional list of edge types to follow. None = all edges

        Returns:
            List of GraphNode objects representing dependents
        """
        if node_id not in self.graph:
            return []

        if edge_types is not None:
            # If edge_types is specified, sub-graph filter edges
            sub_edges = [
                (u, v) for u, v, d in self.graph.edges(data=True)
                if self._matches_edge_types(d, edge_types)
            ]
            sub_g = self.graph.edge_subgraph(sub_edges)
            if node_id not in sub_g:
                return []
            target_graph = sub_g
        else:
            target_graph = self.graph

        if depth is None:
            # Unlimited depth: get all ancestors in target_graph
            ancestor_ids = nx.ancestors(target_graph, node_id)
            return self._ids_to_nodes(ancestor_ids)
        else:
            # Limited depth: BFS traversal (reverse direction)
            return self._traverse_bfs(node_id, depth, forward=False, edge_types=edge_types)

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[EdgeType]] = None
    ) -> List[GraphNode]:
        """Get direct neighbors of a node (depth=1, both directions).

        Args:
            node_id: Node ID to query
            edge_types: Optional list of edge types to follow

        Returns:
            List of GraphNode objects representing neighbors
        """
        if node_id not in self.graph:
            return []

        neighbor_ids = set()

        # Outgoing edges (successors)
        for successor in self.graph.successors(node_id):
            edge_data = self.graph[node_id][successor]
            if self._matches_edge_types(edge_data, edge_types):
                neighbor_ids.add(successor)

        # Incoming edges (predecessors)
        for predecessor in self.graph.predecessors(node_id):
            edge_data = self.graph[predecessor][node_id]
            if self._matches_edge_types(edge_data, edge_types):
                neighbor_ids.add(predecessor)

        return self._ids_to_nodes(neighbor_ids)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a single node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            GraphNode object or None if not found
        """
        if node_id not in self.graph:
            return None

        node_data = self.graph.nodes[node_id]
        return node_data.get('node_obj')

    def _traverse_bfs(
        self,
        start_node: str,
        max_depth: int,
        forward: bool,
        edge_types: Optional[List[EdgeType]] = None
    ) -> List[GraphNode]:
        """Perform breadth-first traversal with depth limit.

        Args:
            start_node: Starting node ID
            max_depth: Maximum depth to traverse
            forward: True for forward edges (dependencies), False for reverse (dependents)
            edge_types: Optional edge types to filter

        Returns:
            List of GraphNode objects found during traversal
        """
        visited: Set[str] = set()
        queue: List[tuple[str, int]] = [(start_node, 0)]
        result_ids: Set[str] = set()

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)

            # Don't add the starting node to results
            if current_id != start_node:
                result_ids.add(current_id)

            # Stop if we've reached max depth
            if current_depth >= max_depth:
                continue

            # Get neighbors based on direction
            if forward:
                neighbors = list(self.graph.successors(current_id))
                edge_getter = lambda src, dst: self.graph[src][dst]
            else:
                neighbors = list(self.graph.predecessors(current_id))
                edge_getter = lambda src, dst: self.graph[dst][src]

            # Add unvisited neighbors to queue
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    edge_data = edge_getter(current_id, neighbor_id)
                    if self._matches_edge_types(edge_data, edge_types):
                        queue.append((neighbor_id, current_depth + 1))

        return self._ids_to_nodes(result_ids)

    def _ids_to_nodes(
        self,
        node_ids: Set[str]
    ) -> List[GraphNode]:
        """Convert node IDs to GraphNode objects.

        Args:
            node_ids: Set of node IDs to convert

        Returns:
            List of GraphNode objects
        """
        nodes = []
        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            node_data = self.graph.nodes[node_id]
            node_obj = node_data.get('node_obj')
            if node_obj:
                nodes.append(node_obj)
        return nodes

    def _matches_edge_types(
        self,
        edge_data: dict,
        edge_types: Optional[List[EdgeType]]
    ) -> bool:
        """Check if an edge matches the given edge type filter.

        Args:
            edge_data: Edge data dictionary
            edge_types: List of EdgeType enums to match, or None for any

        Returns:
            True if edge matches filter, False otherwise
        """
        if edge_types is None:
            return True

        edge_type_value = edge_data.get('edge_type')
        if isinstance(edge_type_value, EdgeType):
            return edge_type_value in edge_types
        elif isinstance(edge_type_value, str):
            return any(et.value == edge_type_value for et in edge_types)

        return False
