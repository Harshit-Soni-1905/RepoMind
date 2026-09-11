"""Tool definitions and implementations for the RepoMind agent.

This module defines the tool registry and implements the three core tools:
- semantic_search: Search repository code using vector similarity
- graph_traverse: Explore structural relationships in the code graph
- read_file: Read actual source code from the repository
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import json

from repomind.agent.llm_provider import ToolDefinition, ToolCall, ToolResult
from repomind.retrieval.hybrid import HybridRetriever
from repomind.graph.traversal import GraphTraverser
from repomind.graph.models import GraphNode, EdgeType
from repomind.ingestion.file_reader import FileReader, SourceFile
from repomind.vectorstore.models import RetrievalResult


@dataclass(frozen=True)
class Tool:
    """Represents a registered tool with its metadata and executor."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    executor: Callable[[Dict[str, Any]], ToolResult]

    def to_definition(self) -> ToolDefinition:
        """Convert to LLM-facing ToolDefinition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


class ToolRegistry:
    """Registry for managing available tools.

    Provides tool registration, lookup, and execution with validation.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions for LLM consumption."""
        return [tool.to_definition() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call with argument validation.

        Args:
            tool_call: ToolCall with name and arguments

        Returns:
            ToolResult with execution outcome
        """
        tool = self._tools.get(tool_call.name)
        if not tool:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Unknown tool '{tool_call.name}'",
                is_error=True,
            )

        # Validate arguments against schema (basic validation)
        try:
            # Basic required field check
            required = tool.input_schema.get("required", [])
            for req_field in required:
                if req_field not in tool_call.arguments:
                    return ToolResult(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        content=f"Error: Missing required argument '{req_field}'",
                        is_error=True,
                    )

            # Execute the tool
            result = tool.executor(tool_call.arguments)
            return result

        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error executing tool: {str(e)}",
                is_error=True,
            )


# ============================================================
# Tool Implementations
# ============================================================

def create_semantic_search_tool(retriever: HybridRetriever) -> Tool:
    """Create the semantic_search tool.

    Args:
        retriever: HybridRetriever instance for performing searches

    Returns:
        Tool instance
    """
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search for in the codebase"
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "minimum": 1,
                "maximum": 20,
                "default": 5
            },
            "depth": {
                "type": "integer",
                "description": "Graph traversal depth for hybrid retrieval (default: 1, 0 = vector only)",
                "minimum": 0,
                "maximum": 3,
                "default": 1
            }
        },
        "required": ["query"]
    }

    def executor(args: Dict[str, Any]) -> ToolResult:
        query = args["query"]
        top_k = int(args["top_k"]) if args.get("top_k") is not None else 5
        depth = int(args["depth"]) if args.get("depth") is not None else 1

        result = retriever.retrieve(query=query, top_k=top_k, depth=depth)

        # Format results for the agent
        formatted_results = []
        for node in result.nodes:
            formatted_results.append({
                "node_id": node.node_id,
                "filepath": str(node.filepath),
                "symbol_name": node.symbol_name,
                "symbol_type": node.symbol_type,
                "source": node.source.value,
                "semantic_score": node.semantic_score,
                "graph_distance": node.graph_distance,
                "related_via": node.related_via,
                "combined_score": node.combined_score,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "docstring": node.docstring,
            })

        content = {
            "query": query,
            "total_results": result.total_results,
            "vector_count": result.vector_count,
            "graph_count": result.graph_count,
            "both_count": result.both_count,
            "results": formatted_results,
        }

        return ToolResult(
            call_id="",  # Will be set by registry
            name="semantic_search",
            content=json.dumps(content, indent=2),
        )

    return Tool(
        name="semantic_search",
        description="Search the codebase using semantic similarity. Finds code relevant to a natural language query by combining vector search with optional graph traversal to discover related code structures.",
        input_schema=schema,
        executor=executor,
    )


def create_graph_traverse_tool(traverser: GraphTraverser) -> Tool:
    """Create the graph_traverse tool.

    Args:
        traverser: GraphTraverser instance for graph queries

    Returns:
        Tool instance
    """
    schema = {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Starting node identifier (format: 'path/to/file.py::symbol_name')"
            },
            "depth": {
                "type": "integer",
                "description": "Traversal depth (default: 1, None = unlimited)",
                "minimum": 0,
                "maximum": 5,
                "default": 1
            },
            "direction": {
                "type": "string",
                "description": "Traversal direction: 'dependencies' (downstream), 'dependents' (upstream), or 'neighbors' (both, depth=1)",
                "enum": ["dependencies", "dependents", "neighbors"],
                "default": "neighbors"
            },
            "edge_types": {
                "type": "array",
                "description": "Optional list of edge types to filter by",
                "items": {
                    "type": "string",
                    "enum": ["imports", "defines", "contains", "calls"]
                },
                "default": []
            }
        },
        "required": ["node_id"]
    }

    def executor(args: Dict[str, Any]) -> ToolResult:
        node_id = args["node_id"]
        depth = int(args["depth"]) if args.get("depth") is not None else 1
        direction = args.get("direction", "neighbors")
        edge_types = args.get("edge_types", [])

        # Convert edge type strings to EdgeType enums
        edge_type_enums = None
        if edge_types:
            edge_type_enums = [EdgeType(et) for et in edge_types]

        if direction == "dependencies":
            nodes = traverser.get_dependencies(node_id, depth=depth, edge_types=edge_type_enums)
        elif direction == "dependents":
            nodes = traverser.get_dependents(node_id, depth=depth, edge_types=edge_type_enums)
        else:  # neighbors
            nodes = traverser.get_neighbors(node_id, edge_types=edge_type_enums)

        formatted_nodes = []
        for node in nodes:
            formatted_nodes.append({
                "node_id": node.node_id,
                "filepath": str(node.filepath),
                "symbol_name": node.symbol_name,
                "node_type": node.node_type.value,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "docstring": node.docstring,
            })

        content = {
            "start_node": node_id,
            "direction": direction,
            "depth": depth,
            "total_found": len(formatted_nodes),
            "nodes": formatted_nodes,
        }

        return ToolResult(
            call_id="",
            name="graph_traverse",
            content=json.dumps(content, indent=2),
        )

    return Tool(
        name="graph_traverse",
        description="Explore structural relationships in the code graph. Starting from a known symbol, traverse dependencies (what it imports/calls), dependents (what imports/calls it), or direct neighbors. Useful for understanding code structure and dependencies.",
        input_schema=schema,
        executor=executor,
    )


def create_read_file_tool(repo_root: Optional[Path] = None) -> Tool:
    """Create the read_file tool.

    Args:
        repo_root: Optional repository root for relative path resolution

    Returns:
        Tool instance
    """
    reader = FileReader(repo_root=repo_root)

    schema = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Path to the file to read (relative to repo root or absolute)"
            },
            "start_line": {
                "type": "integer",
                "description": "Optional starting line number (1-indexed)",
                "minimum": 1
            },
            "end_line": {
                "type": "integer",
                "description": "Optional ending line number (inclusive, 1-indexed)",
                "minimum": 1
            }
        },
        "required": ["filepath"]
    }

    def executor(args: Dict[str, Any]) -> ToolResult:
        filepath = args["filepath"]
        start_line = int(args["start_line"]) if args.get("start_line") is not None else None
        end_line = int(args["end_line"]) if args.get("end_line") is not None else None

        try:
            source_file = reader.read(Path(filepath))

            content = source_file.content
            lines = content.splitlines()

            # Apply line range filter if specified
            if start_line is not None or end_line is not None:
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                selected_lines = lines[start_idx:end_idx]
                content = "\n".join(selected_lines)
                actual_start = start_idx + 1
                actual_end = min(end_idx, len(lines))
            else:
                actual_start = 1
                actual_end = len(lines)

            result = {
                "filepath": str(source_file.relative_path),
                "absolute_path": str(source_file.absolute_path),
                "content": content,
                "start_line": actual_start,
                "end_line": actual_end,
                "total_lines": source_file.line_count,
                "size_bytes": source_file.size_bytes,
            }

            return ToolResult(
                call_id="",
                name="read_file",
                content=json.dumps(result, indent=2),
            )

        except FileNotFoundError as e:
            return ToolResult(
                call_id="",
                name="read_file",
                content=f"Error: File not found: {filepath}",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(
                call_id="",
                name="read_file",
                content=f"Error reading file: {str(e)}",
                is_error=True,
            )

    return Tool(
        name="read_file",
        description="Read the actual source code of a file from the repository. Can optionally read a specific line range. Use this to examine implementation details after finding relevant files through search or graph traversal.",
        input_schema=schema,
        executor=executor,
    )


def create_default_registry(
    retriever: HybridRetriever,
    traverser: GraphTraverser,
    repo_root: Optional[Path] = None
) -> ToolRegistry:
    """Create a tool registry with all three default tools.

    Args:
        retriever: HybridRetriever for semantic_search
        traverser: GraphTraverser for graph_traverse
        repo_root: Repository root for read_file

    Returns:
        Configured ToolRegistry with all three tools
    """
    registry = ToolRegistry()
    registry.register(create_semantic_search_tool(retriever))
    registry.register(create_graph_traverse_tool(traverser))
    registry.register(create_read_file_tool(repo_root))
    return registry