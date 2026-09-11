"""Agent state management for the ReAct loop.

This module defines the state that persists across agent iterations,
including conversation history, tool calls, retrieved context, and iteration tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from repomind.agent.llm_provider import Message, ToolCall, ToolResult
from repomind.retrieval.models import HybridResultNode


@dataclass
class AgentState:
    """Complete state of the agent during a ReAct investigation.

    Tracks the original query, conversation history, tool interactions,
    retrieved context, and execution metadata.
    """
    # Original user query
    query: str

    # Conversation history for LLM
    messages: List[Message] = field(default_factory=list)

    # Tool call history
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)

    # Retrieved context nodes (from semantic_search)
    retrieved_nodes: List[HybridResultNode] = field(default_factory=list)

    # Files read via read_file tool
    read_files: Dict[str, str] = field(default_factory=dict)  # filepath -> content

    # Graph nodes discovered via graph_traverse
    graph_nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # node_id -> node data

    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 10

    # Final answer when available
    final_answer: Optional[str] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)
    consecutive_errors: int = 0

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, message: Message) -> None:
        """Add a message to conversation history."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Record a tool call."""
        self.tool_calls.append(tool_call)
        self.updated_at = datetime.now()

    def add_tool_result(self, tool_result: ToolResult) -> None:
        """Record a tool result and update context."""
        self.tool_results.append(tool_result)
        self.updated_at = datetime.now()

        # If successful, extract context
        if not tool_result.is_error:
            self._extract_context_from_result(tool_result)
        else:
            self.consecutive_errors += 1
            self.errors.append(f"Tool '{tool_result.name}' error: {tool_result.content}")
        self.consecutive_errors = 0 if not tool_result.is_error else self.consecutive_errors

    def _extract_context_from_result(self, result: ToolResult) -> None:
        """Extract useful context from a successful tool result."""
        import json

        try:
            data = json.loads(result.content)

            if result.name == "semantic_search":
                # Store retrieved nodes
                for node_data in data.get("results", []):
                    # Convert back to HybridResultNode if needed
                    # For now, store the raw data
                    node_id = node_data.get("node_id")
                    if node_id:
                        self.retrieved_nodes.append(node_data)

            elif result.name == "graph_traverse":
                # Store graph nodes
                for node_data in data.get("nodes", []):
                    node_id = node_data.get("node_id")
                    if node_id:
                        self.graph_nodes[node_id] = node_data

            elif result.name == "read_file":
                # Store file content
                filepath = data.get("filepath")
                content = data.get("content")
                if filepath and content:
                    self.read_files[filepath] = content

        except json.JSONDecodeError:
            # Non-JSON result, ignore for context extraction
            pass

    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.iteration += 1
        self.updated_at = datetime.now()

    def is_complete(self) -> bool:
        """Check if the agent has finished (has final answer)."""
        return self.final_answer is not None

    def should_stop(self) -> bool:
        """Check if the agent should stop (max iterations or complete)."""
        return self.is_complete() or self.iteration >= self.max_iterations

    def set_final_answer(self, answer: str) -> None:
        """Set the final answer and mark complete."""
        self.final_answer = answer
        self.updated_at = datetime.now()

    def get_context_summary(self, max_tokens: int = 8000) -> str:
        """Get a summarized context for the LLM.

        Args:
            max_tokens: Approximate token budget for context

        Returns:
            Formatted context string
        """
        parts = []

        # Add retrieved nodes summary
        if self.retrieved_nodes:
            parts.append("=== Retrieved Code (semantic search) ===")
            for i, node in enumerate(self.retrieved_nodes[:10]):  # Limit to top 10
                if isinstance(node, dict):
                    parts.append(
                        f"{i+1}. {node.get('node_id', 'unknown')} "
                        f"[source: {node.get('source', 'unknown')}, "
                        f"score: {node.get('combined_score', 0):.3f}]"
                    )
                    if node.get('docstring'):
                        parts.append(f"   Docstring: {node['docstring'][:100]}...")
                else:
                    parts.append(f"{i+1}. {node.node_id} [score: {node.combined_score:.3f}]")

        # Add graph nodes summary
        if self.graph_nodes:
            parts.append("\n=== Graph Nodes (structural) ===")
            for i, (node_id, data) in enumerate(list(self.graph_nodes.items())[:10]):
                parts.append(
                    f"{i+1}. {node_id} [{data.get('node_type', 'unknown')}]"
                )
                if data.get('docstring'):
                    parts.append(f"   Docstring: {data['docstring'][:100]}...")

        # Add read files summary
        if self.read_files:
            parts.append("\n=== File Contents Read ===")
            for filepath, content in list(self.read_files.items())[:5]:
                lines = content.splitlines()
                parts.append(f"--- {filepath} (lines 1-{len(lines)}) ---")
                # Truncate long files
                display_content = "\n".join(lines[:50])
                if len(lines) > 50:
                    display_content += f"\n... ({len(lines) - 50} more lines)"
                parts.append(display_content)

        context = "\n".join(parts)

        # Rough token estimation (4 chars ≈ 1 token)
        if len(context) > max_tokens * 4:
            # Truncate
            context = context[:max_tokens * 4] + "\n... [truncated]"

        return context

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for debugging/logging."""
        return {
            "query": self.query,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "is_complete": self.is_complete(),
            "tool_calls_count": len(self.tool_calls),
            "tool_results_count": len(self.tool_results),
            "retrieved_nodes_count": len(self.retrieved_nodes),
            "graph_nodes_count": len(self.graph_nodes),
            "read_files_count": len(self.read_files),
            "errors": self.errors,
            "final_answer": self.final_answer,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }