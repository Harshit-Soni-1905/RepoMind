"""Query service for executing queries against indexed repositories.

This service extracts the query orchestration logic from cli/commands.py
to make it reusable by both the CLI and the new web API (Stage 10).
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from repomind.application.models import QueryResult, ToolExecution
from repomind.config import config
from repomind.vectorstore.embedder import Embedder
from repomind.vectorstore.store import VectorStore
from repomind.graph.builder import CodeGraphBuilder
from repomind.graph.traversal import GraphTraverser
from repomind.retrieval.hybrid import HybridRetriever
from repomind.agent.llm_provider import GeminiProvider, FreeLocalProvider, LLMProvider
from repomind.agent.tools import create_default_registry
from repomind.agent.agent import ReActAgent


class QueryService:
    """Service for executing queries against indexed repositories."""

    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        default_provider: str = "gemini",
        default_model: str = "gemini-3.5-flash-lite",
    ):
        """Initialize the query service.

        Args:
            vector_store_path: Path for ChromaDB vector store
            default_provider: Default LLM provider name
            default_model: Default LLM model name
        """
        self.vector_store_path = vector_store_path or Path(config.VECTOR_STORE_PATH)
        self.default_provider = default_provider
        self.default_model = default_model

    def execute_query(
        self,
        repo_path: Path,
        repo_id: str,
        query: str,
        graph: Optional[Any] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
        max_iterations: Optional[int] = None,
        stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> QueryResult:
        """Execute a query against an indexed repository.

        Args:
            repo_path: Path to the repository
            repo_id: Repository identifier
            query: User's question
            graph: Pre-loaded graph (optional)
            provider_name: LLM provider override
            model_name: LLM model override
            top_k: Semantic search result count override
            depth: Graph traversal depth override
            max_iterations: Max agent iterations override
            stream_callback: Optional callback for streaming events

        Returns:
            QueryResult with answer and tool executions
        """
        try:
            # Configuration
            active_provider = provider_name or self.default_provider
            active_model = model_name or self.default_model
            active_top_k = top_k if top_k is not None else config.VECTOR_SEARCH_TOP_K
            active_depth = depth if depth is not None else config.GRAPH_TRAVERSAL_DEPTH
            active_max_iter = max_iterations if max_iterations is not None else config.AGENT_MAX_ITERATIONS

            # Setup retrieval components with repo-specific collection
            collection_name = f"repomind_{repo_id}"
            embedder = Embedder()
            store = VectorStore(
                collection_name=collection_name,
                persist_dir=str(self.vector_store_path),
                embedder=embedder,
            )

            # Verify the repository is indexed
            if store.count() == 0:
                return QueryResult(
                    answer="",
                    tool_executions=[],
                    success=False,
                    error="Repository is not indexed",
                )

            # Load graph or use provided graph
            if graph is None:
                graph_builder = CodeGraphBuilder()
                # Empty graph as fallback (vector-only mode)
                actual_graph = graph_builder.graph
            else:
                actual_graph = graph

            traverser = GraphTraverser(actual_graph)
            retriever = HybridRetriever(
                vector_store=store,
                graph_traverser=traverser,
                default_top_k=active_top_k,
                default_depth=active_depth,
            )

            # Instantiate LLM provider
            provider: LLMProvider
            if active_provider == "gemini":
                provider = GeminiProvider(
                    api_key=config.GEMINI_API_KEY,
                    model_name=active_model
                )
                if not provider.is_available():
                    provider = FreeLocalProvider()
            else:
                provider = FreeLocalProvider()

            # Build agent
            registry = create_default_registry(retriever, traverser, repo_root=repo_path)
            agent = ReActAgent(
                llm_provider=provider,
                tool_registry=registry,
                max_iterations=active_max_iter,
            )

            # Execute query with optional streaming
            if stream_callback:
                # Stream tool start events
                def tool_callback(tool_name: str, arguments: Dict[str, Any]):
                    stream_callback({
                        "type": "tool_start",
                        "tool": tool_name,
                        "arguments": arguments,
                    })

                # Note: Full streaming would require modifying the agent
                # For Stage 10, we'll stream structured events instead
                pass

            # Run agent
            state = agent.analyze(query)

            # Extract tool executions
            tool_executions = []
            for i, (tc, tr) in enumerate(zip(state.tool_calls, state.tool_results)):
                tool_exec = ToolExecution(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    result_summary=tr.content[:200] if tr.content else "",
                    execution_order=i,
                )
                tool_executions.append(tool_exec)

                # Stream tool result event
                if stream_callback:
                    stream_callback({
                        "type": "tool_result",
                        "tool": tc.name,
                        "summary": tool_exec.result_summary,
                        "order": i,
                    })

            # Stream answer
            if stream_callback and state.final_answer:
                stream_callback({
                    "type": "answer",
                    "text": state.final_answer,
                })

            if stream_callback:
                stream_callback({"type": "done"})

            return QueryResult(
                answer=state.final_answer or "",
                tool_executions=tool_executions,
                success=state.final_answer is not None,
                error=None if state.final_answer else "Agent failed to produce answer",
            )

        except Exception as e:
            if stream_callback:
                stream_callback({
                    "type": "error",
                    "message": str(e),
                })
            return QueryResult(
                answer="",
                tool_executions=[],
                success=False,
                error=str(e),
            )
