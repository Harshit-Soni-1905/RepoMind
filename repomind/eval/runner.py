"""Evaluation runner for retrieval and agent systems.

This module orchestrates executing retrieval queries (vector-only vs. hybrid)
and agent runs against evaluation datasets, collecting timing, metrics, and raw logs.
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass, field, asdict

from repomind.eval.dataset import EvaluationQuestion, RetrievalGroundTruth, AnswerGroundTruth
from repomind.eval.metrics import compute_all_metrics
from repomind.retrieval.hybrid import HybridRetriever
from repomind.vectorstore.store import VectorStore
from repomind.agent.agent import ReActAgent
from repomind.agent.llm_provider import LLMProvider, FreeLocalProvider
from repomind.agent.tools import create_default_registry


@dataclass
class QueryResultRecord:
    """Raw evaluation result record for a single question and retrieval system."""
    question_id: str
    category: str
    difficulty: str
    system: str  # "vector_only" or "hybrid"
    top_k: int
    depth: int
    retrieved_ids: List[str]
    relevant_ids: List[str]
    latency_ms: float
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSONL serialization."""
        d = asdict(self)
        # Flatten metrics into top-level keys for easy tabular analysis if needed,
        # but keep nested structure for raw results manifest.
        return d


@dataclass
class AgentEvalRecord:
    """Raw evaluation record for an agent run on a question."""
    question_id: str
    category: str
    difficulty: str
    provider_name: str
    success: bool
    iterations: int
    total_tool_calls: int
    tool_counts: Dict[str, int]
    used_graph_tool: bool
    redundant_tool_calls: int
    final_answer: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrievalRunner:
    """Runner for evaluating retrieval performance (vector-only vs. hybrid)."""

    def __init__(self, hybrid_retriever: HybridRetriever):
        """Initialize runner with a hybrid retriever instance.

        Args:
            hybrid_retriever: Pre-configured HybridRetriever (wrapping VectorStore & Graph)
        """
        self.retriever = hybrid_retriever

    def evaluate_question(
        self,
        question: EvaluationQuestion,
        system: str = "hybrid",
        top_k: int = 10,
        depth: int = 2,
        k_values: Optional[List[int]] = None
    ) -> QueryResultRecord:
        """Evaluate a single question under a given retrieval configuration.

        Args:
            question: Evaluation question object
            system: Retrieval mode ("vector_only" or "hybrid")
            top_k: Number of top results to retrieve
            depth: Graph traversal depth (ignored if system == "vector_only")
            k_values: List of k thresholds for metric computation

        Returns:
            QueryResultRecord containing metrics and latency
        """
        ground_truth = question.retrieval
        relevant_ids = ground_truth.relevant_chunk_ids

        # Set effective depth
        effective_depth = 0 if system == "vector_only" else depth

        start_time = time.perf_counter()
        result = self.retriever.retrieve(
            query=question.question,
            top_k=top_k,
            depth=effective_depth
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract retrieved chunk / node IDs
        retrieved_ids = [node.node_id for node in result.nodes]

        # Calculate metrics
        metrics = compute_all_metrics(
            retrieved=retrieved_ids,
            relevant=relevant_ids,
            k_values=k_values or [5, 10, 20]
        )

        return QueryResultRecord(
            question_id=question.question_id,
            category=question.category,
            difficulty=question.difficulty,
            system=system,
            top_k=top_k,
            depth=effective_depth,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            latency_ms=round(elapsed_ms, 2),
            metrics=metrics,
        )

    def evaluate_dataset(
        self,
        questions: List[EvaluationQuestion],
        systems: Optional[List[str]] = None,
        top_k: int = 10,
        depth: int = 2,
    ) -> List[QueryResultRecord]:
        """Evaluate an entire dataset across configured retrieval systems.

        Args:
            questions: List of EvaluationQuestion objects
            systems: Systems to evaluate, defaults to ["vector_only", "hybrid"]
            top_k: Cutoff rank for vector search
            depth: Graph depth for hybrid search

        Returns:
            List of QueryResultRecord objects
        """
        if systems is None:
            systems = ["vector_only", "hybrid"]

        records = []
        for q in questions:
            for sys_name in systems:
                record = self.evaluate_question(
                    question=q,
                    system=sys_name,
                    top_k=top_k,
                    depth=depth
                )
                records.append(record)

        return records


class AgentRunner:
    """Runner for evaluating agent performance on evaluation datasets."""

    def __init__(self, vector_store: VectorStore, hybrid_retriever: HybridRetriever):
        """Initialize agent runner.

        Args:
            vector_store: VectorStore for tool execution
            hybrid_retriever: HybridRetriever for tool execution
        """
        self.vector_store = vector_store
        self.retriever = hybrid_retriever

    def evaluate_question(
        self,
        question: EvaluationQuestion,
        provider: Optional[LLMProvider] = None,
        max_iterations: int = 5,
    ) -> AgentEvalRecord:
        """Run agent on a single evaluation question and log operational metrics.

        Args:
            question: Evaluation question object
            provider: LLMProvider to use (defaults to FreeLocalProvider)
            max_iterations: Max loop iterations

        Returns:
            AgentEvalRecord with execution stats
        """
        if provider is None:
            provider = FreeLocalProvider()

        registry = create_default_registry(self.vector_store, self.retriever)
        agent = ReActAgent(
            provider=provider,
            tool_registry=registry,
            max_iterations=max_iterations
        )

        start_time = time.perf_counter()
        state = agent.run(question.question)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Analyze agent execution state using tool_calls (actual AgentState structure)
        tool_counts: Dict[str, int] = {}
        tool_call_history: List[tuple[str, str]] = []
        redundant_calls = 0

        for tool_call in state.tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.arguments
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            call_sig = (tool_name, str(tool_args))
            if call_sig in tool_call_history:
                redundant_calls += 1
            tool_call_history.append(call_sig)

        used_graph = tool_counts.get("graph_traverse", 0) > 0
        success = state.final_answer is not None and len(state.errors) == 0

        return AgentEvalRecord(
            question_id=question.question_id,
            category=question.category,
            difficulty=question.difficulty,
            provider_name=provider.__class__.__name__,
            success=success,
            iterations=state.current_iteration,
            total_tool_calls=len(tool_call_history),
            tool_counts=tool_counts,
            used_graph_tool=used_graph,
            redundant_tool_calls=redundant_calls,
            final_answer=state.final_answer or "",
            latency_ms=round(elapsed_ms, 2),
        )

    def evaluate_dataset(
        self,
        questions: List[EvaluationQuestion],
        provider: Optional[LLMProvider] = None,
        max_iterations: int = 5,
    ) -> List[AgentEvalRecord]:
        """Evaluate agent across a dataset.

        Args:
            questions: List of evaluation questions
            provider: LLMProvider instance
            max_iterations: Max iterations per query

        Returns:
            List of AgentEvalRecord objects
        """
        records = []
        for q in questions:
            rec = self.evaluate_question(q, provider=provider, max_iterations=max_iterations)
            records.append(rec)
        return records