"""Tests for the evaluation runner module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from repomind.eval.runner import RetrievalRunner, AgentRunner, QueryResultRecord, AgentEvalRecord
from repomind.eval.dataset import EvaluationQuestion, RetrievalGroundTruth
from repomind.retrieval.hybrid import HybridRetriever
from repomind.retrieval.models import HybridQueryResult, HybridResultNode, RetrievalSource
from repomind.vectorstore.store import VectorStore
from repomind.agent.state import AgentState
from repomind.agent.llm_provider import ToolCall
from repomind.agent.llm_provider import FreeLocalProvider
from repomind.agent.tools import ToolRegistry


def create_mock_hybrid_retriever():
    """Create a mock HybridRetriever for testing."""
    mock_retriever = Mock(spec=HybridRetriever)

    # Create mock result nodes
    node1 = HybridResultNode(
        node_id="a.py::foo",
        filepath="a.py",
        symbol_name="foo",
        symbol_type="function",
        source_code="def foo(): pass",
        start_line=1,
        end_line=3,
        docstring="foo function",
        source=RetrievalSource.VECTOR,
        semantic_score=0.9,
        semantic_distance=0.1,
        graph_distance=0,
        related_via=[],
        combined_score=0.8,
    )
    node2 = HybridResultNode(
        node_id="b.py::bar",
        filepath="b.py",
        symbol_name="bar",
        symbol_type="function",
        source_code="def bar(): foo()",
        start_line=1,
        end_line=3,
        docstring="bar function",
        source=RetrievalSource.BOTH,
        semantic_score=0.8,
        semantic_distance=0.2,
        graph_distance=1,
        related_via=["calls"],
        combined_score=0.75,
    )
    node3 = HybridResultNode(
        node_id="c.py::baz",
        filepath="c.py",
        symbol_name="baz",
        symbol_type="function",
        source_code="def baz(): pass",
        start_line=1,
        end_line=3,
        docstring="baz function",
        source=RetrievalSource.GRAPH,
        semantic_score=None,
        semantic_distance=None,
        graph_distance=2,
        related_via=["imports"],
        combined_score=0.5,
    )

    mock_result = HybridQueryResult(
        query="test query",
        nodes=[node1, node2, node3],
        total_results=3,
        vector_count=1,
        graph_count=1,
        both_count=1,
        top_k=10,
        traversal_depth=2,
    )

    mock_retriever.retrieve.return_value = mock_result
    return mock_retriever


def create_test_questions():
    """Create test evaluation questions."""
    gt1 = RetrievalGroundTruth(
        question_id="q1",
        question="What is foo?",
        relevant_chunk_ids=["a.py::foo", "b.py::bar"],
        relevant_filepaths=["a.py", "b.py"],
        category="symbol_lookup",
        difficulty="easy",
    )
    gt2 = RetrievalGroundTruth(
        question_id="q2",
        question="What calls foo?",
        relevant_chunk_ids=["b.py::bar"],
        relevant_filepaths=["b.py"],
        category="call_chain",
        difficulty="medium",
    )
    gt3 = RetrievalGroundTruth(
        question_id="q3",
        question="Is there a qux?",
        relevant_chunk_ids=[],
        relevant_filepaths=[],
        category="negative",
        difficulty="easy",
    )

    return [
        EvaluationQuestion(retrieval=gt1),
        EvaluationQuestion(retrieval=gt2),
        EvaluationQuestion(retrieval=gt3),
    ]


class TestRetrievalRunner:
    """Tests for RetrievalRunner."""

    def test_evaluate_question_vector_only(self):
        """Test evaluating a question with vector-only system."""
        mock_retriever = create_mock_hybrid_retriever()
        runner = RetrievalRunner(mock_retriever)
        questions = create_test_questions()

        record = runner.evaluate_question(
            question=questions[0],
            system="vector_only",
            top_k=10,
            depth=2,
        )

        assert record.question_id == "q1"
        assert record.system == "vector_only"
        assert record.top_k == 10
        assert record.depth == 0  # Should be 0 for vector_only
        assert record.latency_ms >= 0
        assert "precision_at_5" in record.metrics
        assert "recall_at_10" in record.metrics
        assert "mrr" in record.metrics

        # Verify retriever was called with depth=0
        mock_retriever.retrieve.assert_called_once()
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["depth"] == 0

    def test_evaluate_question_hybrid(self):
        """Test evaluating a question with hybrid system."""
        mock_retriever = create_mock_hybrid_retriever()
        runner = RetrievalRunner(mock_retriever)
        questions = create_test_questions()

        record = runner.evaluate_question(
            question=questions[0],
            system="hybrid",
            top_k=10,
            depth=2,
        )

        assert record.system == "hybrid"
        assert record.depth == 2

        # Verify retriever was called with depth=2
        mock_retriever.retrieve.assert_called_once()
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["depth"] == 2

    def test_evaluate_question_negative_case(self):
        """Test evaluating a negative question (no relevant items)."""
        mock_retriever = create_mock_hybrid_retriever()
        runner = RetrievalRunner(mock_retriever)
        questions = create_test_questions()

        record = runner.evaluate_question(
            question=questions[2],  # negative question
            system="hybrid",
            top_k=10,
            depth=2,
        )

        assert record.question_id == "q3"
        assert record.relevant_ids == []
        # With empty relevant, recall should be 1.0, precision 0.0
        assert record.metrics["recall_at_10"] == 1.0
        assert record.metrics["precision_at_10"] == 0.0
        assert record.metrics["ndcg_at_10"] == 1.0

    def test_evaluate_dataset(self):
        """Test evaluating a full dataset."""
        mock_retriever = create_mock_hybrid_retriever()
        runner = RetrievalRunner(mock_retriever)
        questions = create_test_questions()

        records = runner.evaluate_dataset(
            questions=questions,
            systems=["vector_only", "hybrid"],
            top_k=10,
            depth=2,
        )

        # 3 questions * 2 systems = 6 records
        assert len(records) == 6

        # Check both systems represented
        systems_found = {r.system for r in records}
        assert systems_found == {"vector_only", "hybrid"}

        # Check all questions evaluated
        qids_found = {r.question_id for r in records}
        assert qids_found == {"q1", "q2", "q3"}

    def test_record_to_dict(self):
        """Test QueryResultRecord serialization."""
        record = QueryResultRecord(
            question_id="test",
            category="test_cat",
            difficulty="easy",
            system="hybrid",
            top_k=10,
            depth=2,
            retrieved_ids=["a.py::foo"],
            relevant_ids=["a.py::foo"],
            latency_ms=42.5,
            metrics={"precision_at_10": 0.5, "recall_at_10": 1.0},
        )

        d = record.to_dict()
        assert d["question_id"] == "test"
        assert d["system"] == "hybrid"
        assert d["metrics"]["precision_at_10"] == 0.5


class TestAgentRunner:
    """Tests for AgentRunner."""

    def test_evaluate_question_free_local(self):
        """Test agent evaluation with FreeLocalProvider."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_retriever = Mock(spec=HybridRetriever)

        runner = AgentRunner(mock_vector_store, mock_retriever)

        gt = RetrievalGroundTruth(
            question_id="q1",
            question="What is foo?",
            relevant_chunk_ids=["a.py::foo"],
            category="symbol_lookup",
            difficulty="easy",
        )
        question = EvaluationQuestion(retrieval=gt)

        # Mock the agent's run method
        with patch("repomind.eval.runner.ReActAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Create mock state with some history
            mock_state = Mock(spec=AgentState)
            mock_state.current_iteration = 3
            mock_state.final_answer = "foo is a function in a.py"
            mock_state.tool_calls = [
                ToolCall(id="call_1", name="semantic_search", arguments={"query": "foo"}),
                ToolCall(id="call_2", name="graph_traverse", arguments={"node_id": "a.py::foo", "direction": "upstream"}),
            ]
            mock_state.tool_results = []
            mock_state.errors = []
            mock_agent.run.return_value = mock_state

            record = runner.evaluate_question(
                question=question,
                provider=FreeLocalProvider(),
                max_iterations=5,
            )

        assert record.question_id == "q1"
        assert record.provider_name == "FreeLocalProvider"
        assert record.success is True
        assert record.iterations == 3
        assert record.total_tool_calls == 2
        assert record.tool_counts.get("semantic_search") == 1
        assert record.tool_counts.get("graph_traverse") == 1
        assert record.used_graph_tool is True
        assert record.redundant_tool_calls == 0
        assert "foo is a function" in record.final_answer
        assert record.latency_ms >= 0

    def test_evaluate_question_error_state(self):
        """Test agent evaluation with error state."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_retriever = Mock(spec=HybridRetriever)
        runner = AgentRunner(mock_vector_store, mock_retriever)

        gt = RetrievalGroundTruth(
            question_id="q1",
            question="What is foo?",
            relevant_chunk_ids=["a.py::foo"],
        )
        question = EvaluationQuestion(retrieval=gt)

        with patch("repomind.eval.runner.ReActAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            mock_state = Mock(spec=AgentState)
            mock_state.current_iteration = 5
            mock_state.final_answer = None
            mock_state.tool_calls = [
                ToolCall(id="call_1", name="semantic_search", arguments={"query": "foo"}),
            ]
            mock_state.tool_results = []
            mock_state.errors = ["Some error"]
            mock_agent.run.return_value = mock_state

            record = runner.evaluate_question(question)

        assert record.success is False
        assert record.final_answer == ""

    def test_evaluate_question_redundant_calls(self):
        """Test agent evaluation detects redundant tool calls."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_retriever = Mock(spec=HybridRetriever)
        runner = AgentRunner(mock_vector_store, mock_retriever)

        gt = RetrievalGroundTruth(question_id="q1", question="test")
        question = EvaluationQuestion(retrieval=gt)

        with patch("repomind.eval.runner.ReActAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            mock_state = Mock(spec=AgentState)
            mock_state.current_iteration = 4
            mock_state.final_answer = "answer"
            mock_state.tool_calls = [
                ToolCall(id="call_1", name="semantic_search", arguments={"query": "foo"}),
                ToolCall(id="call_2", name="semantic_search", arguments={"query": "foo"}),  # duplicate
                ToolCall(id="call_3", name="graph_traverse", arguments={"node_id": "a"}),
            ]
            mock_state.tool_results = []
            mock_state.errors = []
            mock_agent.run.return_value = mock_state

            record = runner.evaluate_question(question)

        assert record.redundant_tool_calls == 1  # One duplicate semantic_search call

    def test_evaluate_dataset(self):
        """Test evaluating a dataset with the agent."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_retriever = Mock(spec=HybridRetriever)
        runner = AgentRunner(mock_vector_store, mock_retriever)

        questions = create_test_questions()

        with patch("repomind.eval.runner.ReActAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            mock_state = Mock(spec=AgentState)
            mock_state.current_iteration = 2
            mock_state.final_answer = "answer"
            mock_state.tool_calls = [
                ToolCall(id="call_1", name="semantic_search", arguments={}),
            ]
            mock_state.tool_results = []
            mock_state.errors = []
            mock_agent.run.return_value = mock_state

            records = runner.evaluate_dataset(questions, max_iterations=5)

        assert len(records) == 3
        for r in records:
            assert r.success is True


class TestAgentEvalRecordSerialization:
    """Tests for AgentEvalRecord serialization."""

    def test_to_dict(self):
        """Test AgentEvalRecord to_dict."""
        record = AgentEvalRecord(
            question_id="q1",
            category="symbol_lookup",
            difficulty="easy",
            provider_name="FreeLocalProvider",
            success=True,
            iterations=3,
            total_tool_calls=2,
            tool_counts={"semantic_search": 2},
            used_graph_tool=False,
            redundant_tool_calls=0,
            final_answer="test answer",
            latency_ms=123.45,
        )

        d = record.to_dict()
        assert d["question_id"] == "q1"
        assert d["provider_name"] == "FreeLocalProvider"
        assert d["success"] is True
        assert d["tool_counts"] == {"semantic_search": 2}