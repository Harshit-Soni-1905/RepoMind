"""Tests for the LLM judge module."""

import os

import pytest
from unittest.mock import Mock, patch, MagicMock
from repomind.eval.judge import LLMJudge, JudgeResult, evaluate_with_judge


class TestJudgeResult:
    """Tests for JudgeResult dataclass."""

    def test_average_score(self):
        result = JudgeResult(
            faithfulness=4,
            correctness=5,
            completeness=3,
            reasoning="Good answer"
        )
        assert result.average_score == 4.0

    def test_to_dict(self):
        result = JudgeResult(
            faithfulness=4,
            correctness=5,
            completeness=3,
            reasoning="Good answer"
        )
        d = result.to_dict()
        assert d["faithfulness"] == 4
        assert d["correctness"] == 5
        assert d["completeness"] == 3
        assert d["reasoning"] == "Good answer"


class TestLLMJudge:
    """Tests for LLMJudge class."""

    def test_judge_result_parsing(self):
        judge = LLMJudge()

        # Test parsing JSON response
        mock_response = Mock()
        mock_response.text = '{"faithfulness": 4, "correctness": 5, "completeness": 3, "reasoning": "Good"}'

        with patch.object(judge, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Mock is_available to return True
            with patch.object(judge, 'is_available', return_value=True):
                result = judge.evaluate(
                    question="What is foo?",
                    answer="foo is a function",
                    context_chunks=[{"filepath": "a.py", "symbol_name": "foo", "source_code": "def foo(): pass"}]
                )

        assert result is not None
        assert result.faithfulness == 4
        assert result.correctness == 5
        assert result.completeness == 3

    def test_judge_handles_markdown_fences(self):
        """Test that judge handles ```json fences in response."""
        judge = LLMJudge()

        mock_response = Mock()
        mock_response.text = '```json\n{"faithfulness": 3, "correctness": 4, "completeness": 3, "reasoning": "Okay"}\n```'

        with patch.object(judge, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            with patch.object(judge, 'is_available', return_value=True):
                result = judge.evaluate("Q?", "A", [])

        assert result is not None
        assert result.faithfulness == 3

    def test_judge_invalid_json(self):
        """Test judge handles invalid JSON gracefully."""
        judge = LLMJudge()

        mock_response = Mock()
        mock_response.text = "This is not JSON"

        with patch.object(judge, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            with patch.object(judge, 'is_available', return_value=True):
                result = judge.evaluate("Q?", "A", [])

        assert result is None

    def test_judge_missing_fields(self):
        """Test judge handles missing fields in JSON."""
        judge = LLMJudge()

        mock_response = Mock()
        mock_response.text = '{"faithfulness": 4}'  # Missing other fields

        with patch.object(judge, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            with patch.object(judge, 'is_available', return_value=True):
                result = judge.evaluate("Q?", "A", [])

        # Should handle missing fields with defaults
        assert result is not None

    def test_is_available_false_no_key(self):
        """Test is_available returns False when no API key."""
        judge = LLMJudge()

        # Only remove GOOGLE_API_KEY — clearing *all* env vars removes
        # SSL config that google-genai's aiohttp import needs on init.
        env_without_key = {k: v for k, v in os.environ.items() if k != 'GOOGLE_API_KEY'}
        with patch.dict('os.environ', env_without_key, clear=True):
            assert judge.is_available() is False

    def test_is_available_false_no_package(self):
        """Test is_available returns False when package not installed."""
        judge = LLMJudge()

        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            # Patch _get_client to raise RuntimeError (simulates missing package)
            with patch.object(judge, '_get_client', side_effect=RuntimeError("google-genai not installed")):
                assert judge.is_available() is False


class TestEvaluateWithJudge:
    """Tests for evaluate_with_judge function."""

    def test_evaluate_with_judge_unavailable(self):
        """Test evaluate_with_judge returns empty list when judge unavailable."""
        questions = [{"question_id": "q1", "question": "What is foo?"}]
        answers = [{"question_id": "q1", "answer": "foo is a function"}]
        context_map = {"q1": [{"filepath": "a.py", "symbol_name": "foo"}]}

        with patch('repomind.eval.judge.LLMJudge') as mock_judge_class:
            mock_judge = Mock()
            mock_judge.is_available.return_value = False
            mock_judge_class.return_value = mock_judge

            results = evaluate_with_judge(questions, answers, context_map)

        assert results == []

    def test_evaluate_with_judge_basic(self):
        """Test evaluate_with_judge when judge is available."""
        questions = [{"question_id": "q1", "question": "What is foo?"}]
        answers = [{"question_id": "q1", "answer": "foo is a function"}]
        context_map = {"q1": [{"filepath": "a.py", "symbol_name": "foo"}]}

        with patch('repomind.eval.judge.LLMJudge') as mock_judge_class:
            mock_judge = Mock()
            mock_judge.is_available.return_value = True
            mock_judge.evaluate.return_value = JudgeResult(
                faithfulness=4, correctness=5, completeness=3, reasoning="Good"
            )
            mock_judge_class.return_value = mock_judge

            results = evaluate_with_judge(questions, answers, context_map)

        assert len(results) == 1
        assert results[0]["question_id"] == "q1"
        assert results[0]["faithfulness"] == 4
        assert results[0]["correctness"] == 5
        assert results[0]["completeness"] == 3
        assert results[0]["average_score"] == 4.0

    def test_evaluate_with_judge_missing_answer(self):
        """Test evaluate_with_judge skips questions without answers."""
        questions = [
            {"question_id": "q1", "question": "What is foo?"},
            {"question_id": "q2", "question": "What is bar?"},
        ]
        answers = [{"question_id": "q1", "answer": "foo is a function"}]
        context_map = {
            "q1": [{"filepath": "a.py", "symbol_name": "foo"}],
            "q2": [{"filepath": "b.py", "symbol_name": "bar"}],
        }

        with patch('repomind.eval.judge.LLMJudge') as mock_judge_class:
            mock_judge = Mock()
            mock_judge.is_available.return_value = True
            mock_judge.evaluate.return_value = JudgeResult(
                faithfulness=4, correctness=5, completeness=3, reasoning="Good"
            )
            mock_judge_class.return_value = mock_judge

            results = evaluate_with_judge(questions, answers, context_map)

        assert len(results) == 1
        assert results[0]["question_id"] == "q1"

    def test_evaluate_with_judge_evaluation_fails(self):
        """Test evaluate_with_judge skips failed evaluations."""
        questions = [{"question_id": "q1", "question": "What is foo?"}]
        answers = [{"question_id": "q1", "answer": "foo is a function"}]
        context_map = {"q1": [{"filepath": "a.py", "symbol_name": "foo"}]}

        with patch('repomind.eval.judge.LLMJudge') as mock_judge_class:
            mock_judge = Mock()
            mock_judge.is_available.return_value = True
            mock_judge.evaluate.return_value = None  # Evaluation failed
            mock_judge_class.return_value = mock_judge

            results = evaluate_with_judge(questions, answers, context_map)

        assert results == []