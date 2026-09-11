"""Tests for the evaluation reporter module."""

import pytest
import json
import csv
import tempfile
from pathlib import Path

from repomind.eval.reporter import (
    write_jsonl,
    read_jsonl,
    write_csv,
    flatten_metrics_for_csv,
    generate_experiment_manifest,
    write_manifest,
    generate_markdown_report,
    write_markdown_report,
    create_summary_csv,
    ensure_output_dir,
)
from repomind.eval.runner import QueryResultRecord, AgentEvalRecord
from repomind.eval.dataset import EvaluationQuestion, RetrievalGroundTruth


class TestWriteReadJsonl:
    """Tests for JSONL read/write."""

    def test_write_read_jsonl(self):
        records = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            write_jsonl(records, path)
            read_back = read_jsonl(path)
            assert read_back == records
        finally:
            path.unlink()

    def test_write_jsonl_creates_dirs(self):
        records = [{"test": "data"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "file.jsonl"
            write_jsonl(records, path)
            assert path.exists()
            read_back = read_jsonl(path)
            assert read_back == records

    def test_read_jsonl_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n\n{"b": 2}\n \n{"c": 3}\n')
            path = Path(f.name)

        try:
            read_back = read_jsonl(path)
            assert len(read_back) == 3
        finally:
            path.unlink()


class TestWriteCsv:
    """Tests for CSV writing."""

    def test_write_csv_basic(self):
        records = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_csv(records, path)
            # Read back and verify
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["name"] == "a"
            assert rows[0]["value"] == "1"
        finally:
            path.unlink()

    def test_write_csv_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_csv([], path)
            assert path.exists()
        finally:
            path.unlink()

    def test_write_csv_fieldnames(self):
        records = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_csv(records, path, fieldnames=["a", "b"])
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert "c" not in reader.fieldnames  # extrasaction="ignore"
        finally:
            path.unlink()


class TestFlattenMetrics:
    """Tests for flatten_metrics_for_csv."""

    def test_flatten_metrics(self):
        record = {
            "question_id": "q1",
            "system": "hybrid",
            "metrics": {"precision_at_10": 0.5, "recall_at_10": 0.8},
        }
        flat = flatten_metrics_for_csv(record)
        assert flat["question_id"] == "q1"
        assert flat["system"] == "hybrid"
        assert flat["precision_at_10"] == 0.5
        assert flat["recall_at_10"] == 0.8
        assert "metrics" not in flat

    def test_flatten_no_metrics(self):
        record = {"question_id": "q1", "system": "hybrid"}
        flat = flatten_metrics_for_csv(record)
        assert flat == record


class TestGenerateManifest:
    """Tests for generate_experiment_manifest."""

    def test_generate_manifest(self):
        manifest = generate_experiment_manifest(
            experiment_id="test_001",
            git_commit="abc123",
            config={"repos": ["/path/to/repo"], "top_k": 10},
            systems=["vector_only", "hybrid"],
            metrics_computed=["precision_at_10", "recall_at_10"],
        )

        assert manifest["experiment_id"] == "test_001"
        assert manifest["git_commit"] == "abc123"
        assert manifest["config"]["repos"] == ["/path/to/repo"]
        assert manifest["systems"] == ["vector_only", "hybrid"]
        assert "timestamp" in manifest
        assert "environment" in manifest


class TestMarkdownReport:
    """Tests for markdown report generation."""

    def create_test_records(self):
        """Create test QueryResultRecords."""
        gt1 = RetrievalGroundTruth(
            question_id="q1",
            question="What is foo?",
            relevant_chunk_ids=["a.py::foo"],
            category="symbol_lookup",
            difficulty="easy",
        )
        gt2 = RetrievalGroundTruth(
            question_id="q2",
            question="What calls foo?",
            relevant_chunk_ids=["b.py::bar"],
            category="call_chain",
            difficulty="medium",
        )

        return [
            QueryResultRecord(
                question_id="q1", category="symbol_lookup", difficulty="easy",
                system="vector_only", top_k=10, depth=0,
                retrieved_ids=["a.py::foo"], relevant_ids=["a.py::foo"],
                latency_ms=10.5,
                metrics={"precision_at_10": 0.5, "recall_at_10": 0.8, "mrr": 0.6, "ndcg_at_10": 0.7, "r_precision": 0.5},
            ),
            QueryResultRecord(
                question_id="q2", category="call_chain", difficulty="medium",
                system="vector_only", top_k=10, depth=0,
                retrieved_ids=["b.py::bar"], relevant_ids=["b.py::bar"],
                latency_ms=12.0,
                metrics={"precision_at_10": 0.4, "recall_at_10": 0.7, "mrr": 0.5, "ndcg_at_10": 0.6, "r_precision": 0.4},
            ),
            QueryResultRecord(
                question_id="q1", category="symbol_lookup", difficulty="easy",
                system="hybrid", top_k=10, depth=2,
                retrieved_ids=["a.py::foo", "b.py::bar"], relevant_ids=["a.py::foo"],
                latency_ms=15.5,
                metrics={"precision_at_10": 0.8, "recall_at_10": 1.0, "mrr": 0.9, "ndcg_at_10": 0.85, "r_precision": 0.8},
            ),
            QueryResultRecord(
                question_id="q2", category="call_chain", difficulty="medium",
                system="hybrid", top_k=10, depth=2,
                retrieved_ids=["b.py::bar", "a.py::foo"], relevant_ids=["b.py::bar"],
                latency_ms=18.0,
                metrics={"precision_at_10": 0.7, "recall_at_10": 0.9, "mrr": 0.8, "ndcg_at_10": 0.75, "r_precision": 0.7},
            ),
        ]

    def test_generate_markdown_report(self):
        records = self.create_test_records()

        manifest = {
            "experiment_id": "test_001",
            "git_commit": "abc123",
            "config": {
                "repos": ["/test/repo"],
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_top_k": 10,
                "graph_depth": 2,
                "fusion_weights": {"vector": 0.7, "graph": 0.3},
            },
        }

        report = generate_markdown_report(
            experiment_id="test_001",
            manifest=manifest,
            retrieval_records=records,
            statistical_results={},
        )

        assert "# RepoMind Retrieval Benchmark — test_001" in report
        assert "## Configuration" in report
        assert "## Overall Results" in report
        assert "vector_only" in report
        assert "hybrid" in report
        assert "precision_at_10" in report
        assert "## By Category" in report
        assert "## Statistical Significance" not in report  # Empty

    def test_generate_markdown_report_with_stats(self):
        records = self.create_test_records()

        manifest = {"experiment_id": "test_001", "git_commit": "abc123", "config": {}}

        statistical_results = {
            "precision_at_10": {
                "mean_a": 0.45,
                "mean_b": 0.75,
                "mean_difference": 0.3,
                "ci_lower": 0.1,
                "ci_upper": 0.5,
                "ci_confidence": 0.95,
                "wilcoxon_p": 0.03,
                "ttest_p": 0.04,
                "cohens_d": 0.8,
                "cliffs_delta": 0.5,
                "effect_interpretation": "Cohen's d: large, Cliff's delta: medium",
                "n_queries": 2,
            }
        }

        report = generate_markdown_report(
            experiment_id="test_001",
            manifest=manifest,
            retrieval_records=records,
            statistical_results=statistical_results,
        )

        assert "## Statistical Significance" in report
        assert "precision_at_10" in report
        assert "Mean difference" in report
        assert "Wilcoxon p-value" in report

    def test_write_markdown_report(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            write_markdown_report("# Test Report\n\nContent", path)
            assert path.exists()
            content = path.read_text()
            assert content == "# Test Report\n\nContent"
        finally:
            path.unlink()


class TestCreateSummaryCsv:
    """Tests for create_summary_csv."""

    def test_create_summary_csv(self):
        gt1 = RetrievalGroundTruth(question_id="q1", question="test", category="symbol_lookup")
        gt2 = RetrievalGroundTruth(question_id="q2", question="test", category="call_chain")

        records = [
            QueryResultRecord(
                question_id="q1", category="symbol_lookup", difficulty="easy",
                system="vector_only", top_k=10, depth=0,
                retrieved_ids=[], relevant_ids=[],
                latency_ms=10.0,
                metrics={"precision_at_10": 0.5, "recall_at_10": 0.8},
            ),
            QueryResultRecord(
                question_id="q2", category="call_chain", difficulty="medium",
                system="vector_only", top_k=10, depth=0,
                retrieved_ids=[], relevant_ids=[],
                latency_ms=12.0,
                metrics={"precision_at_10": 0.4, "recall_at_10": 0.7},
            ),
            QueryResultRecord(
                question_id="q1", category="symbol_lookup", difficulty="easy",
                system="hybrid", top_k=10, depth=2,
                retrieved_ids=[], relevant_ids=[],
                latency_ms=15.0,
                metrics={"precision_at_10": 0.8, "recall_at_10": 1.0},
            ),
            QueryResultRecord(
                question_id="q2", category="call_chain", difficulty="medium",
                system="hybrid", top_k=10, depth=2,
                retrieved_ids=[], relevant_ids=[],
                latency_ms=16.0,
                metrics={"precision_at_10": 0.7, "recall_at_10": 0.9},
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            create_summary_csv(records, path)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have overall, category, and system_category rows
            assert len(rows) > 0

            # Check overall rows exist
            overall_rows = [r for r in rows if r["level"] == "overall"]
            assert len(overall_rows) == 2  # vector_only and hybrid

            # Check category rows
            cat_rows = [r for r in rows if r["level"] == "category"]
            assert len(cat_rows) == 2  # symbol_lookup and call_chain

            # Check system_category rows
            sys_cat_rows = [r for r in rows if r["level"] == "system_category"]
            assert len(sys_cat_rows) == 4  # 2 systems * 2 categories
        finally:
            path.unlink()


class TestEnsureOutputDir:
    """Tests for ensure_output_dir."""

    def test_ensure_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = ensure_output_dir(base, "exp_001")
            assert run_dir == base / "exp_001"
            assert run_dir.exists()

    def test_ensure_output_dir_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "eval" / "results"
            run_dir = ensure_output_dir(base, "exp_002")
            assert run_dir == base / "exp_002"
            assert run_dir.exists()