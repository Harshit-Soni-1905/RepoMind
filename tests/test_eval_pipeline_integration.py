"""Regression tests for the benchmark driver (scripts/run_evaluation.py).

These tests guard the wiring between the eval library and the CLI driver.
The unit tests for repomind.eval.* all pass with mocked inputs, so they could
not catch two classes of defect that made the driver unrunnable / untruthful:

1. The driver imported subsystem helpers that do not exist (ImportError).
2. The driver fed nested-metrics records to compare_systems_paired, which reads
   metrics as top-level keys — silently producing 0.0 for every metric and a
   fabricated "no difference" statistical result.
"""

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repomind.eval.reporter import flatten_metrics_for_csv
from repomind.eval.runner import QueryResultRecord
from repomind.eval.statistical import compare_systems_paired


class TestDriverImportable:
    """The benchmark driver must import against the real subsystem APIs."""

    def test_run_evaluation_module_imports(self):
        """scripts/run_evaluation.py must import cleanly.

        Regression: the driver previously imported read_files,
        parse_python_files, chunk_parsed_files, build_graph and LocalEmbedder,
        none of which exist, so the benchmark could never start.
        """
        module = importlib.import_module("scripts.run_evaluation")
        assert hasattr(module, "build_repo_index")
        assert hasattr(module, "run_retrieval_evaluation")
        assert hasattr(module, "main")

    def test_build_repo_index_signature_accepts_topk_and_depth(self):
        """Both systems must share one index, parameterised by top_k/depth."""
        import inspect

        module = importlib.import_module("scripts.run_evaluation")
        params = inspect.signature(module.build_repo_index).parameters
        assert "top_k" in params
        assert "depth" in params


class TestStatisticalInputContract:
    """compare_systems_paired requires flattened metrics, not nested ones."""

    def _records(self):
        """Two paired questions where hybrid strictly beats vector_only."""
        vector = [
            QueryResultRecord(
                question_id="q1", category="call_chain", difficulty="easy",
                system="vector_only", top_k=10, depth=0,
                retrieved_ids=["a.py::foo"], relevant_ids=["a.py::foo"],
                latency_ms=10.0,
                metrics={"precision_at_10": 0.1, "recall_at_10": 0.5},
            ),
            QueryResultRecord(
                question_id="q2", category="call_chain", difficulty="easy",
                system="vector_only", top_k=10, depth=0,
                retrieved_ids=["b.py::bar"], relevant_ids=["b.py::bar"],
                latency_ms=11.0,
                metrics={"precision_at_10": 0.2, "recall_at_10": 0.4},
            ),
        ]
        hybrid = [
            QueryResultRecord(
                question_id="q1", category="call_chain", difficulty="easy",
                system="hybrid", top_k=10, depth=2,
                retrieved_ids=["a.py::foo"], relevant_ids=["a.py::foo"],
                latency_ms=20.0,
                metrics={"precision_at_10": 0.4, "recall_at_10": 1.0},
            ),
            QueryResultRecord(
                question_id="q2", category="call_chain", difficulty="easy",
                system="hybrid", top_k=10, depth=2,
                retrieved_ids=["b.py::bar"], relevant_ids=["b.py::bar"],
                latency_ms=21.0,
                metrics={"precision_at_10": 0.5, "recall_at_10": 0.8},
            ),
        ]
        return vector, hybrid

    def test_nested_records_lose_all_metrics(self):
        """Documents the failure mode: nested records read as all zeros.

        This asserts the *buggy* input shape produces a degenerate result, which
        is exactly why the driver must flatten before calling.
        """
        vector, hybrid = self._records()
        nested_v = [r.to_dict() for r in vector]
        nested_h = [r.to_dict() for r in hybrid]

        comp = compare_systems_paired(nested_h, nested_v, "precision_at_10")

        # Every value is absent -> 0.0, so the comparison is meaningless.
        assert comp["mean_a"] == 0.0
        assert comp["mean_b"] == 0.0
        assert comp["mean_difference"] == 0.0

    def test_flattened_records_preserve_real_difference(self):
        """Flattened records must surface the true hybrid-over-vector gain."""
        vector, hybrid = self._records()
        flat_v = [flatten_metrics_for_csv(r.to_dict()) for r in vector]
        flat_h = [flatten_metrics_for_csv(r.to_dict()) for r in hybrid]

        comp = compare_systems_paired(flat_h, flat_v, "precision_at_10")

        # hybrid mean = (0.4+0.5)/2 = 0.45; vector mean = (0.1+0.2)/2 = 0.15
        assert comp["mean_a"] == pytest.approx(0.45)
        assert comp["mean_b"] == pytest.approx(0.15)
        assert comp["mean_difference"] == pytest.approx(0.30)
        assert comp["n_queries"] == 2

    def test_driver_reports_hybrid_as_sample_a(self):
        """A positive mean_difference must mean hybrid improved on baseline.

        Guards the argument order in run_retrieval_evaluation: passing
        vector_only as sample A would invert the sign of every reported gain.
        """
        vector, hybrid = self._records()
        flat_v = [flatten_metrics_for_csv(r.to_dict()) for r in vector]
        flat_h = [flatten_metrics_for_csv(r.to_dict()) for r in hybrid]

        comp = compare_systems_paired(flat_h, flat_v, "recall_at_10")
        assert comp["mean_difference"] > 0
        assert comp["cohens_d"] > 0
