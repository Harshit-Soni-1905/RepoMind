"""Tests for the evaluation metrics module.

These tests use known inputs with manually computed expected outputs
to verify the correctness of each metric.
"""

import pytest

from repomind.eval.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    r_precision,
    compute_all_metrics,
    aggregate_metrics,
)


# ---------------------------------------------------------------------------
# Precision@k Tests
# ---------------------------------------------------------------------------

def test_precision_at_k_basic():
    """Test basic P@k calculation."""
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["A", "C", "E"]
    # P@1: A is relevant -> 1/1 = 1.0
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    # P@2: A, B -> 1/2 = 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    # P@3: A, B, C -> 2/3 ≈ 0.667
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    # P@5: A, B, C, D, E -> 3/5 = 0.6
    assert precision_at_k(retrieved, relevant, 5) == 0.6


def test_precision_at_k_no_relevant():
    """Test P@k when retrieved has no relevant items."""
    retrieved = ["X", "Y", "Z"]
    relevant = ["A", "B", "C"]
    assert precision_at_k(retrieved, relevant, 3) == 0.0
    assert precision_at_k(retrieved, relevant, 5) == 0.0


def test_precision_at_k_empty_retrieved():
    """Test P@k with empty retrieved list."""
    assert precision_at_k([], ["A", "B"], 5) == 0.0


def test_precision_at_k_empty_relevant():
    """Test P@k when no relevant items exist."""
    # Edge case: no relevant items to find
    assert precision_at_k(["A", "B", "C"], [], 3) == 0.0


def test_precision_at_k_k_zero():
    """Test P@k with k <= 0."""
    assert precision_at_k(["A", "B"], ["A"], 0) == 0.0
    assert precision_at_k(["A", "B"], ["A"], -1) == 0.0


def test_precision_at_k_k_larger_than_retrieved():
    """Test P@k when k exceeds retrieved length."""
    retrieved = ["A", "B"]
    relevant = ["A", "B", "C"]
    # Only 2 items retrieved, so max hits = 2
    assert precision_at_k(retrieved, relevant, 5) == 2 / 5


# ---------------------------------------------------------------------------
# Recall@k Tests
# ---------------------------------------------------------------------------

def test_recall_at_k_basic():
    """Test basic R@k calculation."""
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["A", "C", "E"]
    # R@1: A is relevant -> 1/3 ≈ 0.333
    assert recall_at_k(retrieved, relevant, 1) == 1 / 3
    # R@2: A, B -> 1/3 ≈ 0.333
    assert recall_at_k(retrieved, relevant, 2) == 1 / 3
    # R@3: A, B, C -> 2/3 ≈ 0.667
    assert recall_at_k(retrieved, relevant, 3) == 2 / 3
    # R@5: all 3 relevant found -> 3/3 = 1.0
    assert recall_at_k(retrieved, relevant, 5) == 1.0


def test_recall_at_k_empty_relevant():
    """Test R@k when no relevant items exist (vacuous truth)."""
    # All 0 relevant items are found
    assert recall_at_k(["A", "B", "C"], [], 3) == 1.0
    assert recall_at_k([], [], 5) == 1.0


def test_recall_at_k_empty_retrieved():
    """Test R@k with empty retrieved list but non-empty relevant."""
    assert recall_at_k([], ["A", "B", "C"], 5) == 0.0


def test_recall_at_k_k_zero():
    """Test R@k with k <= 0."""
    assert recall_at_k(["A", "B"], ["A"], 0) == 0.0
    assert recall_at_k(["A", "B"], ["A"], -1) == 0.0


def test_recall_at_k_no_hits():
    """Test R@k when no relevant items are retrieved."""
    retrieved = ["X", "Y", "Z"]
    relevant = ["A", "B", "C"]
    assert recall_at_k(retrieved, relevant, 3) == 0.0


# ---------------------------------------------------------------------------
# MRR Tests
# ---------------------------------------------------------------------------

def test_mrr_first_item_relevant():
    """MRR when first item is relevant."""
    retrieved = ["A", "B", "C"]
    relevant = ["A"]
    assert mrr(retrieved, relevant) == 1.0


def test_mrr_second_item_relevant():
    """MRR when second item is relevant."""
    retrieved = ["X", "A", "B"]
    relevant = ["A"]
    assert mrr(retrieved, relevant) == 0.5


def test_mrr_third_item_relevant():
    """MRR when third item is relevant."""
    retrieved = ["X", "Y", "A"]
    relevant = ["A"]
    assert mrr(retrieved, relevant) == 1 / 3


def test_mrr_no_relevant_found():
    """MRR when no relevant items in retrieved."""
    retrieved = ["X", "Y", "Z"]
    relevant = ["A"]
    assert mrr(retrieved, relevant) == 0.0


def test_mrr_multiple_relevant():
    """MRR uses first relevant item only."""
    retrieved = ["X", "A", "B", "C"]
    relevant = ["A", "B", "C"]
    # First relevant is at rank 2
    assert mrr(retrieved, relevant) == 0.5


def test_mrr_empty_retrieved():
    assert mrr([], ["A"],) == 0.0


def test_mrr_empty_relevant():
    assert mrr(["A", "B"], []) == 0.0


# ---------------------------------------------------------------------------
# nDCG@k Tests
# ---------------------------------------------------------------------------

def test_ndcg_at_k_perfect_ranking():
    """nDCG when relevant items are ranked first."""
    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "B", "C"]
    # Perfect ranking -> nDCG = 1.0
    assert ndcg_at_k(retrieved, relevant, 3) == 1.0
    assert ndcg_at_k(retrieved, relevant, 4) == 1.0


def test_ndcg_at_k_reverse_ranking():
    """nDCG when relevant items are at the end."""
    retrieved = ["D", "E", "A", "B", "C"]
    relevant = ["A", "B", "C"]
    # Relevant at ranks 3, 4, 5
    # DCG = 1/log2(4) + 1/log2(5) + 1/log2(6)
    # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 1/log2(3) + 1/log2(4)
    score = ndcg_at_k(retrieved, relevant, 5)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_no_relevant():
    """nDCG when no relevant items exist."""
    assert ndcg_at_k(["A", "B", "C"], [], 3) == 1.0


def test_ndcg_at_k_empty_retrieved():
    assert ndcg_at_k([], ["A", "B"], 3) == 0.0


def test_ndcg_at_k_k_zero():
    assert ndcg_at_k(["A"], ["A"], 0) == 0.0


def test_ndcg_at_k_partial():
    """nDCG with some relevant items retrieved."""
    retrieved = ["A", "X", "B", "Y"]
    relevant = ["A", "B", "C"]
    # At k=2: retrieved = [A, X], relevant in top-2 = [A]
    # DCG@2 = 1/log2(2) = 1.0
    # IDCG@2 = 1/log2(2) + 1/log2(3) ≈ 1 + 0.631 = 1.631
    # nDCG@2 ≈ 1/1.631 ≈ 0.613
    score = ndcg_at_k(retrieved, relevant, 2)
    assert 0.6 < score < 0.62


# ---------------------------------------------------------------------------
# R-Precision Tests
# ---------------------------------------------------------------------------

def test_r_precision_basic():
    """Test R-Precision basic calculation."""
    # R = 3 relevant items
    # P@3 with 2 hits = 2/3
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["A", "C", "E"]
    # R = 3, top-3 = [A, B, C] -> hits = 2 -> 2/3
    assert r_precision(retrieved, relevant) == 2 / 3


def test_r_precision_all_relevant_first():
    """R-Precision when all relevant are at top."""
    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "B", "C"]
    # R = 3, top-3 = [A, B, C] -> hits = 3 -> 1.0
    assert r_precision(retrieved, relevant) == 1.0


def test_r_precision_no_relevant():
    """R-Precision when relevant is empty."""
    assert r_precision(["A", "B"], []) == 1.0


def test_r_precision_empty_retrieved():
    assert r_precision([], ["A", "B", "C"]) == 0.0


def test_r_precision_r_exceeds_retrieved():
    """R-Precision when R > len(retrieved)."""
    retrieved = ["A", "B"]
    relevant = ["A", "B", "C", "D"]
    # R = 4, but only 2 retrieved -> P@4 = 2/4 = 0.5
    assert r_precision(retrieved, relevant) == 0.5


# ---------------------------------------------------------------------------
# compute_all_metrics Tests
# ---------------------------------------------------------------------------

def test_compute_all_metrics_default_k():
    """Test compute_all_metrics with default k values."""
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["A", "C", "E"]
    result = compute_all_metrics(retrieved, relevant)

    # Check all expected keys present
    expected_keys = {
        "mrr", "r_precision",
        "precision_at_5", "recall_at_5", "ndcg_at_5",
        "precision_at_10", "recall_at_10", "ndcg_at_10",
        "precision_at_20", "recall_at_20", "ndcg_at_20",
    }
    assert set(result.keys()) == expected_keys

    # Check values
    assert result["mrr"] == 1.0  # A at rank 1
    assert result["r_precision"] == 2 / 3  # R=3, P@3 = 2/3
    assert result["precision_at_5"] == 3 / 5
    assert result["recall_at_5"] == 1.0
    assert result["precision_at_10"] == 3 / 10
    assert result["recall_at_10"] == 1.0


def test_compute_all_metrics_custom_k():
    """Test compute_all_metrics with custom k values."""
    retrieved = ["A", "B", "C"]
    relevant = ["A", "C"]
    result = compute_all_metrics(retrieved, relevant, k_values=[1, 3])

    assert set(result.keys()) == {
        "mrr", "r_precision",
        "precision_at_1", "recall_at_1", "ndcg_at_1",
        "precision_at_3", "recall_at_3", "ndcg_at_3",
    }


# ---------------------------------------------------------------------------
# aggregate_metrics Tests
# ---------------------------------------------------------------------------

def test_aggregate_metrics_basic():
    """Test aggregating metrics across queries."""
    per_query = [
        {"precision_at_10": 0.5, "recall_at_10": 0.8, "mrr": 0.6},
        {"precision_at_10": 0.3, "recall_at_10": 0.6, "mrr": 0.4},
        {"precision_at_10": 0.7, "recall_at_10": 0.9, "mrr": 0.8},
    ]
    result = aggregate_metrics(per_query)

    assert result["precision_at_10"] == (0.5 + 0.3 + 0.7) / 3
    assert result["recall_at_10"] == (0.8 + 0.6 + 0.9) / 3
    assert result["mrr"] == (0.6 + 0.4 + 0.8) / 3


def test_aggregate_metrics_empty():
    assert aggregate_metrics([]) == {}


def test_aggregate_metrics_subset():
    """Test aggregating only specific metrics."""
    per_query = [
        {"precision_at_10": 0.5, "recall_at_10": 0.8, "mrr": 0.6},
        {"precision_at_10": 0.3, "recall_at_10": 0.6, "mrr": 0.4},
    ]
    result = aggregate_metrics(per_query, metric_names=["precision_at_10", "mrr"])
    assert "precision_at_10" in result
    assert "mrr" in result
    assert "recall_at_10" not in result


# ---------------------------------------------------------------------------
# Integration-style Tests with Known Ground Truth
# ---------------------------------------------------------------------------

def test_metrics_with_realistic_scenario():
    """Test metrics with a realistic retrieval scenario."""
    # Ground truth: 5 relevant chunks
    relevant = [
        "auth/middleware.py::validate_token",
        "auth/handlers.py::LoginHandler.post",
        "api/routes.py::login_endpoint",
        "auth/utils.py::hash_password",
        "db/models.py::User",
    ]

    # System A (vector-only): finds 3 relevant, but mixed with irrelevant
    retrieved_a = [
        "auth/middleware.py::validate_token",  # relevant (rank 1)
        "auth/handlers.py::LoginHandler.post",  # relevant (rank 2)
        "ui/components.py::Button",            # irrelevant
        "api/routes.py::login_endpoint",       # relevant (rank 4)
        "config/settings.py::DEBUG",           # irrelevant
    ]

    # System B (hybrid): finds 4 relevant, better ranking
    retrieved_b = [
        "auth/middleware.py::validate_token",  # relevant (rank 1)
        "auth/handlers.py::LoginHandler.post",  # relevant (rank 2)
        "api/routes.py::login_endpoint",       # relevant (rank 3)
        "auth/utils.py::hash_password",        # relevant (rank 4)
        "db/migrations/001.py::create_users",  # irrelevant
    ]

    metrics_a = compute_all_metrics(retrieved_a, relevant)
    metrics_b = compute_all_metrics(retrieved_b, relevant)

    # System B should dominate on all metrics
    assert metrics_b["precision_at_5"] > metrics_a["precision_at_5"]
    assert metrics_b["recall_at_5"] > metrics_a["recall_at_5"]
    assert metrics_b["mrr"] >= metrics_a["mrr"]
    assert metrics_b["ndcg_at_5"] > metrics_a["ndcg_at_5"]
    assert metrics_b["r_precision"] > metrics_a["r_precision"]


def test_metrics_negative_case():
    """Test metrics for negative queries (no relevant items)."""
    relevant = []
    retrieved = ["A", "B", "C", "D", "E"]

    result = compute_all_metrics(retrieved, relevant)

    # With no relevant items:
    # - Precision@k = 0 (no hits possible)
    # - Recall@k = 1.0 (vacuous truth)
    # - MRR = 0.0 (no relevant to find)
    # - nDCG@k = 1.0 (perfect ranking of empty set)
    # - R-Precision = 1.0
    assert result["precision_at_5"] == 0.0
    assert result["recall_at_5"] == 1.0
    assert result["mrr"] == 0.0
    assert result["ndcg_at_5"] == 1.0
    assert result["r_precision"] == 1.0


def test_metrics_single_relevant():
    """Test metrics when only one relevant item exists."""
    relevant = ["auth/middleware.py::validate_token"]
    retrieved = [
        "unrelated.py::foo",
        "auth/middleware.py::validate_token",  # relevant at rank 2
        "other.py::bar",
    ]

    result = compute_all_metrics(retrieved, relevant, k_values=[1, 2, 3])

    # MRR = 1/2 = 0.5
    assert result["mrr"] == 0.5
    # P@1 = 0/1 = 0
    assert result["precision_at_1"] == 0.0
    # P@2 = 1/2 = 0.5
    assert result["precision_at_2"] == 0.5
    # R@1 = 0/1 = 0
    assert result["recall_at_1"] == 0.0
    # R@2 = 1/1 = 1.0
    assert result["recall_at_2"] == 1.0
    # R-Precision: R=1, P@1 = 0
    assert result["r_precision"] == 0.0