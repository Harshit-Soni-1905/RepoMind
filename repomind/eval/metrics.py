"""Deterministic retrieval evaluation metrics.

This module provides standard Information Retrieval metrics for evaluating
retrieval systems. All metrics are deterministic and have no external dependencies.

Metrics implemented:
- Precision@k
- Recall@k
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)
- R-Precision
"""

from typing import List, Dict, Any
import math


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Precision@k.

    Precision@k = |relevant ∩ retrieved@k| / k

    Args:
        retrieved: Ranked list of retrieved item IDs (e.g., chunk_ids)
        relevant: List of relevant item IDs (ground truth)
        k: Cutoff rank

    Returns:
        Precision at k (0.0 to 1.0)

    Edge cases:
        - If k <= 0: returns 0.0
        - If retrieved is empty: returns 0.0
        - If relevant is empty: returns 0.0 (no relevant items to find)
    """
    if k <= 0:
        return 0.0
    if not retrieved or not relevant:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_at_k if item in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Recall@k.

    Recall@k = |relevant ∩ retrieved@k| / |relevant|

    Args:
        retrieved: Ranked list of retrieved item IDs
        relevant: List of relevant item IDs (ground truth)
        k: Cutoff rank

    Returns:
        Recall at k (0.0 to 1.0)

    Edge cases:
        - If k <= 0: returns 0.0
        - If relevant is empty: returns 1.0 (vacuously all relevant found)
        - If retrieved is empty: returns 0.0
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 1.0  # Vacuous truth: all 0 relevant items are found
    if not retrieved:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_at_k if item in relevant_set)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    MRR = 1 / rank of first relevant item (or 0 if none found).
    For a single query, this is just the reciprocal rank.

    Args:
        retrieved: Ranked list of retrieved item IDs
        relevant: List of relevant item IDs (ground truth)

    Returns:
        Reciprocal rank of first relevant item (0.0 to 1.0)

    Edge cases:
        - If retrieved is empty: returns 0.0
        - If relevant is empty: returns 0.0
        - If no relevant item in retrieved: returns 0.0
    """
    if not retrieved or not relevant:
        return 0.0

    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, 1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Discounted Cumulative Gain at k (binary relevance).

    DCG@k = sum_{i=1}^k (rel_i / log2(i+1))
    where rel_i = 1 if item at rank i is relevant, else 0.

    Args:
        retrieved: Ranked list of retrieved item IDs
        relevant: List of relevant item IDs
        k: Cutoff rank

    Returns:
        DCG score
    """
    if k <= 0 or not retrieved:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(retrieved_at_k, 1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k (nDCG@k).

    nDCG@k = DCG@k / IDCG@k
    where IDCG is the ideal DCG (relevant items ranked first).

    Args:
        retrieved: Ranked list of retrieved item IDs
        relevant: List of relevant item IDs (ground truth)
        k: Cutoff rank

    Returns:
        nDCG at k (0.0 to 1.0)

    Edge cases:
        - If k <= 0: returns 0.0
        - If relevant is empty: returns 1.0 (perfect ranking of empty set)
        - If retrieved is empty: returns 0.0
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 1.0  # Perfect ranking of empty relevant set
    if not retrieved:
        return 0.0

    dcg = _dcg_at_k(retrieved, relevant, k)

    # Ideal DCG: relevant items ranked first
    ideal_retrieved = relevant[:k]
    idcg = _dcg_at_k(ideal_retrieved, relevant, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def r_precision(retrieved: List[str], relevant: List[str]) -> float:
    """Compute R-Precision.

    R-Precision = Precision@R where R = |relevant|
    This is the precision at the cutoff equal to the number of relevant items.

    Args:
        retrieved: Ranked list of retrieved item IDs
        relevant: List of relevant item IDs (ground truth)

    Returns:
        R-Precision (0.0 to 1.0)

    Edge cases:
        - If relevant is empty: returns 1.0
        - If retrieved is empty: returns 0.0
    """
    if not relevant:
        return 1.0
    if not retrieved:
        return 0.0

    r = len(relevant)
    return precision_at_k(retrieved, relevant, r)


def compute_all_metrics(
    retrieved: List[str],
    relevant: List[str],
    k_values: List[int] = None
) -> Dict[str, Any]:
    """Compute all retrieval metrics for a single query.

    Args:
        retrieved: Ranked list of retrieved item IDs
        relevant: List of relevant item IDs (ground truth)
        k_values: List of k values to compute P@k, R@k, nDCG@k for.
                  Defaults to [5, 10, 20].

    Returns:
        Dictionary with all metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]

    result = {
        "mrr": mrr(retrieved, relevant),
        "r_precision": r_precision(retrieved, relevant),
    }

    for k in k_values:
        result[f"precision_at_{k}"] = precision_at_k(retrieved, relevant, k)
        result[f"recall_at_{k}"] = recall_at_k(retrieved, relevant, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(retrieved, relevant, k)

    return result


def aggregate_metrics(
    per_query_metrics: List[Dict[str, float]],
    metric_names: List[str] = None
) -> Dict[str, float]:
    """Aggregate per-query metrics across a dataset (mean).

    Args:
        per_query_metrics: List of metric dicts from compute_all_metrics
        metric_names: Specific metrics to aggregate. If None, aggregates all numeric keys.

    Returns:
        Dictionary of mean values for each metric
    """
    if not per_query_metrics:
        return {}

    if metric_names is None:
        metric_names = list(per_query_metrics[0].keys())

    aggregated = {}
    for name in metric_names:
        values = [m.get(name, 0.0) for m in per_query_metrics]
        aggregated[name] = sum(values) / len(values)

    return aggregated