"""Statistical analysis for evaluation results.

Provides bootstrap confidence intervals, paired statistical tests,
and effect size calculations for comparing retrieval/agent systems.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import random
from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    """Represents a confidence interval with lower and upper bounds."""
    lower: float
    upper: float
    confidence: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
        }


@dataclass
class EffectSize:
    """Effect size calculation results."""
    cohens_d: float
    cliffs_delta: float
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohens_d": self.cohens_d,
            "cliffs_delta": self.cliffs_delta,
            "interpretation": self.interpretation,
        }


def bootstrap_mean(
    data: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, ConfidenceInterval]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        data: List of numeric values
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (default 0.95)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sample_mean, ConfidenceInterval)
    """
    if not data:
        return 0.0, ConfidenceInterval(lower=0.0, upper=0.0, confidence=confidence)

    n = len(data)
    rng = random.Random(seed)
    sample_mean = sum(data) / n

    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_resamples):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    # Percentile confidence interval
    alpha = (1.0 - confidence) / 2.0
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1.0 - alpha) * n_resamples)

    bootstrap_means.sort()
    lower = bootstrap_means[lower_idx]
    upper = bootstrap_means[upper_idx]

    return sample_mean, ConfidenceInterval(lower=lower, upper=upper, confidence=confidence)


def bootstrap_median(
    data: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, ConfidenceInterval]:
    """Compute bootstrap confidence interval for the median."""
    if not data:
        return 0.0, ConfidenceInterval(lower=0.0, upper=0.0, confidence=confidence)

    n = len(data)
    sorted_data = sorted(data)
    sample_median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

    rng = random.Random(seed)
    bootstrap_medians = []
    for _ in range(n_resamples):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        sample.sort()
        m = sample[n // 2] if n % 2 == 1 else (sample[n // 2 - 1] + sample[n // 2]) / 2
        bootstrap_medians.append(m)

    alpha = (1.0 - confidence) / 2.0
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1.0 - alpha) * n_resamples)

    bootstrap_medians.sort()
    lower = bootstrap_medians[lower_idx]
    upper = bootstrap_medians[upper_idx]

    return sample_median, ConfidenceInterval(lower=lower, upper=upper, confidence=confidence)


def paired_wilcoxon(
    sample_a: List[float],
    sample_b: List[float],
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test on paired samples.

    This is a non-parametric alternative to paired t-test.
    Returns (W_statistic, p_value).

    Args:
        sample_a: First paired sample
        sample_b: Second paired sample (same length)
        alternative: "two-sided", "greater" (A > B), or "less" (A < B)

    Returns:
        Tuple of (W_statistic, p_value)
    """
    if len(sample_a) != len(sample_b):
        raise ValueError("Paired samples must have same length")

    n = len(sample_a)
    if n == 0:
        return 0.0, 1.0

    # Compute differences
    diffs = [a - b for a, b in zip(sample_a, sample_b)]

    # Remove zero differences
    non_zero_diffs = [(d, i) for i, d in enumerate(diffs) if d != 0]
    if not non_zero_diffs:
        return 0.0, 1.0

    # Rank absolute differences
    abs_diffs = [(abs(d), i) for d, i in non_zero_diffs]
    abs_diffs.sort(key=lambda x: x[0])

    # Handle ties by assigning average rank
    ranks = {}
    i = 0
    while i < len(abs_diffs):
        j = i
        while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j

    # Sum positive and negative ranks
    w_plus = sum(ranks[idx] for d, idx in non_zero_diffs if d > 0)
    w_minus = sum(ranks[idx] for d, idx in non_zero_diffs if d < 0)

    # For one-sided tests, use the appropriate rank sum
    # For "greater" (A > B): test if w_plus is large
    # For "less" (A < B): test if w_minus is large
    # For "two-sided": use min(w_plus, w_minus)
    if alternative == "greater":
        W = w_plus
    elif alternative == "less":
        W = w_minus
    else:  # two-sided
        W = min(w_plus, w_minus)

    # For small samples, use exact distribution approximation
    # For larger samples, use normal approximation
    n_nonzero = len(non_zero_diffs)

    if n_nonzero < 10:
        # Exact p-value calculation for small n (simplified)
        # In practice, would use scipy.stats.wilcoxon
        # Here we use normal approximation with continuity correction
        pass

    # Normal approximation with continuity correction
    mu = n_nonzero * (n_nonzero + 1) / 4.0
    sigma = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)

    if sigma == 0:
        return float(W), 1.0

    z = (W - mu + 0.5) / sigma  # Continuity correction

    # Two-sided p-value from standard normal
    p_two_sided = 2 * (1 - _normal_cdf(abs(z)))

    if alternative == "two-sided":
        p_value = p_two_sided
    elif alternative == "greater":
        p_value = 1 - _normal_cdf(z)
    elif alternative == "less":
        p_value = _normal_cdf(z)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(W), p_value


def paired_t_test(
    sample_a: List[float],
    sample_b: List[float],
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Perform paired t-test on paired samples.

    Returns (t_statistic, p_value).
    """
    if len(sample_a) != len(sample_b):
        raise ValueError("Paired samples must have same length")

    n = len(sample_a)
    if n < 2:
        return 0.0, 1.0

    diffs = [a - b for a, b in zip(sample_a, sample_b)]
    mean_diff = sum(diffs) / n

    # Standard deviation of differences
    if n == 1:
        return 0.0, 1.0

    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    if variance == 0:
        return 0.0, 1.0

    std_diff = math.sqrt(variance)
    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # p-value using Student's t-distribution
    # Using normal approximation for simplicity (conservative for small n)
    # For proper implementation, would need scipy.stats.t
    df = n - 1

    # Normal approximation for t-distribution (reasonable for df >= 10)
    z = t_stat
    p_two_sided = 2 * (1 - _normal_cdf(abs(z)))

    if alternative == "two-sided":
        p_value = p_two_sided
    elif alternative == "greater":
        p_value = 1 - _normal_cdf(z)
    elif alternative == "less":
        p_value = _normal_cdf(z)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(t_stat), p_value


def cohens_d(
    sample_a: List[float],
    sample_b: List[float],
    paired: bool = True
) -> float:
    """Calculate Cohen's d effect size.

    Args:
        sample_a: First sample
        sample_b: Second sample
        paired: If True, use paired Cohen's d (mean diff / SD of diffs)

    Returns:
        Cohen's d value
    """
    if len(sample_a) != len(sample_b):
        raise ValueError("Samples must have same length for paired Cohen's d")

    if paired:
        diffs = [a - b for a, b in zip(sample_a, sample_b)]
        n = len(diffs)
        if n < 2:
            return 0.0
        mean_diff = sum(diffs) / n
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        if variance == 0:
            return 0.0
        return mean_diff / math.sqrt(variance)
    else:
        # Independent samples Cohen's d
        n1, n2 = len(sample_a), len(sample_b)
        if n1 < 2 or n2 < 2:
            return 0.0
        mean1 = sum(sample_a) / n1
        mean2 = sum(sample_b) / n2
        var1 = sum((x - mean1) ** 2 for x in sample_a) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in sample_b) / (n2 - 1)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (mean1 - mean2) / pooled_std


def cliffs_delta(
    sample_a: List[float],
    sample_b: List[float]
) -> float:
    """Calculate Cliff's delta (non-parametric effect size).

    Measures the probability that a random value from A is greater than
    a random value from B, minus the reverse probability.

    Returns:
        Cliff's delta in range [-1, 1]
    """
    if not sample_a or not sample_b:
        return 0.0

    n1, n2 = len(sample_a), len(sample_b)
    total_pairs = n1 * n2

    # Count dominance pairs
    greater = sum(1 for a in sample_a for b in sample_b if a > b)
    less = sum(1 for a in sample_a for b in sample_b if a < b)

    delta = (greater - less) / total_pairs
    return delta


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def compute_effect_size(
    sample_a: List[float],
    sample_b: List[float],
    paired: bool = True
) -> EffectSize:
    """Compute both Cohen's d and Cliff's delta with interpretations."""
    d = cohens_d(sample_a, sample_b, paired=paired)
    delta = cliffs_delta(sample_a, sample_b)

    # Use paired interpretation for Cohen's d when paired
    if paired:
        interp = interpret_cohens_d(d)
    else:
        interp = interpret_cohens_d(d)

    return EffectSize(
        cohens_d=d,
        cliffs_delta=delta,
        interpretation=f"Cohen's d: {interp}, Cliff's delta: {interpret_cliffs_delta(delta)}"
    )


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compare_systems_paired(
    results_a: List[Dict[str, float]],
    results_b: List[Dict[str, float]],
    metric_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Compare two systems on a metric using paired analysis.

    Args:
        results_a: List of metric dicts for system A (same questions as B)
        results_b: List of metric dicts for system B
        metric_name: Metric key to compare
        n_bootstrap: Bootstrap resamples for CI
        seed: Random seed

    Returns:
        Dictionary with mean diff, CI, p-values, effect sizes
    """
    if len(results_a) != len(results_b):
        raise ValueError("Paired comparison requires same number of queries")

    vals_a = [r.get(metric_name, 0.0) for r in results_a]
    vals_b = [r.get(metric_name, 0.0) for r in results_b]

    # Mean difference (A - B)
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    mean_diff = sum(diffs) / len(diffs)

    # Bootstrap CI for mean difference
    _, ci = bootstrap_mean(diffs, n_resamples=n_bootstrap, seed=seed)

    # Paired statistical tests
    _, p_wilcoxon = paired_wilcoxon(vals_a, vals_b)
    _, p_ttest = paired_t_test(vals_a, vals_b)

    # Effect sizes
    effect = compute_effect_size(vals_a, vals_b, paired=True)

    return {
        "metric": metric_name,
        "mean_a": sum(vals_a) / len(vals_a),
        "mean_b": sum(vals_b) / len(vals_b),
        "mean_difference": mean_diff,
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
        "ci_confidence": ci.confidence,
        "wilcoxon_p": p_wilcoxon,
        "ttest_p": p_ttest,
        "cohens_d": effect.cohens_d,
        "cliffs_delta": effect.cliffs_delta,
        "effect_interpretation": effect.interpretation,
        "n_queries": len(vals_a),
    }


def aggregate_results_by_category(
    records: List[Dict[str, Any]],
    category_key: str = "category",
    metric_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Aggregate metric results by category.

    Args:
        records: List of result records with metrics
        category_key: Field name for category
        metric_keys: Specific metrics to aggregate (defaults to all numeric keys)

    Returns:
        Dict mapping category -> metric -> mean value
    """
    if not records:
        return {}

    # Group by category
    by_cat: Dict[str, List[Dict]] = {}
    for r in records:
        cat = r.get(category_key, "unknown")
        by_cat.setdefault(cat, []).append(r)

    if metric_keys is None:
        # Get all numeric keys from first record
        first = records[0]
        metric_keys = [k for k, v in first.items() if isinstance(v, (int, float)) and k not in ("top_k", "depth")]

    result = {}
    for cat, cat_records in by_cat.items():
        cat_metrics = {}
        for metric in metric_keys:
            vals = [r.get(metric, 0.0) for r in cat_records]
            cat_metrics[metric] = sum(vals) / len(vals) if vals else 0.0
        result[cat] = cat_metrics

    return result


def aggregate_results_by_system(
    records: List[Dict[str, Any]],
    system_key: str = "system",
    metric_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Aggregate metric results by system (e.g., vector_only vs hybrid)."""
    if not records:
        return {}

    by_sys: Dict[str, List[Dict]] = {}
    for r in records:
        sys = r.get(system_key, "unknown")
        by_sys.setdefault(sys, []).append(r)

    if metric_keys is None:
        first = records[0]
        metric_keys = [k for k, v in first.items() if isinstance(v, (int, float)) and k not in ("top_k", "depth")]

    result = {}
    for sys, sys_records in by_sys.items():
        sys_metrics = {}
        for metric in metric_keys:
            vals = [r.get(metric, 0.0) for r in sys_records]
            sys_metrics[metric] = sum(vals) / len(vals) if vals else 0.0
        result[sys] = sys_metrics

    return result