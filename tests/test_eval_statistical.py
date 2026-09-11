"""Tests for the statistical analysis module."""

import pytest
from repomind.eval.statistical import (
    bootstrap_mean,
    bootstrap_median,
    paired_wilcoxon,
    paired_t_test,
    cohens_d,
    cliffs_delta,
    interpret_cohens_d,
    interpret_cliffs_delta,
    compute_effect_size,
    compare_systems_paired,
    aggregate_results_by_category,
    aggregate_results_by_system,
)


class TestBootstrapMean:
    """Tests for bootstrap_mean."""

    def test_basic_bootstrap(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, ci = bootstrap_mean(data, n_resamples=100, seed=42)

        assert abs(mean - 3.0) < 0.1
        assert ci.lower <= mean <= ci.upper
        assert ci.confidence == 0.95

    def test_bootstrap_custom_confidence(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, ci = bootstrap_mean(data, n_resamples=100, confidence=0.90, seed=42)

        assert ci.confidence == 0.90

    def test_bootstrap_empty(self):
        mean, ci = bootstrap_mean([], n_resamples=100, seed=42)
        assert mean == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_bootstrap_single_value(self):
        mean, ci = bootstrap_mean([5.0], n_resamples=100, seed=42)
        assert mean == 5.0
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_bootstrap_deterministic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean1, ci1 = bootstrap_mean(data, n_resamples=1000, seed=42)
        mean2, ci2 = bootstrap_mean(data, n_resamples=1000, seed=42)

        assert mean1 == mean2
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper


class TestBootstrapMedian:
    """Tests for bootstrap_median."""

    def test_basic_median(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        median, ci = bootstrap_median(data, n_resamples=100, seed=42)

        assert median == 3.0
        assert ci.lower <= median <= ci.upper

    def test_even_length_median(self):
        data = [1.0, 2.0, 3.0, 4.0]
        median, ci = bootstrap_median(data, n_resamples=100, seed=42)

        assert median == 2.5

    def test_empty_median(self):
        median, ci = bootstrap_median([], n_resamples=100, seed=42)
        assert median == 0.0


class TestPairedWilcoxon:
    """Tests for paired_wilcoxon."""

    def test_wilcoxon_basic(self):
        a = [5, 6, 7, 8, 9]
        b = [1, 1, 2, 2, 3]
        # Differences: 4, 5, 5, 6, 6 - all positive, A > B
        w, p = paired_wilcoxon(a, b, alternative="greater")
        assert p < 0.05  # Should be significant

    def test_wilcoxon_no_difference(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        w, p = paired_wilcoxon(a, b)
        assert p == 1.0  # No difference

    def test_wilcoxon_zero_diffs_removed(self):
        a = [3, 3, 3]
        b = [3, 3, 3]
        w, p = paired_wilcoxon(a, b)
        assert w == 0.0
        assert p == 1.0

    def test_wilcoxon_mismatched_lengths(self):
        with pytest.raises(ValueError):
            paired_wilcoxon([1, 2], [1, 2, 3])


class TestPairedTTest:
    """Tests for paired_t_test."""

    def test_ttest_basic(self):
        a = [5, 6, 7, 8, 9]
        b = [1, 1, 2, 2, 3]
        # Differences: 4, 5, 5, 6, 6 - all positive, A > B
        t, p = paired_t_test(a, b, alternative="greater")
        assert p < 0.05

    def test_ttest_no_difference(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        t, p = paired_t_test(a, b)
        assert p == 1.0

    def test_ttest_zero_variance(self):
        a = [3, 3, 3]
        b = [3, 3, 3]
        t, p = paired_t_test(a, b)
        assert t == 0.0
        assert p == 1.0

    def test_ttest_single_pair(self):
        t, p = paired_t_test([5], [3])
        assert t == 0.0  # Can't compute with n=1
        assert p == 1.0


class TestCohensD:
    """Tests for cohens_d."""

    def test_paired_cohens_d(self):
        a = [5, 4, 3, 2, 1]
        b = [1, 2, 3, 4, 5]
        # Differences: 4, 2, 0, -2, -4; mean = 0; std = sqrt(8) = 2.828
        # Wait, mean diff = (4+2+0-2-4)/5 = 0
        d = cohens_d(a, b, paired=True)
        assert abs(d - 0.0) < 0.01

    def test_paired_cohens_d_positive(self):
        a = [10, 9, 8]
        b = [1, 2, 3]
        # Diffs: 9, 7, 5; mean = 7; std = sqrt((4+0+4)/2) = 2
        # d = 7/2 = 3.5
        d = cohens_d(a, b, paired=True)
        assert abs(d - 3.5) < 0.1

    def test_independent_cohens_d(self):
        a = [10, 10, 10]
        b = [0, 0, 0]
        d = cohens_d(a, b, paired=False)
        # Means: 10 and 0, pooled std = 0 -> should handle
        assert d == 0.0  # Because variance is 0

    def test_independent_cohens_d_normal(self):
        a = [5, 5, 5]
        b = [0, 0, 0]
        d = cohens_d(a, b, paired=False)
        # mean diff = 5, pooled std = 0 (both have zero variance)
        assert d == 0.0

    def test_mismatched_lengths_paired(self):
        with pytest.raises(ValueError):
            cohens_d([1, 2], [1], paired=True)


class TestCliffsDelta:
    """Tests for cliffs_delta."""

    def test_delta_all_greater(self):
        a = [5, 4, 3]
        b = [1, 2]
        # All 3*2=6 pairs: a > b
        delta = cliffs_delta(a, b)
        assert delta == 1.0

    def test_delta_all_less(self):
        a = [1, 2]
        b = [5, 4, 3]
        delta = cliffs_delta(a, b)
        assert delta == -1.0

    def test_delta_equal(self):
        a = [3, 3, 3]
        b = [3, 3, 3]
        delta = cliffs_delta(a, b)
        assert delta == 0.0

    def test_delta_empty(self):
        assert cliffs_delta([], [1, 2]) == 0.0
        assert cliffs_delta([1, 2], []) == 0.0


class TestInterpretations:
    """Tests for effect size interpretations."""

    def test_interpret_cohens_d(self):
        assert interpret_cohens_d(0.1) == "negligible"
        assert interpret_cohens_d(0.3) == "small"
        assert interpret_cohens_d(0.6) == "medium"
        assert interpret_cohens_d(1.0) == "large"
        assert interpret_cohens_d(-0.6) == "medium"  # Uses absolute value

    def test_interpret_cliffs_delta(self):
        assert interpret_cliffs_delta(0.1) == "negligible"
        assert interpret_cliffs_delta(0.2) == "small"
        assert interpret_cliffs_delta(0.4) == "medium"
        assert interpret_cliffs_delta(0.5) == "large"
        assert interpret_cliffs_delta(-0.4) == "medium"


class TestComputeEffectSize:
    """Tests for compute_effect_size."""

    def test_compute_effect_size(self):
        a = [10, 9, 8, 7]
        b = [1, 2, 3, 4]
        effect = compute_effect_size(a, b, paired=True)

        assert effect.cohens_d > 0
        assert effect.cliffs_delta > 0
        assert "large" in effect.interpretation or "medium" in effect.interpretation


class TestCompareSystemsPaired:
    """Tests for compare_systems_paired."""

    def test_compare_systems_basic(self):
        # System A better than B - larger sample for statistical significance
        results_a = [
            {"precision_at_10": 0.8, "recall_at_10": 0.9},
            {"precision_at_10": 0.7, "recall_at_10": 0.8},
            {"precision_at_10": 0.9, "recall_at_10": 1.0},
            {"precision_at_10": 0.75, "recall_at_10": 0.85},
            {"precision_at_10": 0.85, "recall_at_10": 0.95},
            {"precision_at_10": 0.9, "recall_at_10": 0.9},
            {"precision_at_10": 0.8, "recall_at_10": 0.85},
        ]
        results_b = [
            {"precision_at_10": 0.4, "recall_at_10": 0.5},
            {"precision_at_10": 0.3, "recall_at_10": 0.4},
            {"precision_at_10": 0.5, "recall_at_10": 0.6},
            {"precision_at_10": 0.35, "recall_at_10": 0.45},
            {"precision_at_10": 0.45, "recall_at_10": 0.55},
            {"precision_at_10": 0.4, "recall_at_10": 0.5},
            {"precision_at_10": 0.35, "recall_at_10": 0.4},
        ]

        result = compare_systems_paired(results_a, results_b, "precision_at_10", n_bootstrap=100, seed=42)

        assert result["metric"] == "precision_at_10"
        assert result["mean_a"] > result["mean_b"]
        assert result["mean_difference"] > 0
        assert result["wilcoxon_p"] < 0.05
        assert result["cohens_d"] > 0

    def test_compare_systems_equal(self):
        results_a = [{"m": 0.5}, {"m": 0.6}, {"m": 0.7}]
        results_b = [{"m": 0.5}, {"m": 0.6}, {"m": 0.7}]

        result = compare_systems_paired(results_a, results_b, "m", n_bootstrap=100, seed=42)

        assert result["mean_difference"] == 0.0
        assert result["wilcoxon_p"] == 1.0
        assert result["cohens_d"] == 0.0

    def test_compare_systems_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compare_systems_paired([{"m": 1}], [{"m": 1}, {"m": 2}], "m")


class TestAggregateByCategory:
    """Tests for aggregate_results_by_category."""

    def test_aggregate_by_category(self):
        records = [
            {"category": "symbol_lookup", "precision_at_10": 0.8, "recall_at_10": 0.9},
            {"category": "symbol_lookup", "precision_at_10": 0.6, "recall_at_10": 0.7},
            {"category": "call_chain", "precision_at_10": 0.4, "recall_at_10": 0.5},
        ]

        agg = aggregate_results_by_category(records, metric_keys=["precision_at_10", "recall_at_10"])

        assert "symbol_lookup" in agg
        assert "call_chain" in agg
        assert agg["symbol_lookup"]["precision_at_10"] == 0.7
        assert agg["call_chain"]["precision_at_10"] == 0.4

    def test_aggregate_empty(self):
        assert aggregate_results_by_category([]) == {}


class TestAggregateBySystem:
    """Tests for aggregate_results_by_system."""

    def test_aggregate_by_system(self):
        records = [
            {"system": "vector_only", "precision_at_10": 0.5, "recall_at_10": 0.6},
            {"system": "vector_only", "precision_at_10": 0.4, "recall_at_10": 0.5},
            {"system": "hybrid", "precision_at_10": 0.8, "recall_at_10": 0.9},
        ]

        agg = aggregate_results_by_system(records, metric_keys=["precision_at_10", "recall_at_10"])

        assert "vector_only" in agg
        assert "hybrid" in agg
        assert agg["vector_only"]["precision_at_10"] == 0.45
        assert agg["hybrid"]["precision_at_10"] == 0.8