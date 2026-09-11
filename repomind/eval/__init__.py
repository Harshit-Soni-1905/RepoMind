"""Evaluation subsystem: retrieval and answer quality metrics.

This subsystem provides evaluation harness for comparing retrieval strategies
(vector-only vs. hybrid) and measuring answer quality.
"""

from repomind.eval.dataset import (
    RetrievalGroundTruth,
    AnswerGroundTruth,
    EvaluationQuestion,
    load_retrieval_dataset,
    load_answer_dataset,
    load_combined_dataset,
    split_dataset,
    filter_by_category,
    filter_by_difficulty,
)
from repomind.eval.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    r_precision,
    compute_all_metrics,
    aggregate_metrics,
)
from repomind.eval.runner import (
    RetrievalRunner,
    AgentRunner,
    QueryResultRecord,
    AgentEvalRecord,
)
from repomind.eval.statistical import (
    bootstrap_mean,
    bootstrap_median,
    paired_wilcoxon,
    paired_t_test,
    cohens_d,
    cliffs_delta,
    compute_effect_size,
    compare_systems_paired,
    aggregate_results_by_category,
    aggregate_results_by_system,
)
from repomind.eval.reporter import (
    write_jsonl,
    read_jsonl,
    write_csv,
    generate_experiment_manifest,
    write_manifest,
    generate_markdown_report,
    write_markdown_report,
    create_summary_csv,
    ensure_output_dir,
)
from repomind.eval.judge import (
    LLMJudge,
    JudgeResult,
    evaluate_with_judge,
)

__all__ = [
    "RetrievalGroundTruth",
    "AnswerGroundTruth",
    "EvaluationQuestion",
    "load_retrieval_dataset",
    "load_answer_dataset",
    "load_combined_dataset",
    "split_dataset",
    "filter_by_category",
    "filter_by_difficulty",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "r_precision",
    "compute_all_metrics",
    "aggregate_metrics",
    "RetrievalRunner",
    "AgentRunner",
    "QueryResultRecord",
    "AgentEvalRecord",
    "bootstrap_mean",
    "bootstrap_median",
    "paired_wilcoxon",
    "paired_t_test",
    "cohens_d",
    "cliffs_delta",
    "compute_effect_size",
    "compare_systems_paired",
    "aggregate_results_by_category",
    "aggregate_results_by_system",
    "write_jsonl",
    "read_jsonl",
    "write_csv",
    "generate_experiment_manifest",
    "write_manifest",
    "generate_markdown_report",
    "write_markdown_report",
    "create_summary_csv",
    "ensure_output_dir",
    "LLMJudge",
    "JudgeResult",
    "evaluate_with_judge",
]
