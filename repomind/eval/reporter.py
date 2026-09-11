"""Report generation for evaluation results.

Produces JSONL raw results, CSV summaries, and Markdown reports
for the benchmark evaluation.
"""

from typing import List, Dict, Any, Optional
import json
import csv
import os
from datetime import datetime
from pathlib import Path

from repomind.eval.runner import QueryResultRecord, AgentEvalRecord
from repomind.eval.statistical import (
    compare_systems_paired,
    aggregate_results_by_category,
    aggregate_results_by_system,
)


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records to JSONL file (one JSON object per line).

    Args:
        records: List of dictionaries to write
        path: Output file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read records from JSONL file.

    Args:
        path: Input file path

    Returns:
        List of dictionaries
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_csv(records: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write records to CSV file.

    Args:
        records: List of dictionaries
        path: Output file path
        fieldnames: Column names (auto-detected if None)
    """
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        # Collect all unique keys
        fieldnames = []
        seen = set()
        for r in records:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def flatten_metrics_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested metrics dict for CSV output."""
    flat = {k: v for k, v in record.items() if k != "metrics"}
    if "metrics" in record:
        for mk, mv in record["metrics"].items():
            flat[mk] = mv
    return flat


def generate_experiment_manifest(
    experiment_id: str,
    git_commit: str,
    config: Dict[str, Any],
    systems: List[str],
    metrics_computed: List[str],
) -> Dict[str, Any]:
    """Generate experiment manifest metadata.

    Args:
        experiment_id: Unique identifier for this run
        git_commit: Git commit hash
        config: Configuration dictionary
        systems: List of system names evaluated
        metrics_computed: List of metric names computed

    Returns:
        Manifest dictionary
    """
    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit,
        "config": config,
        "systems": systems,
        "metrics_computed": metrics_computed,
        "environment": {
            "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            "platform": __import__('sys').platform,
        }
    }


def write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """Write experiment manifest as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def generate_markdown_report(
    experiment_id: str,
    manifest: Dict[str, Any],
    retrieval_records: List[QueryResultRecord],
    agent_records: Optional[List[AgentEvalRecord]] = None,
    statistical_results: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a human-readable Markdown benchmark report.

    Args:
        experiment_id: Unique experiment identifier
        manifest: Experiment manifest
        retrieval_records: List of retrieval evaluation records
        agent_records: Optional list of agent evaluation records
        statistical_results: Optional dict of statistical comparisons

    Returns:
        Markdown formatted report string
    """
    lines = []

    # Header
    lines.append(f"# RepoMind Retrieval Benchmark — {experiment_id}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append(f"- **Repositories**: {manifest.get('config', {}).get('repos', 'N/A')}")
    lines.append(f"- **Embedding model**: {manifest.get('config', {}).get('embedding_model', 'N/A')}")
    lines.append(f"- **Vector top-k**: {manifest.get('config', {}).get('vector_top_k', 'N/A')}")
    lines.append(f"- **Graph depth**: {manifest.get('config', {}).get('graph_depth', 'N/A')}")
    fusion = manifest.get('config', {}).get('fusion_weights', {})
    lines.append(f"- **Fusion weights**: vector={fusion.get('vector', 'N/A')}, graph={fusion.get('graph', 'N/A')}")
    lines.append("")

    # Overall Results Table
    lines.append("## Overall Results")
    lines.append("")

    # Aggregate by system
    flat_records = [flatten_metrics_for_csv(r.to_dict()) for r in retrieval_records]
    sys_agg = aggregate_results_by_system(flat_records)

    if sys_agg:
        # Determine metric columns
        metric_cols = []
        for sys_name, metrics in sys_agg.items():
            for m in metrics:
                if m not in metric_cols and m not in ("top_k", "depth", "latency_ms"):
                    metric_cols.append(m)

        # Header
        header = ["System"] + metric_cols + ["Latency (ms)"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for sys_name, metrics in sys_agg.items():
            row = [sys_name]
            for m in metric_cols:
                row.append(f"{metrics.get(m, 0):.3f}")
            row.append(f"{metrics.get('latency_ms', 0):.1f}")
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # By Category
    lines.append("## By Category")
    lines.append("")

    cat_agg = aggregate_results_by_category(flat_records)
    if cat_agg:
        metric_cols = []
        for cat_name, metrics in cat_agg.items():
            for m in metrics:
                if m not in metric_cols and m not in ("top_k", "depth", "latency_ms"):
                    metric_cols.append(m)

        header = ["Category", "System"] + metric_cols
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for cat_name, metrics in cat_agg.items():
            # Note: This aggregates across systems within category
            # For system breakdown, we'd need a different aggregation
            row = [cat_name, "combined"]
            for m in metric_cols:
                row.append(f"{metrics.get(m, 0):.3f}")
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Statistical Significance
    if statistical_results:
        lines.append("## Statistical Significance")
        lines.append("")
        for metric, result in statistical_results.items():
            lines.append(f"### {metric}")
            lines.append(f"- Mean difference (A - B): {result['mean_difference']:.4f}")
            lines.append(f"- 95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            lines.append(f"- Wilcoxon p-value: {result['wilcoxon_p']:.4f}")
            lines.append(f"- Paired t-test p-value: {result['ttest_p']:.4f}")
            lines.append(f"- Cohen's d: {result['cohens_d']:.3f} ({result['effect_interpretation']})")
            lines.append("")

    # Agent Evaluation
    if agent_records:
        lines.append("## Agent Evaluation")
        lines.append("")

        # Aggregate agent metrics by system/provider
        agent_flat = [r.to_dict() for r in agent_records]
        agent_by_provider = aggregate_results_by_system(agent_flat, system_key="provider_name")

        if agent_by_provider:
            header = ["Provider", "Success Rate", "Avg Iterations", "Avg Tool Calls", "Graph Tool Usage %", "Redundant Calls %"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")

            for provider, metrics in agent_by_provider.items():
                success_rate = metrics.get("success", 0)
                iterations = metrics.get("iterations", 0)
                tool_calls = metrics.get("total_tool_calls", 0)
                graph_usage = metrics.get("used_graph_tool", 0) * 100
                redundant = metrics.get("redundant_tool_calls", 0)

                lines.append(f"| {provider} | {success_rate:.1%} | {iterations:.1f} | {tool_calls:.1f} | {graph_usage:.1f}% | {redundant:.1f} |")

    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Report generated: {datetime.utcnow().isoformat()}Z*")
    lines.append(f"*Git commit: {manifest.get('git_commit', 'unknown')}*")

    return "\n".join(lines)


def write_markdown_report(content: str, path: Path) -> None:
    """Write Markdown report to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def create_summary_csv(
    retrieval_records: List[QueryResultRecord],
    path: Path,
    statistical_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a comprehensive summary CSV with per-query, per-system, per-category rows.

    Args:
        retrieval_records: List of retrieval evaluation records
        path: Output CSV path
        statistical_results: Optional statistical comparison results
    """
    flat_records = [flatten_metrics_for_csv(r.to_dict()) for r in retrieval_records]

    # System-level summary
    sys_agg = aggregate_results_by_system(flat_records)
    cat_agg = aggregate_results_by_category(flat_records)

    all_rows = []

    # Overall system rows
    for sys_name, metrics in sys_agg.items():
        row = {
            "level": "overall",
            "system": sys_name,
            "category": "all",
            **metrics
        }
        all_rows.append(row)

    # Per-category rows (combined across systems)
    for cat_name, metrics in cat_agg.items():
        row = {
            "level": "category",
            "system": "combined",
            "category": cat_name,
            **metrics
        }
        all_rows.append(row)

    # Per-system per-category rows
    # Group by system and category
    by_sys_cat: Dict[tuple, List[Dict]] = {}
    for r in flat_records:
        key = (r.get("system", "unknown"), r.get("category", "unknown"))
        by_sys_cat.setdefault(key, []).append(r)

    for (sys_name, cat_name), cat_records in by_sys_cat.items():
        metrics = {}
        for k in cat_records[0].keys():
            if isinstance(cat_records[0].get(k), (int, float)) and k not in ("top_k", "depth"):
                vals = [rc.get(k, 0.0) for rc in cat_records]
                metrics[k] = sum(vals) / len(vals)

        row = {
            "level": "system_category",
            "system": sys_name,
            "category": cat_name,
            **metrics
        }
        all_rows.append(row)

    write_csv(all_rows, path)


def ensure_output_dir(base_dir: Path, experiment_id: str) -> Path:
    """Create output directory for an experiment run.

    Args:
        base_dir: Base output directory
        experiment_id: Unique experiment ID

    Returns:
        Path to experiment output directory
    """
    run_dir = base_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir