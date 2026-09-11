#!/usr/bin/env python3
"""CLI entry point for running RepoMind evaluation benchmarks.

Usage:
    python -m scripts.run_evaluation --repo <path> --system both --out <output_dir>
    python -m scripts.run_evaluation --repo <path> --system vector_only --questions <jsonl> --answers <jsonl>
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from repomind.ingestion.scanner import RepositoryScanner
from repomind.ingestion.file_reader import FileReader
from repomind.parsing.ast_parser import parse as parse_source
from repomind.chunking.chunker import CodeChunker
from repomind.vectorstore.store import VectorStore
from repomind.vectorstore.embedder import Embedder
from repomind.graph.builder import CodeGraphBuilder
from repomind.graph.traversal import GraphTraverser
from repomind.retrieval.hybrid import HybridRetriever
from repomind.eval.dataset import (
    load_combined_dataset,
    load_retrieval_dataset,
    load_answer_dataset,
    split_dataset,
)
from repomind.eval.runner import RetrievalRunner, AgentRunner
from repomind.eval.statistical import compare_systems_paired
from repomind.eval.reporter import (
    write_jsonl,
    write_manifest,
    generate_markdown_report,
    write_markdown_report,
    create_summary_csv,
    generate_experiment_manifest,
    ensure_output_dir,
    flatten_metrics_for_csv,
)
from repomind.agent.llm_provider import FreeLocalProvider
from repomind.config import Config


def load_corpus_provenance(repo_path: Path) -> Optional[dict]:
    """Read a corpus PROVENANCE.json if the evaluated repository ships one.

    A benchmark manifest that records only a local filesystem path cannot be
    reproduced by anyone else, so corpora pinned to an upstream commit carry a
    PROVENANCE.json that is copied verbatim into the manifest. Returns None for
    corpora without one (e.g. the synthetic fixture), which keeps this optional.
    """
    provenance_path = repo_path / "PROVENANCE.json"
    if not provenance_path.is_file():
        return None
    try:
        with open(provenance_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"      Warning: could not read {provenance_path}: {e}")
        return None


def build_repo_index(
    repo_path: Path,
    collection_name: str = "repomind_eval",
    top_k: int = 10,
    depth: int = 2,
) -> tuple:
    """Build a fresh vector + graph index for a repository.

    The collection is cleared before indexing so a benchmark run measures only
    this repository's chunks — a stale collection would silently contaminate
    precision/recall with chunks the ground truth never referenced.

    Returns:
        Tuple of (vector_store, hybrid_retriever, graph_traverser)
    """
    print(f"[1/5] Scanning repository: {repo_path}")
    scanner = RepositoryScanner(root_path=repo_path)
    python_files = scanner.scan()
    print(f"      Found {len(python_files)} Python files")

    print(f"[2/5] Reading, parsing, and chunking...")
    reader = FileReader(repo_root=repo_path)
    chunker = CodeChunker()

    all_chunks = []
    parsed_files = []
    for file_path in python_files:
        source_file = reader.read(file_path)
        parsed_file = parse_source(source_file.content, source_file.relative_path)
        if parsed_file.has_syntax_error:
            print(f"      Skipping {file_path}: {parsed_file.syntax_error_message}")
            continue
        all_chunks.extend(chunker.chunk(source_file, parsed_file))
        parsed_files.append(parsed_file)
    print(f"      Extracted {len(all_chunks)} chunks from {len(parsed_files)} files")

    print(f"[3/5] Embedding chunks into vector store...")
    embedder = Embedder()
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_dir=Config.VECTOR_STORE_PATH,
        embedder=embedder,
    )
    vector_store.clear()
    vector_store.add_chunks(all_chunks)
    print(f"      Vector store holds {vector_store.count()} chunks")

    print(f"[4/5] Building code dependency graph...")
    graph_builder = CodeGraphBuilder()
    graph_builder.build_graph(parsed_files, repo_root=repo_path)
    graph = graph_builder.graph
    print(f"      Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print(f"[5/5] Wiring hybrid retriever...")
    graph_traverser = GraphTraverser(graph)
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        graph_traverser=graph_traverser,
        default_top_k=top_k,
        default_depth=depth,
    )

    return vector_store, hybrid_retriever, graph_traverser


def run_retrieval_evaluation(
    repo_path: Path,
    questions_path: Path,
    answers_path: Optional[Path],
    systems: List[str],
    top_k: int,
    depth: int,
    output_dir: Path,
    experiment_id: str,
    git_commit: str = "unknown",
) -> None:
    """Run retrieval evaluation and produce outputs."""
    print(f"\n=== Running Retrieval Evaluation ===")
    print(f"Repository: {repo_path}")
    print(f"Questions: {questions_path}")
    print(f"Systems: {systems}")
    print(f"Top-k: {top_k}, Depth: {depth}")
    print(f"Output: {output_dir}")

    # Build indices. Both systems share this one index, so the only difference
    # between vector_only and hybrid is the graph traversal depth.
    vector_store, hybrid_retriever, _ = build_repo_index(
        repo_path, top_k=top_k, depth=depth
    )

    # Load dataset
    print("\nLoading evaluation dataset...")
    questions = load_combined_dataset(questions_path, answers_path)
    print(f"Loaded {len(questions)} evaluation questions")

    # Create runner
    runner = RetrievalRunner(hybrid_retriever)

    # Run evaluation
    print("\nEvaluating...")
    records = runner.evaluate_dataset(
        questions=questions,
        systems=systems,
        top_k=top_k,
        depth=depth,
    )
    print(f"Generated {len(records)} result records")

    # Convert to dict for serialization
    record_dicts = [r.to_dict() for r in records]

    # Write raw results
    raw_path = output_dir / "raw_results.jsonl"
    write_jsonl(record_dicts, raw_path)
    print(f"Raw results: {raw_path}")

    # Write summary CSV
    summary_path = output_dir / "summary.csv"
    create_summary_csv(records, summary_path)
    print(f"Summary CSV: {summary_path}")

    # Statistical comparison
    print("\nRunning statistical analysis...")
    statistical_results = {}
    if "vector_only" in systems and "hybrid" in systems:
        # compare_systems_paired reads metrics as top-level keys, so records must
        # be flattened first — passing the nested form silently yields 0.0 for
        # every metric and a fabricated "no difference" result.
        flat_dicts = [flatten_metrics_for_csv(r) for r in record_dicts]
        vector_records = [r for r in flat_dicts if r["system"] == "vector_only"]
        hybrid_records = [r for r in flat_dicts if r["system"] == "hybrid"]

        # Sort by question_id to ensure pairing
        vector_records.sort(key=lambda x: x["question_id"])
        hybrid_records.sort(key=lambda x: x["question_id"])

        # Compare on primary metrics. A is hybrid and B is vector_only so a
        # positive mean_difference means hybrid improved on the baseline.
        for metric in ["precision_at_10", "recall_at_10", "mrr", "ndcg_at_10", "r_precision"]:
            try:
                comp = compare_systems_paired(
                    results_a=hybrid_records,
                    results_b=vector_records,
                    metric_name=metric,
                )
                statistical_results[metric] = comp
                print(f"  {metric}: hybrid={comp['mean_a']:.4f} vector={comp['mean_b']:.4f} "
                      f"diff={comp['mean_difference']:+.4f}, "
                      f"p_wilcoxon={comp['wilcoxon_p']:.4f}, d={comp['cohens_d']:.3f}")
            except Exception as e:
                print(f"  Warning: Could not compute stats for {metric}: {e}")

    # Generate manifest
    manifest = generate_experiment_manifest(
        experiment_id=experiment_id,
        git_commit=git_commit,
        config={
            "repos": [str(repo_path)],
            # Upstream commit of the evaluated corpus. Without this the manifest
            # records only a local path, which is not enough to reproduce a run.
            "corpus_provenance": load_corpus_provenance(repo_path),
            "questions_file": str(questions_path),
            "answers_file": str(answers_path) if answers_path else None,
            "embedding_model": Config.EMBEDDING_MODEL,
            "vector_top_k": top_k,
            "graph_depth": depth,
            "fusion_weights": {"vector": 0.7, "graph": 0.3},
        },
        systems=systems,
        metrics_computed=list(statistical_results.keys()) + ["precision_at_5", "recall_at_5", "ndcg_at_5"],
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"Manifest: {manifest_path}")

    # Generate Markdown report
    print("\nGenerating Markdown report...")
    report = generate_markdown_report(
        experiment_id=experiment_id,
        manifest=manifest,
        retrieval_records=records,
        statistical_results=statistical_results,
    )
    report_path = output_dir / "report.md"
    write_markdown_report(report, report_path)
    print(f"Report: {report_path}")

    print(f"\n✅ Evaluation complete! Results in: {output_dir}")


def run_agent_evaluation(
    repo_path: Path,
    questions_path: Path,
    answers_path: Optional[Path],
    provider_name: str,
    max_iterations: int,
    output_dir: Path,
) -> None:
    """Run agent evaluation."""
    print(f"\n=== Running Agent Evaluation ===")
    print(f"Provider: {provider_name}")
    print(f"Max iterations: {max_iterations}")

    # Build indices
    vector_store, hybrid_retriever, _ = build_repo_index(repo_path)

    # Load dataset
    questions = load_combined_dataset(questions_path, answers_path)
    print(f"Loaded {len(questions)} evaluation questions")

    # Create runner
    runner = AgentRunner(vector_store, hybrid_retriever)

    # Select provider
    if provider_name == "free_local":
        provider = FreeLocalProvider()
    else:
        # Try to load Gemini
        try:
            from repomind.agent.llm_provider import GeminiProvider
            provider = GeminiProvider()
        except (ImportError, ValueError) as e:
            print(f"Warning: Could not load {provider_name}, falling back to FreeLocalProvider: {e}")
            provider = FreeLocalProvider()

    # Run evaluation
    print("\nEvaluating agent...")
    agent_records = runner.evaluate_dataset(
        questions=questions,
        provider=provider,
        max_iterations=max_iterations,
    )
    print(f"Generated {len(agent_records)} agent records")

    # Write agent results
    agent_path = output_dir / "agent_results.jsonl"
    write_jsonl([r.to_dict() for r in agent_records], agent_path)
    print(f"Agent results: {agent_path}")


def get_git_commit(repo_path: Path) -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="RepoMind Evaluation Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full retrieval benchmark (vector + hybrid)
  python -m scripts.run_evaluation --repo ./my_repo --system both --out eval_outputs

  # Run only vector-only baseline
  python -m scripts.run_evaluation --repo ./my_repo --system vector_only --out eval_outputs/vector

  # Run agent evaluation with free local provider
  python -m scripts.run_evaluation --repo ./my_repo --agent --provider free_local --out eval_outputs/agent

  # Custom top-k and depth
  python -m scripts.run_evaluation --repo ./my_repo --top-k 20 --depth 3 --out eval_outputs/deep
        """
    )

    # Required arguments
    parser.add_argument("--repo", type=Path, required=True,
                        help="Path to repository to evaluate")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for results")

    # Evaluation type
    parser.add_argument("--system", choices=["vector_only", "hybrid", "both"],
                        default="both", help="Retrieval systems to evaluate")
    parser.add_argument("--agent", action="store_true",
                        help="Run agent evaluation in addition to retrieval")

    # Dataset
    parser.add_argument("--questions", type=Path,
                        help="Path to questions JSONL (default: eval_data/questions/<repo>_v1.jsonl)")
    parser.add_argument("--answers", type=Path,
                        help="Path to answers JSONL (optional)")

    # Retrieval parameters
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top vector results")
    parser.add_argument("--depth", type=int, default=2,
                        help="Graph traversal depth for hybrid")

    # Agent parameters
    parser.add_argument("--provider", choices=["free_local", "gemini"],
                        default="free_local", help="LLM provider for agent")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max agent iterations per query")

    # Experiment metadata
    parser.add_argument("--experiment-id", type=str,
                        help="Custom experiment ID (default: auto-generated)")

    args = parser.parse_args()

    # Validate repo path
    repo_path = args.repo.resolve()
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)
    if not repo_path.is_dir():
        print(f"Error: Repository path is not a directory: {repo_path}")
        sys.exit(1)

    # Determine dataset paths
    if args.questions:
        questions_path = args.questions
    else:
        # Default to eval_data/questions/<repo_name>_v1.jsonl
        repo_name = repo_path.name
        questions_path = Path("eval_data") / "questions" / f"{repo_name}_v1.jsonl"
        if not questions_path.exists():
            # Try fixtures for testing
            questions_path = Path("eval_data") / "fixtures" / "tiny_questions.jsonl"

    if not questions_path.exists():
        print(f"Error: Questions file not found: {questions_path}")
        print("Specify with --questions or place in eval_data/questions/<repo>_v1.jsonl")
        sys.exit(1)

    answers_path = args.answers
    if answers_path and not answers_path.exists():
        print(f"Error: Answers file not found: {answers_path}")
        sys.exit(1)

    # Determine systems
    if args.system == "both":
        systems = ["vector_only", "hybrid"]
    else:
        systems = [args.system]

    # Generate experiment ID
    import uuid
    from datetime import datetime
    experiment_id = args.experiment_id or f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Get git commit
    git_commit = get_git_commit(repo_path)

    # Create output directory
    output_dir = ensure_output_dir(args.out.resolve(), experiment_id)

    # Run retrieval evaluation
    run_retrieval_evaluation(
        repo_path=repo_path,
        questions_path=questions_path,
        answers_path=answers_path,
        systems=systems,
        top_k=args.top_k,
        depth=args.depth,
        output_dir=output_dir,
        experiment_id=experiment_id,
        git_commit=git_commit,
    )

    # Run agent evaluation if requested
    if args.agent:
        agent_output_dir = output_dir / "agent"
        agent_output_dir.mkdir(exist_ok=True)
        run_agent_evaluation(
            repo_path=repo_path,
            questions_path=questions_path,
            answers_path=answers_path,
            provider_name=args.provider,
            max_iterations=args.max_iterations,
            output_dir=agent_output_dir,
        )

    print(f"\n🎉 All done! Experiment: {experiment_id}")
    print(f"   Results: {output_dir}")


if __name__ == "__main__":
    main()