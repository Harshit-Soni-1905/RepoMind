"""Build and persist the vector + graph index for a corpus, then report counts.

Uses the same component sequence as scripts/run_evaluation.py:build_repo_index so
that what gets validated here is what the benchmark will actually use.

Usage:
    python scripts/build_index.py --repo eval_data/repos/fastapi_corpus \
        --collection repomind_eval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_evaluation import build_repo_index


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a corpus index.")
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--collection", default="repomind_eval")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    vector_store, retriever, traverser = build_repo_index(
        repo_path=args.repo.resolve(),
        collection_name=args.collection,
        top_k=args.top_k,
        depth=args.depth,
    )

    print("\n=== INDEX SUMMARY ===")
    print(f"Vector store chunks : {vector_store.count()}")
    print(f"Graph nodes         : {traverser.graph.number_of_nodes()}")
    print(f"Graph edges         : {traverser.graph.number_of_edges()}")
    print(f"Retriever top_k     : {retriever.default_top_k}")
    print(f"Retriever depth     : {retriever.default_depth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
