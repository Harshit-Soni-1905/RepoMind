"""Validate an evaluation dataset against a corpus before benchmarking.

This checks that the dataset is *structurally* sound: every ground-truth chunk id
really exists in the corpus, ids are unique, questions and answers correspond
1:1, and the category mix matches the plan. It deliberately does NOT run
retrieval. Measuring whether questions are "answerable" and then editing them
would be tuning the benchmark to a desired result.

It also reports how many ground-truth chunks are graph nodes. That number is
diagnostic only — chunk types such as module_context have no graph node, so they
can never receive graph corroboration. Reporting it keeps the asymmetry visible
instead of letting it silently shape the comparison.

Usage:
    python scripts/validate_dataset.py --repo eval_data/repos/fastapi_corpus \
        --questions eval_data/questions/fastapi_corpus_v1.jsonl \
        --answers eval_data/answers/fastapi_corpus_v1.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from repomind.ingestion.scanner import RepositoryScanner
from repomind.ingestion.file_reader import FileReader
from repomind.parsing.ast_parser import parse as parse_source
from repomind.chunking.chunker import CodeChunker
from repomind.graph.builder import CodeGraphBuilder
from repomind.eval.dataset import load_combined_dataset

# Target mix after removing call_chain (20% in docs/stage9_plan.md) and scaling
# the five remaining categories proportionally by 100/80. Proportional scaling is
# used deliberately: reallocating the freed share toward cross_file or
# dependency_trace would favour the hybrid system by construction.
TARGET_SHARE = {
    "symbol_lookup": 0.2500,
    "dependency_trace": 0.1875,
    "cross_file": 0.2500,
    "behavioral": 0.1875,
    "negative": 0.1250,
}


def build_corpus(repo_path: Path):
    """Chunk and graph the corpus with the same components the benchmark uses."""
    scanner = RepositoryScanner(root_path=repo_path)
    python_files = scanner.scan()
    reader = FileReader(repo_root=repo_path)
    chunker = CodeChunker()

    chunks, parsed_files = [], []
    for file_path in python_files:
        source_file = reader.read(file_path)
        parsed = parse_source(source_file.content, source_file.relative_path)
        if parsed.has_syntax_error:
            continue
        chunks.extend(chunker.chunk(source_file, parsed))
        parsed_files.append(parsed)

    builder = CodeGraphBuilder()
    builder.build_graph(parsed_files, repo_root=repo_path)
    return chunks, builder.graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate an evaluation dataset.")
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--answers", type=Path)
    args = ap.parse_args()

    repo_path = args.repo.resolve()
    print(f"Corpus  : {repo_path}")
    print(f"Questions: {args.questions}")
    print(f"Answers  : {args.answers}\n")

    chunks, graph = build_corpus(repo_path)
    chunk_ids = {c.chunk_id for c in chunks}
    chunk_by_id = {c.chunk_id: c for c in chunks}
    filepaths = {str(c.filepath).replace("\\", "/") for c in chunks}
    graph_nodes = set(graph.nodes())
    print(f"Corpus holds {len(chunk_ids)} chunks across {len(filepaths)} files, "
          f"{len(graph_nodes)} graph nodes\n")

    errors: list[str] = []
    warnings: list[str] = []

    # Loading through the real loader proves the harness can consume the files.
    questions = load_combined_dataset(args.questions, args.answers)
    print(f"Loader accepted {len(questions)} questions\n")

    raw_q = [json.loads(line) for line in
             args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_q) != len(questions):
        errors.append(f"loader returned {len(questions)} of {len(raw_q)} question lines")

    ids = [q["question_id"] for q in raw_q]
    dupes = [qid for qid, n in Counter(ids).items() if n > 1]
    if dupes:
        errors.append(f"duplicate question_ids: {dupes}")

    if args.answers:
        raw_a = [json.loads(line) for line in
                 args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
        a_ids = {a["question_id"] for a in raw_a}
        if a_ids != set(ids):
            errors.append(f"question/answer id mismatch: "
                          f"questions-only={sorted(set(ids) - a_ids)}, "
                          f"answers-only={sorted(a_ids - set(ids))}")
        for a in raw_a:
            if not a.get("reference_answer"):
                errors.append(f"{a['question_id']}: empty reference_answer")

    graph_visible = graph_invisible = 0
    invisible_types: Counter = Counter()

    for q in raw_q:
        qid = q["question_id"]
        gt = q.get("relevant_chunk_ids", [])
        is_negative = q.get("category") == "negative"

        if is_negative and gt:
            errors.append(f"{qid}: negative question must have empty ground truth")
        if not is_negative and not gt:
            errors.append(f"{qid}: non-negative question has empty ground truth")

        for cid in gt:
            if cid not in chunk_ids:
                errors.append(f"{qid}: ground-truth chunk does not exist: {cid}")
                continue
            if cid in graph_nodes:
                graph_visible += 1
            else:
                graph_invisible += 1
                invisible_types[chunk_by_id[cid].chunk_type] += 1

        # relevant_filepaths should be exactly the files of the ground-truth chunks.
        declared = set(q.get("relevant_filepaths", []))
        derived = {str(chunk_by_id[c].filepath).replace("\\", "/") for c in gt if c in chunk_by_id}
        if declared != derived:
            errors.append(f"{qid}: relevant_filepaths {sorted(declared)} != "
                          f"files of ground-truth chunks {sorted(derived)}")
        for fp in declared:
            if fp not in filepaths:
                errors.append(f"{qid}: filepath not in corpus: {fp}")

    print("=== CATEGORY MIX ===")
    counts = Counter(q.get("category", "general") for q in raw_q)
    total = len(raw_q)
    for cat in sorted(TARGET_SHARE, key=lambda c: -TARGET_SHARE[c]):
        want = TARGET_SHARE[cat] * total
        got = counts.get(cat, 0)
        flag = "" if abs(got - want) < 1.0 else "   <-- off target"
        print(f"  {cat:<18} {got:>3}  ({got / total:6.2%})  target {want:5.2f}{flag}")
    for cat in counts:
        if cat not in TARGET_SHARE:
            errors.append(f"unexpected category present: {cat}")
    print(f"  {'TOTAL':<18} {total:>3}")

    print("\n=== DIFFICULTY MIX ===")
    for diff, n in sorted(Counter(q.get("difficulty", "medium") for q in raw_q).items()):
        print(f"  {diff:<18} {n:>3}")

    print("\n=== GROUND-TRUTH SPAN ===")
    non_neg = [q for q in raw_q if q.get("category") != "negative"]
    multi_chunk = sum(1 for q in non_neg if len(q["relevant_chunk_ids"]) > 1)
    multi_file = sum(1 for q in non_neg if len(set(q["relevant_filepaths"])) > 1)
    sizes = [len(q["relevant_chunk_ids"]) for q in non_neg]
    print(f"  answerable questions          : {len(non_neg)}")
    print(f"  multi-chunk ground truth      : {multi_chunk}")
    print(f"  multi-FILE (cross-file) truth : {multi_file}")
    print(f"  chunks per question  min/mean/max: "
          f"{min(sizes)}/{sum(sizes) / len(sizes):.2f}/{max(sizes)}")
    print(f"  total ground-truth chunk refs : {sum(sizes)}")
    print(f"  distinct chunks referenced    : "
          f"{len({c for q in non_neg for c in q['relevant_chunk_ids']})}")

    print("\n=== GRAPH VISIBILITY OF GROUND TRUTH (diagnostic) ===")
    print(f"  refs that ARE graph nodes  : {graph_visible}")
    print(f"  refs with NO graph node    : {graph_invisible}")
    for ctype, n in invisible_types.most_common():
        print(f"      {ctype:<16} {n}")
    if graph_invisible:
        warnings.append(
            f"{graph_invisible} ground-truth refs have no graph node "
            f"({dict(invisible_types)}); graph corroboration cannot reach them"
        )

    print("\n=== RESULT ===")
    for w in warnings:
        print(f"  WARNING: {w}")
    if errors:
        print(f"  {len(errors)} ERROR(S):")
        for e in errors:
            print(f"    - {e}")
        return 1
    print("  All structural checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
