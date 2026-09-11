#!/usr/bin/env python3
"""Corpus inspection utility: measure a repository through RepoMind's own pipeline.

Reports the structural statistics needed to judge whether a repository is a
viable retrieval benchmark, and dumps a chunk inventory so evaluation ground
truth can be authored against real chunk IDs instead of guessed ones.

This is a read-only measurement tool. It does not embed, retrieve, or evaluate.

Usage:
    python -m scripts.inspect_corpus --repo eval_data/repos/fastapi_corpus \
        --out eval_data/corpus_stats/fastapi
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from repomind.ingestion.scanner import RepositoryScanner
from repomind.ingestion.file_reader import FileReader
from repomind.parsing.ast_parser import parse as parse_source
from repomind.chunking.chunker import CodeChunker
from repomind.graph.builder import CodeGraphBuilder
from repomind.graph.models import EdgeType, NodeType


def build_corpus(repo_path: Path):
    """Scan, read, parse and chunk a repository, then build its code graph."""
    scanner = RepositoryScanner(root_path=repo_path)
    python_files = scanner.scan()

    reader = FileReader(repo_root=repo_path)
    chunker = CodeChunker()

    all_chunks = []
    parsed_files = []
    syntax_errors = []

    for file_path in python_files:
        source_file = reader.read(file_path)
        parsed_file = parse_source(source_file.content, source_file.relative_path)
        if parsed_file.has_syntax_error:
            syntax_errors.append((str(source_file.relative_path),
                                  parsed_file.syntax_error_message))
            continue
        all_chunks.extend(chunker.chunk(source_file, parsed_file))
        parsed_files.append(parsed_file)

    builder = CodeGraphBuilder()
    builder.build_graph(parsed_files, repo_root=repo_path)

    return python_files, parsed_files, all_chunks, builder.graph, syntax_errors


def edge_type_counts(graph: nx.DiGraph) -> Counter:
    """Count edges by EdgeType value."""
    counts = Counter()
    for _, _, data in graph.edges(data=True):
        et = data.get("edge_type")
        counts[et.value if isinstance(et, EdgeType) else str(et)] += 1
    return counts


def node_type_counts(graph: nx.DiGraph) -> Counter:
    """Count nodes by NodeType value."""
    counts = Counter()
    for _, data in graph.nodes(data=True):
        node_obj = data.get("node_obj")
        if node_obj is None:
            counts["<no node_obj>"] += 1
            continue
        nt = node_obj.node_type
        counts[nt.value if isinstance(nt, NodeType) else str(nt)] += 1
    return counts


def depth_characteristics(graph: nx.DiGraph, file_ids: set) -> dict:
    """Measure traversal distances the way HybridRetriever actually walks them.

    HybridRetriever._expand_graph_from_seed follows both successors and
    predecessors, so reachability is effectively undirected. Distances are
    therefore measured on the undirected projection.
    """
    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)

    stats = {
        "weakly_connected_components": len(components),
        "largest_component_nodes": len(components[0]) if components else 0,
        "isolated_nodes": sum(1 for n in undirected.nodes() if undirected.degree(n) == 0),
    }

    if not components:
        return stats

    largest = undirected.subgraph(components[0])
    stats["largest_component_diameter"] = nx.diameter(largest)
    stats["largest_component_avg_shortest_path"] = round(
        nx.average_shortest_path_length(largest), 3
    )

    # Import subgraph over file nodes only: the sole cross-file edge type.
    import_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if (d.get("edge_type") == EdgeType.IMPORTS)
    ]
    import_graph = nx.Graph()
    import_graph.add_nodes_from(file_ids)
    import_graph.add_edges_from(import_edges)
    imp_components = sorted(nx.connected_components(import_graph), key=len, reverse=True)
    stats["import_graph_files"] = import_graph.number_of_nodes()
    stats["import_graph_edges"] = import_graph.number_of_edges()
    stats["import_graph_components"] = len(imp_components)
    stats["import_graph_largest_component"] = len(imp_components[0]) if imp_components else 0
    if imp_components:
        imp_largest = import_graph.subgraph(imp_components[0])
        stats["import_graph_diameter"] = nx.diameter(imp_largest)
        stats["import_graph_avg_shortest_path"] = round(
            nx.average_shortest_path_length(imp_largest), 3
        )

    # Distribution of shortest-path lengths between symbol nodes in DIFFERENT
    # files. This is what depth>=3 is supposed to unlock, so measure it rather
    # than assume it.
    symbol_ids = [n for n in graph.nodes() if n not in file_ids]
    sample = symbol_ids[:400]  # deterministic slice; keeps all-pairs affordable
    cross_file_dists = Counter()
    for src in sample:
        src_file = src.split("::")[0]
        lengths = nx.single_source_shortest_path_length(undirected, src, cutoff=8)
        for dst, dist in lengths.items():
            if dst in file_ids or dst == src:
                continue
            if dst.split("::")[0] != src_file:
                cross_file_dists[dist] += 1
    stats["cross_file_symbol_distance_distribution"] = dict(sorted(cross_file_dists.items()))
    stats["cross_file_sample_symbols"] = len(sample)

    # Hub detection: highest-degree file nodes distort corroboration.
    file_degrees = sorted(
        ((n, undirected.degree(n)) for n in file_ids if n in undirected),
        key=lambda kv: -kv[1],
    )
    stats["top_degree_files"] = [{"file": n, "degree": d} for n, d in file_degrees[:10]]

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a repository corpus.")
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    repo_path = args.repo.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting: {repo_path}")
    files, parsed, chunks, graph, syntax_errors = build_corpus(repo_path)

    file_ids = {pf.filepath.as_posix() for pf in parsed}
    e_counts = edge_type_counts(graph)
    n_counts = node_type_counts(graph)

    print("\n=== CORPUS ===")
    print(f"Python source files scanned : {len(files)}")
    print(f"Files parsed successfully    : {len(parsed)}")
    print(f"Files with syntax errors     : {len(syntax_errors)}")
    print(f"AST chunks produced          : {len(chunks)}")

    chunk_types = Counter(c.chunk_type for c in chunks)
    print("\nChunk types:")
    for k, v in sorted(chunk_types.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v}")

    print("\n=== GRAPH ===")
    print(f"Nodes : {graph.number_of_nodes()}")
    print(f"Edges : {graph.number_of_edges()}")
    print("\nNode types:")
    for k, v in sorted(n_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v}")
    print("\nEdge types:")
    for k, v in sorted(e_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v}")

    print("\n=== DEPTH CHARACTERISTICS ===")
    depth = depth_characteristics(graph, file_ids)
    for k, v in depth.items():
        if k == "top_degree_files":
            print(f"  {k}:")
            for entry in v:
                print(f"      {entry['degree']:>4}  {entry['file']}")
        else:
            print(f"  {k}: {v}")

    if syntax_errors:
        print("\n=== SYNTAX ERRORS ===")
        for path, msg in syntax_errors:
            print(f"  {path}: {msg}")

    # Dump chunk inventory for ground-truth authoring.
    inventory = [
        {
            "chunk_id": c.chunk_id,
            "filepath": c.filepath.as_posix(),
            "chunk_type": c.chunk_type,
            "symbol_name": c.symbol_name,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "has_docstring": bool(c.docstring),
        }
        for c in chunks
    ]
    (out_dir / "chunk_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )

    # Dump resolved import edges: the only cross-file structure available.
    imports = defaultdict(list)
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type") == EdgeType.IMPORTS:
            imports[u].append(v)
    (out_dir / "import_edges.json").write_text(
        json.dumps({k: sorted(v) for k, v in sorted(imports.items())}, indent=2),
        encoding="utf-8",
    )

    stats = {
        "repo_path": str(repo_path),
        "source_files_scanned": len(files),
        "files_parsed": len(parsed),
        "files_with_syntax_errors": len(syntax_errors),
        "ast_chunks": len(chunks),
        "chunk_types": dict(chunk_types),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "node_types": dict(n_counts),
        "edge_types": dict(e_counts),
        "depth_characteristics": depth,
    }
    (out_dir / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(f"\nArtifacts written to: {out_dir}")
    print("  corpus_stats.json, chunk_inventory.json, import_edges.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
