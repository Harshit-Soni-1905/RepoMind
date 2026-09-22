"""CLI command implementations for RepoMind.

Implements the logic for the 'index' and 'ask' commands by orchestrating the
existing RepoMind subsystems (ingestion, parsing, chunking, vector store,
graph, retrieval, agent).

The functions here are deliberately thin: they wire subsystems together and
translate results into formatter output. All heavy lifting lives in the
individual subsystem modules built in Stages 1-7.
"""

import pickle
from pathlib import Path
from typing import List, Optional

import networkx as nx

from repomind.config import config
from repomind.cli.formatter import CLIFormatter
from repomind.ingestion.scanner import RepositoryScanner as RepoScanner
from repomind.ingestion.file_reader import FileReader
# Stage 2 parsing exposes a module-level `parse(source, filepath)` function,
# not a class, so we alias it for clarity at the call site.
from repomind.parsing.ast_parser import parse as parse_source
from repomind.parsing.notebook_parser import parse_notebook
from repomind.chunking.chunker import CodeChunker
from repomind.vectorstore.embedder import Embedder as CodeEmbedder
from repomind.vectorstore.store import VectorStore
from repomind.graph.builder import CodeGraphBuilder as GraphBuilder
from repomind.graph.traversal import GraphTraverser
from repomind.retrieval.hybrid import HybridRetriever
from repomind.agent.llm_provider import GeminiProvider, FreeLocalProvider, LLMProvider
from repomind.agent.tools import create_default_registry
from repomind.agent.agent import ReActAgent


# Shared collection name so `index` and `ask` read/write the same vector store.
DEFAULT_COLLECTION = "repomind_code"

# Per-repository persisted graph. NetworkX 3.x removed read/write_gpickle, so we
# persist the DiGraph ourselves with pickle (nodes carry dataclass/enum metadata,
# which pickles cleanly but is not JSON-serializable).
GRAPH_FILENAME = ".repomind_graph.pickle"


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    """Persist a code graph to disk via pickle."""
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.DiGraph:
    """Load a previously persisted code graph from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_repository_path(repo_path_str: str, formatter: CLIFormatter) -> Path:
    """Validate that the repository path exists, is a directory, and has supported code files.

    Args:
        repo_path_str: Raw string path to repository
        formatter: CLIFormatter for outputting errors

    Returns:
        Resolved absolute Path object

    Raises:
        ValueError: If validation fails
    """
    repo_path = Path(repo_path_str).resolve()

    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    # Confirm there is at least one supported code file to work with.
    scanner = RepoScanner(root_path=repo_path)
    code_files = scanner.scan()

    if not code_files:
        raise ValueError(f"No supported code files (.py, .ipynb) found in repository: {repo_path}")

    return repo_path


def run_index_command(
    repo_path_str: str,
    formatter: CLIFormatter,
    collection_name: str = DEFAULT_COLLECTION,
) -> bool:
    """Execute the 'index' CLI command.

    Scans, parses, chunks, embeds, and builds a graph index for a repository.

    Args:
        repo_path_str: Path to repository
        formatter: CLI output formatter
        collection_name: ChromaDB collection name

    Returns:
        True if indexing succeeds, False otherwise
    """
    try:
        formatter.info(f"Validating repository at: {repo_path_str}")
        repo_path = validate_repository_path(repo_path_str, formatter)

        formatter.info("Stage 1: Scanning code files...")
        scanner = RepoScanner(root_path=repo_path)
        code_files = scanner.scan()
        py_count = sum(1 for f in code_files if f.suffix == ".py")
        ipynb_count = sum(1 for f in code_files if f.suffix == ".ipynb")
        formatter.info(f"Found {len(code_files)} code files ({py_count} .py, {ipynb_count} .ipynb).")

        formatter.info("Stage 2 & 3: Reading, parsing AST, and chunking code...")
        reader = FileReader(repo_root=repo_path)
        chunker = CodeChunker()

        all_chunks = []
        parsed_files = []

        for file_path in code_files:
            try:
                source_file = reader.read(file_path)

                # Route .ipynb through NotebookParser, .py through standard parser
                if file_path.suffix == ".ipynb":
                    parsed_file, chunks = parse_notebook(source_file.content, source_file.relative_path)
                else:
                    parsed_file = parse_source(source_file.content, source_file.relative_path)

                    # parse_source reports syntax errors by returning a flagged
                    # ParsedFile rather than raising, so the except branch below
                    # never sees them. Without this check a broken file would be
                    # indexed as an empty file and silently degrade retrieval.
                    if parsed_file.has_syntax_error:
                        formatter.warning(
                            f"Skipping {file_path}: {parsed_file.syntax_error_message}"
                        )
                        continue
                    chunks = chunker.chunk(source_file, parsed_file)

                all_chunks.extend(chunks)
                parsed_files.append(parsed_file)
            except Exception as e:
                formatter.warning(f"Skipping {file_path} due to parsing/chunking error: {str(e)}")

        formatter.info(f"Extracted {len(all_chunks)} code chunks across {len(parsed_files)} files.")

        formatter.info("Stage 4: Generating embeddings and populating vector store...")
        embedder = CodeEmbedder()
        store = VectorStore(
            collection_name=collection_name,
            persist_dir=config.VECTOR_STORE_PATH,
            embedder=embedder,
        )
        store.add_chunks(all_chunks)
        formatter.info(f"Vector store populated. Total items: {store.count()}")

        formatter.info("Stage 5: Building code dependency graph...")
        graph_builder = GraphBuilder()
        graph_builder.build_graph(parsed_files, repo_root=repo_path)

        graph_path = repo_path / GRAPH_FILENAME
        save_graph(graph_builder.graph, graph_path)
        formatter.info(
            f"Graph store built with {graph_builder.graph.number_of_nodes()} nodes "
            f"and {graph_builder.graph.number_of_edges()} edges."
        )

        formatter.success(f"Successfully indexed repository: {repo_path}")
        return True

    except Exception as e:
        formatter.error(f"Indexing failed: {str(e)}", exception=e)
        return False


def run_ask_command(
    repo_path_str: str,
    query: str,
    formatter: CLIFormatter,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    top_k: Optional[int] = None,
    depth: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> bool:
    """Execute the 'ask' CLI command.

    Args:
        repo_path_str: Path to repository
        query: Developer's question about the repository
        formatter: CLI output formatter
        provider_name: LLM provider name override ("gemini", "free-local")
        model_name: LLM model name override
        top_k: Semantic search result count override
        depth: Graph traversal depth override
        max_iterations: Maximum agent ReAct loop iterations override

    Returns:
        True if the query succeeds and an answer is displayed, False otherwise
    """
    try:
        # 1. Validate inputs
        if not query or not query.strip():
            raise ValueError("Question cannot be empty.")

        repo_path = validate_repository_path(repo_path_str, formatter)

        # 2. Configuration precedence (CLI overrides > config defaults)
        active_provider = provider_name or config.LLM_PROVIDER
        active_model = model_name or config.GEMINI_MODEL
        active_top_k = top_k if top_k is not None else config.VECTOR_SEARCH_TOP_K
        active_depth = depth if depth is not None else config.GRAPH_TRAVERSAL_DEPTH
        active_max_iter = max_iterations if max_iterations is not None else config.AGENT_MAX_ITERATIONS

        formatter.info(f"Analyzing repo: {repo_path.name}")
        formatter.info(f"LLM Provider: {active_provider} (model: {active_model})")
        formatter.info(
            f"Retrieval config: top_k={active_top_k}, depth={active_depth}, "
            f"max_iterations={active_max_iter}"
        )

        # 3. Setup retrieval components
        embedder = CodeEmbedder()
        store = VectorStore(
            collection_name=DEFAULT_COLLECTION,
            persist_dir=config.VECTOR_STORE_PATH,
            embedder=embedder,
        )

        # Auto-index if the vector store has no data yet.
        if store.count() == 0:
            formatter.warning("Vector store is empty. Auto-indexing repository first...")
            success = run_index_command(str(repo_path), formatter)
            if not success:
                raise RuntimeError("Auto-indexing failed.")

        # Load the persisted graph if present; otherwise fall back to an empty
        # graph (retrieval degrades gracefully to vector-only).
        graph_builder = GraphBuilder()
        graph_path = repo_path / GRAPH_FILENAME
        if graph_path.exists():
            graph_builder.graph = load_graph(graph_path)

        traverser = GraphTraverser(graph_builder.graph)
        retriever = HybridRetriever(
            vector_store=store,
            graph_traverser=traverser,
            default_top_k=active_top_k,
            default_depth=active_depth,
        )

        # 4. Instantiate the LLM provider (free-tier Gemini or zero-cost local)
        provider: LLMProvider
        if active_provider == "gemini":
            provider = GeminiProvider(api_key=config.GEMINI_API_KEY, model_name=active_model)
            if not provider.is_available():
                formatter.warning(
                    "Gemini provider is unavailable (missing API key or package). "
                    "Falling back to 'free-local'."
                )
                provider = FreeLocalProvider()
        else:
            provider = FreeLocalProvider()

        # 5. Build the ToolRegistry and ReActAgent
        registry = create_default_registry(retriever, traverser, repo_root=repo_path)
        agent = ReActAgent(
            llm_provider=provider,
            tool_registry=registry,
            max_iterations=active_max_iter,
        )

        # 6. Run the agent analysis
        formatter.info(f"Starting ReAct agent query: '{query}'")
        state = agent.analyze(query)

        # Output intermediate steps when verbose/debug is enabled.
        if formatter.verbose or formatter.debug:
            for i, (tc, tr) in enumerate(zip(state.tool_calls, state.tool_results), 1):
                formatter.step(i, tc.name, f"Summary length: {len(tr.content)}")

        # 7. Print the final answer
        if state.final_answer:
            formatter.answer(state.final_answer)
            return True
        else:
            formatter.error("Agent failed to produce a final answer.")
            return False

    except Exception as e:
        formatter.error(f"Error analyzing query: {str(e)}", exception=e)
        return False
