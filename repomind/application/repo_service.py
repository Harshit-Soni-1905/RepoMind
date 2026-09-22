"""Repository service for indexing and managing code repositories.

This service extracts the core indexing orchestration logic from cli/commands.py
(Stage 8) to make it reusable by both the CLI and the new web API (Stage 10).
"""

import gc
import pickle
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Callable, List
from datetime import datetime
import uuid

import networkx as nx

from repomind.application.models import (
    JobStatus,
    IndexingProgress,
    RepositoryInfo,
)
from repomind.config import config
from repomind.ingestion.scanner import RepositoryScanner
from repomind.ingestion.file_reader import FileReader
from repomind.parsing.ast_parser import parse as parse_source
from repomind.parsing.notebook_parser import parse_notebook
from repomind.chunking.chunker import CodeChunker
from repomind.vectorstore.embedder import Embedder
from repomind.vectorstore.store import VectorStore
from repomind.graph.builder import CodeGraphBuilder
from repomind.utils import log_memory


GRAPH_FILENAME = ".repomind_graph.pickle"
MAX_FILES_LIMIT = 1000
MAX_REPO_SIZE_MB = 50


class RepositoryService:
    """Service for managing repository indexing lifecycle."""

    def __init__(
        self,
        storage_root: Optional[Path] = None,
        vector_store_path: Optional[Path] = None,
    ):
        """Initialize the repository service.

        Args:
            storage_root: Root directory for storing cloned repositories
            vector_store_path: Path for ChromaDB vector store persistence
        """
        self.storage_root = storage_root or Path(tempfile.gettempdir()) / "repomind_repos"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.vector_store_path = vector_store_path or Path(config.VECTOR_STORE_PATH)

    def generate_repo_id(self) -> str:
        """Generate a unique repository identifier."""
        return str(uuid.uuid4())

    def validate_repository_url(self, repo_url: str) -> tuple[bool, Optional[str]]:
        """Validate a repository URL for security and format.

        Args:
            repo_url: Repository URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Allow only GitHub and GitLab
        allowed_hosts = ["github.com", "gitlab.com"]

        if not repo_url.startswith(("https://", "http://")):
            return False, "Only HTTP/HTTPS URLs are supported"

        for host in allowed_hosts:
            if host in repo_url:
                return True, None

        return False, f"Only GitHub and GitLab repositories are supported"

    def clone_repository(
        self,
        repo_url: str,
        repo_id: str,
        branch: str = "main",
        progress_callback: Optional[Callable[[IndexingProgress], None]] = None,
    ) -> tuple[bool, Optional[Path], Optional[str]]:
        """Clone a git repository.

        Args:
            repo_url: URL of the repository to clone
            repo_id: Unique repository identifier
            branch: Branch to clone
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success, local_path, error_message)
        """
        clone_path = self.storage_root / repo_id
        log_memory("BEFORE_CLONE", f"repo_id={repo_id}")

        if progress_callback:
            progress_callback(IndexingProgress(
                repo_id=repo_id,
                job_id=repo_id,
                status=JobStatus.CLONING,
                progress=5,
                message="Cloning repository...",
            ))

        try:
            # Clone with depth=1 for speed
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth", "1",
                    "--branch", branch,
                    "--single-branch",
                    repo_url,
                    str(clone_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )

            # Verify the clone succeeded and has supported code files
            if not clone_path.exists():
                return False, None, "Repository clone failed"

            has_supported_files = any(
                any(clone_path.rglob(f"*{ext}")) for ext in config.SUPPORTED_EXTENSIONS
            )
            if not has_supported_files:
                return False, None, "No supported code files (.py, .ipynb) found in repository"

            log_memory("AFTER_CLONE", f"repo_id={repo_id}")
            return True, clone_path, None

        except subprocess.TimeoutExpired:
            return False, None, "Repository clone timed out (5 minute limit)"
        except subprocess.CalledProcessError as e:
            return False, None, f"Git clone failed: {e.stderr}"
        except Exception as e:
            return False, None, f"Clone error: {str(e)}"

    def validate_repository_size(self, repo_path: Path) -> tuple[bool, Optional[str]]:
        """Validate that the repository meets size constraints.

        Args:
            repo_path: Path to the repository

        Returns:
            Tuple of (is_valid, error_message)
        """
        code_files = []
        for ext in config.SUPPORTED_EXTENSIONS:
            code_files.extend(repo_path.rglob(f"*{ext}"))

        if len(code_files) > MAX_FILES_LIMIT:
            return False, f"Repository exceeds {MAX_FILES_LIMIT} file limit ({len(code_files)} files)"

        # Calculate total size
        total_size = sum(f.stat().st_size for f in code_files if f.is_file())
        total_mb = total_size / (1024 * 1024)

        if total_mb > MAX_REPO_SIZE_MB:
            return False, f"Repository exceeds {MAX_REPO_SIZE_MB}MB limit ({total_mb:.1f}MB)"

        return True, None

    def index_repository(
        self,
        repo_path: Path,
        repo_id: str,
        progress_callback: Optional[Callable[[IndexingProgress], None]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Index a repository by running the complete ingestion pipeline.

        Args:
            repo_path: Path to the repository
            repo_id: Unique repository identifier
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Use repo-specific collection name for isolation
            collection_name = f"repomind_{repo_id}"

            # Stage 1: Scanning
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.SCANNING,
                    progress=15,
                    message="Scanning code files...",
                ))

            scanner = RepositoryScanner(root_path=repo_path)
            code_files = scanner.scan()
            log_memory("AFTER_SCAN", f"files={len(code_files)}")

            if not code_files:
                return False, "No supported code files (.py, .ipynb) found in repository"

            # Stage 2 & 3: Parsing and Chunking
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.PARSING,
                    progress=30,
                    message=f"Parsing {len(code_files)} code files...",
                ))

            # Initialize VectorStore before ingestion loop
            embedder = Embedder()
            log_memory("BEFORE_VECTORSTORE_INIT")
            store = VectorStore(
                collection_name=collection_name,
                persist_dir=str(self.vector_store_path),
                embedder=embedder,
            )
            log_memory("AFTER_VECTORSTORE_INIT")

            # Clear any existing data for this collection
            log_memory("BEFORE_STORE_CLEAR")
            try:
                store.clear()
            except Exception:
                pass
            log_memory("AFTER_STORE_CLEAR")

            reader = FileReader(repo_root=repo_path)
            chunker = CodeChunker()
            parsed_files = []
            total_chunks = 0
            total_files = len(code_files)

            log_memory("BEFORE_FIRST_FILE", f"total_files={total_files}")

            # Stream chunk generation and vector store insertion per-file
            # to avoid accumulating all chunks in memory across the entire repository
            for idx, file_path in enumerate(code_files, 1):
                try:
                    source_file = reader.read(file_path)

                    # Route .ipynb through NotebookParser, .py through standard parser
                    if file_path.suffix == ".ipynb":
                        parsed_file, chunks = parse_notebook(source_file.content, source_file.relative_path)
                    else:
                        parsed_file = parse_source(source_file.content, source_file.relative_path)
                        if parsed_file.has_syntax_error:
                            continue
                        chunks = chunker.chunk(source_file, parsed_file)

                    if chunks:
                        log_memory("BEFORE_ADD_CHUNKS", f"chunks={len(chunks)}")
                        store.add_chunks(chunks)
                        total_chunks += len(chunks)
                        log_memory("AFTER_ADD_CHUNKS", f"chunks={len(chunks)} | total_chunks={total_chunks}")

                    parsed_files.append(parsed_file)

                    if idx % 10 == 0 or idx == total_files:
                        log_memory(
                            "FILE_PROCESSING",
                            f"file_num={idx} | total_files={total_files} | total_chunks={total_chunks}",
                        )
                except Exception:
                    continue

            if total_chunks == 0:
                return False, "No code chunks extracted from repository"

            log_memory("VECTOR_INGESTION_COMPLETE", f"total_chunks={total_chunks}")

            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.CHUNKING,
                    progress=50,
                    message=f"Extracted {total_chunks} code chunks",
                ))

            # Stage 4: Embedding / Vector indexing complete
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.EMBEDDING,
                    progress=65,
                    message=f"Vector indexing complete ({total_chunks} chunks embedded)",
                ))

            # Trigger garbage collection after vector store ingestion
            gc.collect()
            log_memory("AFTER_GC_COLLECT", "context=post_vector_ingestion")

            # Stage 5: Graph Building
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.GRAPH_BUILDING,
                    progress=85,
                    message="Building code dependency graph...",
                ))

            log_memory("BEFORE_GRAPH_BUILD", f"parsed_files={len(parsed_files)}")
            graph_builder = CodeGraphBuilder()
            graph_builder.build_graph(parsed_files, repo_root=repo_path)
            node_count = graph_builder.graph.number_of_nodes()
            edge_count = graph_builder.graph.number_of_edges()
            log_memory("GRAPH_BUILT", f"nodes={node_count} | edges={edge_count}")

            graph_path = repo_path / GRAPH_FILENAME
            log_memory("BEFORE_GRAPH_PICKLE")
            with open(graph_path, "wb") as f:
                pickle.dump(graph_builder.graph, f)
            log_memory("AFTER_GRAPH_PICKLE")

            del graph_builder
            del parsed_files
            gc.collect()
            log_memory("AFTER_GC_COLLECT", "context=post_graph_build")

            # Complete
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.READY,
                    progress=100,
                    message=f"Indexing complete: {store.count()} chunks, "
                            f"{node_count} nodes",
                ))

            log_memory("INDEXING_READY", f"total_chunks={total_chunks} | nodes={node_count}")
            return True, None

        except Exception as e:
            log_memory("INDEXING_ERROR", f"error_type={type(e).__name__}")
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.FAILED,
                    progress=0,
                    message="Indexing failed",
                    error=str(e),
                ))
            return False, str(e)
        finally:
            gc.collect()
            log_memory("INDEXING_FINALLY")

    def cleanup_repository(self, repo_id: str) -> None:
        """Remove a cloned repository and its data.

        Args:
            repo_id: Repository identifier to clean up
        """
        repo_path = self.storage_root / repo_id
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

    def get_collection_name(self, repo_id: str) -> str:
        """Get the ChromaDB collection name for a repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Collection name
        """
        return f"repomind_{repo_id}"

    def load_graph(self, repo_path: Path) -> Optional[nx.DiGraph]:
        """Load the persisted graph for a repository.

        Args:
            repo_path: Path to the repository

        Returns:
            The loaded graph, or None if not found
        """
        graph_path = repo_path / GRAPH_FILENAME
        if not graph_path.exists():
            return None

        try:
            with open(graph_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
