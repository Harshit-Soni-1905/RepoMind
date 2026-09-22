"""Repository service for indexing and managing code repositories.

This service extracts the core indexing orchestration logic from cli/commands.py
(Stage 8) to make it reusable by both the CLI and the new web API (Stage 10).
"""

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
                        if parsed_file.has_syntax_error:
                            continue
                        chunks = chunker.chunk(source_file, parsed_file)

                    all_chunks.extend(chunks)
                    parsed_files.append(parsed_file)
                except Exception:
                    continue

            if not all_chunks:
                return False, "No code chunks extracted from repository"

            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.CHUNKING,
                    progress=50,
                    message=f"Extracted {len(all_chunks)} code chunks",
                ))

            # Stage 4: Embedding
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.EMBEDDING,
                    progress=65,
                    message="Generating embeddings...",
                ))

            embedder = Embedder()
            store = VectorStore(
                collection_name=collection_name,
                persist_dir=str(self.vector_store_path),
                embedder=embedder,
            )

            # Clear any existing data for this collection
            try:
                store.clear()
            except Exception:
                pass

            store.add_chunks(all_chunks)

            # Stage 5: Graph Building
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.GRAPH_BUILDING,
                    progress=85,
                    message="Building code dependency graph...",
                ))

            graph_builder = CodeGraphBuilder()
            graph_builder.build_graph(parsed_files, repo_root=repo_path)

            graph_path = repo_path / GRAPH_FILENAME
            with open(graph_path, "wb") as f:
                pickle.dump(graph_builder.graph, f)

            # Complete
            if progress_callback:
                progress_callback(IndexingProgress(
                    repo_id=repo_id,
                    job_id=repo_id,
                    status=JobStatus.READY,
                    progress=100,
                    message=f"Indexing complete: {store.count()} chunks, "
                            f"{graph_builder.graph.number_of_nodes()} nodes",
                ))

            return True, None

        except Exception as e:
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
