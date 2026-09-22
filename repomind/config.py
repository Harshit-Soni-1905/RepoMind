"""Central configuration for RepoMind.

All configurable parameters, model choices, and paths are defined here.
Configuration can be overridden via environment variables.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """RepoMind configuration with sensible defaults.

    Configuration priority:
    1. Environment variables (highest priority)
    2. Explicit config file (future feature)
    3. Defaults defined here (lowest priority)
    """

    # ===== LLM Configuration =====
    # Provider: "gemini", "openai", "anthropic" (currently only gemini supported)
    LLM_PROVIDER: str = os.getenv("REPOMIND_LLM_PROVIDER", "gemini")

    # Gemini configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # Free tier model

    # Generic LLM parameters
    LLM_TEMPERATURE: float = float(os.getenv("REPOMIND_LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("REPOMIND_LLM_MAX_TOKENS", "2048"))

    # ===== Embedding Configuration =====
    EMBEDDING_MODEL: str = os.getenv(
        "REPOMIND_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, fast, local
    )
    EMBEDDING_DIMENSION: int = 384  # Matches all-MiniLM-L6-v2
    # Batch size for embedding generation to control memory usage
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("REPOMIND_EMBEDDING_BATCH_SIZE", "32"))

    # ===== Vector Store Configuration =====
    VECTOR_STORE_PATH: Path = Path(
        os.getenv("REPOMIND_VECTOR_STORE_PATH", "./.repomind_data/vectorstore")
    )
    VECTOR_SEARCH_TOP_K: int = int(os.getenv("REPOMIND_VECTOR_TOP_K", "10"))

    # ===== Graph Store Configuration =====
    GRAPH_STORE_PATH: Path = Path(
        os.getenv("REPOMIND_GRAPH_STORE_PATH", "./.repomind_data/graph.gpickle")
    )
    GRAPH_TRAVERSAL_DEPTH: int = int(os.getenv("REPOMIND_GRAPH_DEPTH", "2"))

    # ===== Chunking Configuration =====
    # Maximum lines per chunk (for very large functions/classes)
    MAX_CHUNK_LINES: int = int(os.getenv("REPOMIND_MAX_CHUNK_LINES", "150"))
    # Whether to include imports in module-level chunks
    INCLUDE_IMPORTS_IN_CHUNKS: bool = os.getenv(
        "REPOMIND_INCLUDE_IMPORTS", "true"
    ).lower() == "true"

    # ===== Ingestion Configuration =====
    # Supported file extensions for code ingestion
    SUPPORTED_EXTENSIONS: set = {".py", ".ipynb"}
    # Maximum file size to process (in bytes)
    MAX_FILE_SIZE: int = int(os.getenv("REPOMIND_MAX_FILE_SIZE", str(1024 * 1024)))  # 1 MB
    # Directories to always skip
    SKIP_DIRECTORIES: set = {
        "__pycache__", ".git", ".venv", "venv", "env",
        "node_modules", ".tox", ".pytest_cache", "htmlcov"
    }

    # ===== Agent Configuration =====
    # Maximum ReAct loop iterations before forcing an answer
    AGENT_MAX_ITERATIONS: int = int(os.getenv("REPOMIND_AGENT_MAX_ITER", "10"))
    # Maximum tokens of context to send to the agent (approximate)
    AGENT_MAX_CONTEXT_TOKENS: int = int(
        os.getenv("REPOMIND_AGENT_MAX_CONTEXT", "8000")
    )

    # ===== Retrieval Configuration =====
    # Strategy: "vector", "graph", "hybrid"
    RETRIEVAL_STRATEGY: str = os.getenv("REPOMIND_RETRIEVAL_STRATEGY", "hybrid")
    # Weight for vector results in hybrid retrieval (0.0 to 1.0)
    HYBRID_VECTOR_WEIGHT: float = float(
        os.getenv("REPOMIND_HYBRID_VECTOR_WEIGHT", "0.6")
    )
    # Weight for graph results in hybrid retrieval
    HYBRID_GRAPH_WEIGHT: float = float(
        os.getenv("REPOMIND_HYBRID_GRAPH_WEIGHT", "0.4")
    )

    # ===== Evaluation Configuration =====
    EVAL_DATA_PATH: Path = Path(
        os.getenv("REPOMIND_EVAL_DATA_PATH", "./eval_data")
    )

    # ===== API Configuration (Stage 10) =====
    API_HOST: str = os.getenv("REPOMIND_API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("REPOMIND_API_PORT", "8000"))
    DATA_DIR: Path = Path(
        os.getenv("REPOMIND_DATA_DIR", "./.repomind_data")
    )
    MAX_INDEXING_WORKERS: int = int(os.getenv("REPOMIND_MAX_WORKERS", "4"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("REPOMIND_RATE_LIMIT", "60"))

    @classmethod
    def validate(cls) -> None:
        """Validate configuration and raise helpful errors if misconfigured."""
        if cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY environment variable must be set when using Gemini provider. "
                "Get your free API key at https://makersuite.google.com/app/apikey"
            )

        if cls.HYBRID_VECTOR_WEIGHT + cls.HYBRID_GRAPH_WEIGHT != 1.0:
            raise ValueError(
                f"Hybrid weights must sum to 1.0, got vector={cls.HYBRID_VECTOR_WEIGHT}, "
                f"graph={cls.HYBRID_GRAPH_WEIGHT}"
            )

        if cls.RETRIEVAL_STRATEGY not in {"vector", "graph", "hybrid"}:
            raise ValueError(
                f"Invalid RETRIEVAL_STRATEGY: {cls.RETRIEVAL_STRATEGY}. "
                "Must be 'vector', 'graph', or 'hybrid'."
            )

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.GRAPH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.EVAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)


# Module-level convenience instance
config = Config()
