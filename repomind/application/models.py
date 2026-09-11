"""Data models for the application service layer."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class JobStatus(str, Enum):
    """Repository indexing job status."""
    PENDING = "pending"
    CLONING = "cloning"
    SCANNING = "scanning"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    GRAPH_BUILDING = "graph_building"
    READY = "ready"
    FAILED = "failed"


@dataclass
class IndexingProgress:
    """Represents the current state of a repository indexing job."""
    repo_id: str
    job_id: str
    status: JobStatus
    progress: int  # 0-100
    message: str
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "repo_id": self.repo_id,
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class RepositoryInfo:
    """Repository metadata."""
    repo_id: str
    repo_url: Optional[str] = None
    local_path: Optional[Path] = None
    branch: str = "main"
    status: JobStatus = JobStatus.PENDING
    file_count: int = 0
    chunk_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ToolExecution:
    """Represents a single tool execution by the agent."""
    tool_name: str
    arguments: Dict[str, Any]
    result_summary: str
    execution_order: int


@dataclass
class QueryResult:
    """Result of a repository query."""
    answer: str
    tool_executions: List[ToolExecution]
    success: bool
    error: Optional[str] = None
