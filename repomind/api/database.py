"""SQLite database for repository and job metadata persistence."""

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import sqlite3
from contextlib import contextmanager

from repomind.application.models import JobStatus, IndexingProgress, RepositoryInfo


class Database:
    """SQLite database for repository metadata."""

    def __init__(self, db_path: Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create database schema if not exists."""
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    repo_id TEXT PRIMARY KEY,
                    repo_url TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT 'main',
                    local_path TEXT,
                    status TEXT NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (repo_id) REFERENCES repositories (repo_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_repos_status
                ON repositories (status)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_repo_id
                ON jobs (repo_id)
            """)

            conn.commit()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_repository(self, repo_info: RepositoryInfo) -> None:
        """Save or update repository metadata.

        Args:
            repo_info: Repository information
        """
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO repositories (
                    repo_id, repo_url, branch, local_path,
                    status, file_count, chunk_count,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    status = excluded.status,
                    file_count = excluded.file_count,
                    chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
            """, (
                repo_info.repo_id,
                repo_info.repo_url or "",
                repo_info.branch,
                str(repo_info.local_path) if repo_info.local_path else None,
                repo_info.status.value,
                repo_info.file_count,
                repo_info.chunk_count,
                repo_info.created_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

    def get_repository(self, repo_id: str) -> Optional[RepositoryInfo]:
        """Retrieve repository metadata.

        Args:
            repo_id: Repository identifier

        Returns:
            RepositoryInfo or None if not found
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM repositories WHERE repo_id = ?
            """, (repo_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return RepositoryInfo(
                repo_id=row["repo_id"],
                repo_url=row["repo_url"],
                local_path=Path(row["local_path"]) if row["local_path"] else None,
                branch=row["branch"],
                status=JobStatus(row["status"]),
                file_count=row["file_count"],
                chunk_count=row["chunk_count"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def save_job_progress(self, progress: IndexingProgress) -> None:
        """Save job progress update.

        Args:
            progress: Job progress information
        """
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO jobs (
                    job_id, repo_id, status, progress,
                    message, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    progress = excluded.progress,
                    message = excluded.message,
                    error = excluded.error,
                    updated_at = excluded.updated_at
            """, (
                progress.job_id,
                progress.repo_id,
                progress.status.value,
                progress.progress,
                progress.message,
                progress.error,
                progress.created_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

    def get_job_progress(self, job_id: str) -> Optional[IndexingProgress]:
        """Retrieve job progress.

        Args:
            job_id: Job identifier

        Returns:
            IndexingProgress or None if not found
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM jobs WHERE job_id = ?
            """, (job_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return IndexingProgress(
                repo_id=row["repo_id"],
                job_id=row["job_id"],
                status=JobStatus(row["status"]),
                progress=row["progress"],
                message=row["message"],
                error=row["error"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_repositories(self, limit: int = 50) -> list[RepositoryInfo]:
        """List all repositories.

        Args:
            limit: Maximum number of repositories to return

        Returns:
            List of RepositoryInfo objects
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM repositories
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            return [
                RepositoryInfo(
                    repo_id=row["repo_id"],
                    repo_url=row["repo_url"],
                    local_path=Path(row["local_path"]) if row["local_path"] else None,
                    branch=row["branch"],
                    status=JobStatus(row["status"]),
                    file_count=row["file_count"],
                    chunk_count=row["chunk_count"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]
