"""Background job management for repository indexing."""

import asyncio
from typing import Dict, Optional, Callable, List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

from repomind.application.models import JobStatus, IndexingProgress
from repomind.application.repo_service import RepositoryService

logger = logging.getLogger(__name__)


class JobManager:
    """Manages asynchronous repository indexing background jobs."""

    def __init__(
        self,
        repo_service: Optional[RepositoryService] = None,
        max_workers: int = 4,
    ):
        """Initialize job manager.

        Args:
            repo_service: Repository service instance
            max_workers: Max concurrent indexing workers
        """
        self.repo_service = repo_service or RepositoryService()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, IndexingProgress] = {}
        self.listeners: Dict[str, List[Callable[[IndexingProgress], None]]] = {}

    def get_job_status(self, job_id: str) -> Optional[IndexingProgress]:
        """Get the current progress of a job.

        Args:
            job_id: Job identifier

        Returns:
            IndexingProgress or None if job not found
        """
        return self.jobs.get(job_id)

    def subscribe(self, job_id: str, callback: Callable[[IndexingProgress], None]) -> None:
        """Subscribe to job status updates.

        Args:
            job_id: Job identifier
            callback: Function to call on progress updates
        """
        if job_id not in self.listeners:
            self.listeners[job_id] = []
        self.listeners[job_id].append(callback)

    def unsubscribe(self, job_id: str, callback: Callable[[IndexingProgress], None]) -> None:
        """Unsubscribe from job status updates."""
        if job_id in self.listeners:
            self.listeners[job_id] = [c for c in self.listeners[job_id] if c != callback]

    def _update_progress(self, progress: IndexingProgress) -> None:
        """Update job progress state and notify listeners."""
        self.jobs[progress.job_id] = progress
        listeners = self.listeners.get(progress.job_id, [])
        for listener in listeners:
            try:
                listener(progress)
            except Exception as e:
                logger.error(f"Error in job progress listener: {e}")

    def start_indexing_job(self, repo_url: str, repo_id: str, branch: str = "main") -> str:
        """Start an indexing job in background thread pool.

        Args:
            repo_url: Repository URL
            repo_id: Unique repository ID
            branch: Repository branch

        Returns:
            job_id (same as repo_id)
        """
        job_id = repo_id

        # Initial pending progress
        initial_progress = IndexingProgress(
            repo_id=repo_id,
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            message="Job queued for processing",
        )
        self._update_progress(initial_progress)

        # Submit background task to thread pool
        self.executor.submit(self._run_indexing_job, repo_url, repo_id, branch)
        return job_id

    def _run_indexing_job(self, repo_url: str, repo_id: str, branch: str) -> None:
        """Execute the indexing workflow inside thread pool worker."""
        def progress_cb(progress: IndexingProgress):
            self._update_progress(progress)

        # Step 1: Clone
        clone_ok, local_path, clone_err = self.repo_service.clone_repository(
            repo_url=repo_url,
            repo_id=repo_id,
            branch=branch,
            progress_callback=progress_cb,
        )

        if not clone_ok or not local_path:
            self._update_progress(IndexingProgress(
                repo_id=repo_id,
                job_id=repo_id,
                status=JobStatus.FAILED,
                progress=0,
                message="Clone failed",
                error=clone_err or "Unknown clone failure",
            ))
            return

        # Step 2: Validate Size
        size_ok, size_err = self.repo_service.validate_repository_size(local_path)
        if not size_ok:
            self.repo_service.cleanup_repository(repo_id)
            self._update_progress(IndexingProgress(
                repo_id=repo_id,
                job_id=repo_id,
                status=JobStatus.FAILED,
                progress=0,
                message="Size validation failed",
                error=size_err or "Repository exceeds limits",
            ))
            return

        # Step 3: Index
        index_ok, index_err = self.repo_service.index_repository(
            repo_path=local_path,
            repo_id=repo_id,
            progress_callback=progress_cb,
        )

        if not index_ok:
            self.repo_service.cleanup_repository(repo_id)
            self._update_progress(IndexingProgress(
                repo_id=repo_id,
                job_id=repo_id,
                status=JobStatus.FAILED,
                progress=0,
                message="Indexing failed",
                error=index_err or "Unknown indexing error",
            ))
            return
