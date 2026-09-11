"""Tests for the database layer."""

import pytest
from pathlib import Path
import tempfile
from datetime import datetime

from repomind.api.database import Database
from repomind.application.models import JobStatus, IndexingProgress, RepositoryInfo


@pytest.fixture
def db():
    """Create a test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield Database(db_path)


def test_save_and_get_repository(db):
    """Test saving and retrieving repository metadata."""
    repo_info = RepositoryInfo(
        repo_id="test-repo-1",
        repo_url="https://github.com/user/repo",
        branch="main",
        status=JobStatus.PENDING,
        file_count=0,
        chunk_count=0,
    )

    db.save_repository(repo_info)
    retrieved = db.get_repository("test-repo-1")

    assert retrieved is not None
    assert retrieved.repo_id == "test-repo-1"
    assert retrieved.repo_url == "https://github.com/user/repo"
    assert retrieved.branch == "main"
    assert retrieved.status == JobStatus.PENDING


def test_update_repository(db):
    """Test updating repository metadata."""
    repo_info = RepositoryInfo(
        repo_id="test-repo-2",
        repo_url="https://github.com/user/repo",
        branch="main",
        status=JobStatus.PENDING,
    )
    db.save_repository(repo_info)

    # Update status
    repo_info.status = JobStatus.READY
    repo_info.file_count = 100
    repo_info.chunk_count = 500
    db.save_repository(repo_info)

    retrieved = db.get_repository("test-repo-2")
    assert retrieved.status == JobStatus.READY
    assert retrieved.file_count == 100
    assert retrieved.chunk_count == 500


def test_get_nonexistent_repository(db):
    """Test retrieving non-existent repository returns None."""
    retrieved = db.get_repository("nonexistent")
    assert retrieved is None


def test_save_and_get_job_progress(db):
    """Test saving and retrieving job progress."""
    progress = IndexingProgress(
        repo_id="test-repo-3",
        job_id="job-1",
        status=JobStatus.PARSING,
        progress=50,
        message="Parsing files...",
    )

    db.save_job_progress(progress)
    retrieved = db.get_job_progress("job-1")

    assert retrieved is not None
    assert retrieved.job_id == "job-1"
    assert retrieved.repo_id == "test-repo-3"
    assert retrieved.status == JobStatus.PARSING
    assert retrieved.progress == 50
    assert retrieved.message == "Parsing files..."


def test_update_job_progress(db):
    """Test updating job progress."""
    progress = IndexingProgress(
        repo_id="test-repo-4",
        job_id="job-2",
        status=JobStatus.EMBEDDING,
        progress=75,
        message="Generating embeddings...",
    )
    db.save_job_progress(progress)

    # Update progress
    progress.status = JobStatus.READY
    progress.progress = 100
    progress.message = "Indexing complete"
    db.save_job_progress(progress)

    retrieved = db.get_job_progress("job-2")
    assert retrieved.status == JobStatus.READY
    assert retrieved.progress == 100
    assert retrieved.message == "Indexing complete"


def test_list_repositories(db):
    """Test listing repositories."""
    for i in range(5):
        repo_info = RepositoryInfo(
            repo_id=f"test-repo-{i}",
            repo_url=f"https://github.com/user/repo{i}",
            branch="main",
            status=JobStatus.READY,
        )
        db.save_repository(repo_info)

    repos = db.list_repositories(limit=10)
    assert len(repos) == 5
    # Should be ordered by created_at DESC
    assert repos[0].repo_id == "test-repo-4"


def test_list_repositories_with_limit(db):
    """Test listing repositories with limit."""
    for i in range(10):
        repo_info = RepositoryInfo(
            repo_id=f"test-repo-limit-{i}",
            repo_url=f"https://github.com/user/repo{i}",
            branch="main",
            status=JobStatus.READY,
        )
        db.save_repository(repo_info)

    repos = db.list_repositories(limit=3)
    assert len(repos) == 3
