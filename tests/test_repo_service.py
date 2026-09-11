"""Tests for the repository service."""

import pytest
from pathlib import Path
import tempfile

from repomind.application.repo_service import RepositoryService
from repomind.application.models import JobStatus


@pytest.fixture
def repo_service():
    """Create a repository service with temporary storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield RepositoryService(
            storage_root=Path(tmpdir) / "repos",
            vector_store_path=Path(tmpdir) / "vectorstore",
        )


def test_generate_repo_id(repo_service):
    """Test repository ID generation."""
    repo_id = repo_service.generate_repo_id()
    assert isinstance(repo_id, str)
    assert len(repo_id) > 0
    # Should be UUID format
    assert "-" in repo_id


def test_validate_github_url(repo_service):
    """Test GitHub URL validation."""
    is_valid, error = repo_service.validate_repository_url(
        "https://github.com/user/repo"
    )
    assert is_valid is True
    assert error is None


def test_validate_gitlab_url(repo_service):
    """Test GitLab URL validation."""
    is_valid, error = repo_service.validate_repository_url(
        "https://gitlab.com/user/repo"
    )
    assert is_valid is True
    assert error is None


def test_validate_invalid_url(repo_service):
    """Test validation rejects non-GitHub/GitLab URLs."""
    is_valid, error = repo_service.validate_repository_url(
        "https://bitbucket.org/user/repo"
    )
    assert is_valid is False
    assert "GitHub and GitLab" in error


def test_validate_non_https_url(repo_service):
    """Test validation rejects non-HTTPS URLs."""
    is_valid, error = repo_service.validate_repository_url(
        "git@github.com:user/repo.git"
    )
    assert is_valid is False
    assert "HTTP/HTTPS" in error


def test_get_collection_name(repo_service):
    """Test collection name generation."""
    repo_id = "test-repo-123"
    collection_name = repo_service.get_collection_name(repo_id)
    assert collection_name == "repomind_test-repo-123"
    # Should be deterministic
    assert repo_service.get_collection_name(repo_id) == collection_name


def test_validate_repository_size_small(repo_service, tmp_path):
    """Test size validation passes for small repository."""
    # Create a few small files
    for i in range(10):
        file_path = tmp_path / f"file{i}.py"
        file_path.write_text("print('hello')")

    is_valid, error = repo_service.validate_repository_size(tmp_path)
    assert is_valid is True
    assert error is None


def test_validate_repository_size_too_many_files(repo_service, tmp_path):
    """Test size validation rejects repositories with too many files."""
    # Create more than MAX_FILES_LIMIT files
    for i in range(1100):
        file_path = tmp_path / f"file{i}.py"
        file_path.write_text("x = 1")

    is_valid, error = repo_service.validate_repository_size(tmp_path)
    assert is_valid is False
    assert "file limit" in error.lower()
