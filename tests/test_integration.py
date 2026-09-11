"""Stage 10 API integration tests.

These tests verify the complete API workflow including:
- Repository indexing
- Job status polling
- Query execution with SSE streaming
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import time

from repomind.api.main import create_app
from repomind.api.database import Database
from repomind.application.repo_service import RepositoryService
from repomind.application.job_manager import JobManager
from repomind.application.models import IndexingProgress, JobStatus, RepositoryInfo


@pytest.fixture
def integration_client():
    """Create a fully initialized test client with high rate limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create fresh app with high rate limit
        test_app = create_app(requests_per_minute=10000)
        test_app.state.db = Database(db_path)
        test_app.state.repo_service = RepositoryService(
            storage_root=Path(tmpdir) / "repos",
            vector_store_path=Path(tmpdir) / "vectorstore",
        )
        test_app.state.job_manager = JobManager(repo_service=test_app.state.repo_service)

        with TestClient(test_app) as test_client:
            yield test_client


def test_full_workflow_structure(integration_client):
    """Test the complete workflow structure (won't actually index)."""
    # Step 1: Submit indexing request
    index_response = integration_client.post(
        "/api/repos/index",
        json={
            "repo_url": "https://github.com/test/repo",
            "branch": "main",
        },
    )

    # Should accept the request
    assert index_response.status_code in [200, 400]

    if index_response.status_code == 200:
        data = index_response.json()
        assert "repo_id" in data
        assert "job_id" in data

        repo_id = data["repo_id"]

        # Step 2: Check status
        status_response = integration_client.get(f"/api/repos/{repo_id}/status")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert "status" in status_data
        assert "progress" in status_data


def test_cors_preflight(integration_client):
    """Test CORS preflight request."""
    response = integration_client.options(
        "/api/repos/index",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Should handle CORS
    assert response.status_code in [200, 204, 405]


def test_error_handling(integration_client):
    """Test error handling for invalid requests."""
    # Invalid URL
    response = integration_client.post(
        "/api/repos/index",
        json={"repo_url": "not-a-url"},
    )
    assert response.status_code == 422  # Pydantic validation error

    # Missing required field
    response = integration_client.post(
        "/api/repos/index",
        json={"branch": "main"},
    )
    assert response.status_code == 422


def test_health_check_structure(integration_client):
    """Test health check returns expected structure."""
    response = integration_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert "components" in data

    assert data["status"] == "healthy"
    assert isinstance(data["components"], dict)


def test_mocked_background_indexing(integration_client, monkeypatch):
    """Test full background indexing flow using a mocked clone and index_repository."""
    import time

    # 1. Mock clone_repository to immediately succeed by creating dummy python files
    def fake_clone(repo_url, repo_id, branch="main", progress_callback=None):
        repo_dir = integration_client.app.state.repo_service.storage_root / repo_id
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "main.py").write_text("def hello(): pass\n")
        return True, repo_dir, None

    # 2. Mock index_repository to also succeed quickly
    def fake_index(repo_id, repo_path, progress_callback=None):
        if progress_callback:
            progress_callback(IndexingProgress(
                repo_id=repo_id,
                job_id=repo_id,
                status=JobStatus.READY,
                progress=100,
                message="Indexing complete",
            ))
        return True, None

    monkeypatch.setattr(
        integration_client.app.state.repo_service,
        "clone_repository",
        fake_clone
    )
    monkeypatch.setattr(
        integration_client.app.state.repo_service,
        "index_repository",
        fake_index
    )

    # 3. Trigger index
    res = integration_client.post(
        "/api/repos/index",
        json={"repo_url": "https://github.com/test/repo", "branch": "main"}
    )
    assert res.status_code == 200
    data = res.json()
    repo_id = data["repo_id"]

    # 4. Wait for background thread to mark it READY
    for _ in range(20):
        time.sleep(0.2)
        status_res = integration_client.get(f"/api/repos/{repo_id}/status")
        if status_res.status_code == 200:
            status_data = status_res.json()
            if status_data["status"] == "ready":
                break

    # 5. Verify completion
    status_res = integration_client.get(f"/api/repos/{repo_id}/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "ready"
    assert status_res.json()["progress"] == 100


def test_multi_repository_isolation(integration_client):
    """Test that multiple repositories maintain isolated storage and metadata."""
    db = integration_client.app.state.db
    repo_service = integration_client.app.state.repo_service

    # Create Repo A
    repo_a = RepositoryInfo(
        repo_id="repo-alpha",
        repo_url="https://github.com/org/alpha",
        branch="main",
        status=JobStatus.READY,
        file_count=5,
        chunk_count=20,
    )
    db.save_repository(repo_a)
    db.save_job_progress(IndexingProgress(
        repo_id="repo-alpha",
        job_id="repo-alpha",
        status=JobStatus.READY,
        progress=100,
        message="Ready",
    ))

    # Create Repo B
    repo_b = RepositoryInfo(
        repo_id="repo-beta",
        repo_url="https://github.com/org/beta",
        branch="develop",
        status=JobStatus.READY,
        file_count=12,
        chunk_count=45,
    )
    db.save_repository(repo_b)
    db.save_job_progress(IndexingProgress(
        repo_id="repo-beta",
        job_id="repo-beta",
        status=JobStatus.READY,
        progress=100,
        message="Ready",
    ))

    # Verify collection names are isolated
    col_a = repo_service.get_collection_name("repo-alpha")
    col_b = repo_service.get_collection_name("repo-beta")
    assert col_a == "repomind_repo-alpha"
    assert col_b == "repomind_repo-beta"
    assert col_a != col_b

    # Verify database isolation
    fetched_a = db.get_repository("repo-alpha")
    fetched_b = db.get_repository("repo-beta")
    assert fetched_a.repo_url == "https://github.com/org/alpha"
    assert fetched_b.repo_url == "https://github.com/org/beta"
    assert fetched_a.branch == "main"
    assert fetched_b.branch == "develop"
    assert fetched_a.file_count == 5
    assert fetched_b.file_count == 12

    # Verify API status isolation
    status_a = integration_client.get("/api/repos/repo-alpha/status")
    status_b = integration_client.get("/api/repos/repo-beta/status")
    assert status_a.status_code == 200
    assert status_b.status_code == 200
    assert status_a.json()["repo_id"] == "repo-alpha"
    assert status_b.json()["repo_id"] == "repo-beta"
    assert status_a.json()["file_count"] == 5
    assert status_b.json()["file_count"] == 12


def test_sse_streaming_query(integration_client, monkeypatch):
    """Test SSE streaming endpoint with a mocked QueryService."""
    from repomind.application.models import QueryResult, ToolExecution

    db = integration_client.app.state.db
    repo_service = integration_client.app.state.repo_service

    # Setup ready repository on filesystem and DB
    repo_id = "test-sse-repo"
    repo_dir = repo_service.storage_root / repo_id
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("def test(): pass\n")

    repo_info = RepositoryInfo(
        repo_id=repo_id,
        repo_url="https://github.com/test/sse-repo",
        branch="main",
        status=JobStatus.READY,
        file_count=1,
        chunk_count=1,
    )
    db.save_repository(repo_info)

    # Mock QueryService.execute_query
    def fake_execute_query(self, repo_path, repo_id, query, **kwargs):
        stream_callback = kwargs.get("stream_callback")
        if stream_callback:
            stream_callback({
                "type": "tool_start",
                "tool": "semantic_search",
                "arguments": {"query": "test"},
            })
            stream_callback({
                "type": "tool_result",
                "tool": "semantic_search",
                "summary": "found test function",
                "order": 0,
            })
            stream_callback({
                "type": "answer",
                "text": "This repository contains a test function.",
            })
        return QueryResult(
            answer="This repository contains a test function.",
            tool_executions=[
                ToolExecution(
                    tool_name="semantic_search",
                    arguments={"query": "test"},
                    result_summary="found test function",
                    execution_order=0,
                )
            ],
            success=True,
            error=None,
        )

    from repomind.application.query_service import QueryService
    monkeypatch.setattr(QueryService, "execute_query", fake_execute_query)

    # Post query
    response = integration_client.post(
        f"/api/repos/{repo_id}/ask",
        json={"question": "What does test do?", "provider": "local"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Verify event stream contents
    content = response.text
    assert "event: start" in content
    assert "event: tool_start" in content
    assert "event: tool_result" in content
    assert "event: answer" in content
    assert "event: done" in content

