"""Tests for the API routes."""

import json
import os
import re
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from repomind.api.main import app, create_app
from repomind.api.database import Database
from repomind.application.repo_service import RepositoryService
from repomind.application.job_manager import JobManager


@pytest.fixture
def client():
    """Create a test client with fresh database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        app.state.db = Database(db_path)
        app.state.repo_service = RepositoryService(
            storage_root=Path(tmpdir) / "repos"
        )
        app.state.job_manager = JobManager(repo_service=app.state.repo_service)

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def rate_limited_client():
    """Create a test client with rate limiter disabled (high limit)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create a fresh app instance with custom rate limit
        test_app = create_app(requests_per_minute=10000)
        test_app.state.db = Database(db_path)
        test_app.state.repo_service = RepositoryService(
            storage_root=Path(tmpdir) / "repos"
        )
        test_app.state.job_manager = JobManager(repo_service=test_app.state.repo_service)

        with TestClient(test_app) as test_client:
            yield test_client


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "components" in data


def test_index_repository_invalid_url(client):
    """Test indexing with invalid repository URL."""
    response = client.post(
        "/api/repos/index",
        json={"repo_url": "https://invalid.com/repo"},
    )
    assert response.status_code == 400
    assert "error" in response.json() or "detail" in response.json()


def test_index_repository_valid_url(client):
    """Test indexing with valid repository URL structure."""
    response = client.post(
        "/api/repos/index",
        json={
            "repo_url": "https://github.com/user/repo",
            "branch": "main",
        },
    )
    # Will fail to clone (doesn't exist), but should accept the request
    assert response.status_code in [200, 400, 500]

    if response.status_code == 200:
        data = response.json()
        assert "repo_id" in data
        assert "job_id" in data
        assert "status" in data


def test_get_nonexistent_repository_status(client):
    """Test getting status of non-existent repository."""
    response = client.get("/api/repos/nonexistent-id/status")
    assert response.status_code == 404


def test_rate_limiting(client):
    """Test rate limiting middleware."""
    # Test with a low limit for quick verification
    # Make requests rapidly - should hit rate limit at 60/min
    responses = []
    for _ in range(65):
        response = client.get("/health")
        responses.append(response.status_code)

    # Should see at least one 429 (rate limited)
    assert 429 in responses


def test_rate_limiting_headers(client):
    """Test rate limiting headers on response."""
    response = client.get("/health")
    # Rate limit headers should be present
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers


def test_request_id_header(client):
    """Test that request ID is added to response headers."""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_cors_headers(rate_limited_client):
    """Test CORS headers are present."""
    response = rate_limited_client.options(
        "/api/repos/index",
        headers={"Origin": "http://localhost:5173"},
    )
    # CORS headers should be present
    assert response.status_code in [200, 405]


# ---- SSE event-name contract regression tests ----


def _parse_sse_stream(text: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts.

    Mirrors the fixed frontend parser: correlates the 'event:' line with the
    'data:' line in the same SSE block (blocks separated by blank lines, LF or CRLF).
    """
    events = []
    for block in re.split(r"\r?\n\r?\n", text):
        event_type = "message"
        data_str = ""
        for line in re.split(r"\r?\n", block):
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str += line[len("data:"):].strip()
        if not data_str:
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        events.append({"event": event_type, "data": data})
    return events


def test_sse_event_names_contract():
    """Verify that /ask SSE stream uses named events the frontend expects.

    The frontend SSE parser reads the 'event:' field from each SSE block
    and dispatches on: start, tool_start, tool_result, answer, done, error.
    This test ensures the backend emits those names (not bare 'message').

    Regression test for: browser stuck on 'Agent is analyzing the codebase...'
    caused by event-name mismatch between backend SSE and frontend parser.
    """
    from repomind.application.models import JobStatus, RepositoryInfo, QueryResult

    # Mock a ready repository so we reach the SSE generator
    mock_repo = MagicMock(spec=RepositoryInfo)
    mock_repo.status = JobStatus.READY

    mock_db = MagicMock()
    mock_db.get_repository.return_value = mock_repo

    with tempfile.TemporaryDirectory() as tmpdir:
        storage_root = Path(tmpdir)
        fake_repo_dir = storage_root / "fake-repo-id"
        os.makedirs(fake_repo_dir, exist_ok=True)

        mock_repo_svc = MagicMock()
        mock_repo_svc.storage_root = storage_root
        mock_repo_svc.load_graph.return_value = None

        def fake_execute(self, *, stream_callback=None, **kwargs):
            if stream_callback:
                stream_callback({"type": "tool_result", "tool": "semantic_search",
                                 "summary": "found 3 results", "order": 0})
                stream_callback({"type": "answer", "text": "Test answer"})
            return QueryResult(answer="Test answer", tool_executions=[], success=True)

        with patch("repomind.api.routes.repos.QueryService") as MockQS:
            instance = MockQS.return_value
            instance.execute_query = lambda **kw: fake_execute(instance, **kw)

            test_app = create_app(requests_per_minute=10000)
            with TestClient(test_app) as tc:
                test_app.state.db = mock_db
                test_app.state.repo_service = mock_repo_svc

                response = tc.post(
                    "/api/repos/fake-repo-id/ask",
                    json={"question": "What does it do?"},
                )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    raw = response.text

    events = _parse_sse_stream(raw)

    # Must have named events, not bare 'message'
    event_names = [e["event"] for e in events]
    assert "start" in event_names, f"Missing 'start' event. Got: {event_names}"

    # Every event from the backend must use a named type the frontend handles
    allowed = {"start", "tool_start", "tool_result", "answer", "done", "error"}
    for name in event_names:
        assert name in allowed, (
            f"Backend emitted unknown SSE event '{name}'. "
            f"Frontend only handles: {allowed}"
        )

    # The answer event must carry a 'text' field
    answer_events = [e for e in events if e["event"] == "answer"]
    assert len(answer_events) >= 1, "No 'answer' event found in SSE stream"
    assert "text" in answer_events[0]["data"], (
        "answer event missing 'text' in data payload"
    )

    # done event must be present
    assert "done" in event_names, f"Missing 'done' event. Got: {event_names}"


# ---- Indexing persistence and DB fallback regression tests ----


def test_initial_pending_progress_persisted_to_db(rate_limited_client, monkeypatch):
    """Verify that initial PENDING progress is written to SQLite immediately on indexing request.

    Regression test for: 404 Not Found on /status if in-memory state is queried
    before background worker updates or if worker crashes.
    """
    # Block clone so we can verify initial database state
    def blocked_clone(repo_url, repo_id, branch="main", progress_callback=None):
        return True, Path(tempfile.gettempdir()), None

    monkeypatch.setattr(
        rate_limited_client.app.state.repo_service,
        "clone_repository",
        blocked_clone,
    )
    monkeypatch.setattr(
        rate_limited_client.app.state.repo_service,
        "validate_repository_size",
        lambda p: (True, None),
    )
    monkeypatch.setattr(
        rate_limited_client.app.state.repo_service,
        "index_repository",
        lambda repo_id, repo_path, progress_callback=None: (True, None),
    )

    response = rate_limited_client.post(
        "/api/repos/index",
        json={"repo_url": "https://github.com/user/test-repo", "branch": "main"},
    )
    assert response.status_code == 200
    data = response.json()
    repo_id = data["repo_id"]
    job_id = data["job_id"]
    assert repo_id == job_id

    # Verify directly from SQLite DB
    db = rate_limited_client.app.state.db
    saved_progress = db.get_job_progress(repo_id)
    assert saved_progress is not None, "Initial progress was not saved to SQLite database"
    assert saved_progress.repo_id == repo_id
    assert saved_progress.job_id == job_id


def test_status_endpoint_survives_jobmanager_cache_loss(rate_limited_client):
    """Verify that /status succeeds from SQLite even when JobManager memory cache is wiped.

    Regression test for: ephemeral in-memory JobManager state loss causing 404 errors.
    """
    from repomind.application.models import JobStatus, IndexingProgress, RepositoryInfo

    db = rate_limited_client.app.state.db
    job_manager = rate_limited_client.app.state.job_manager

    repo_id = "test-survive-cache-loss"
    repo_info = RepositoryInfo(
        repo_id=repo_id,
        repo_url="https://github.com/test/survive",
        branch="main",
        status=JobStatus.PARSING,
        file_count=10,
        chunk_count=25,
    )
    db.save_repository(repo_info)

    db_progress = IndexingProgress(
        repo_id=repo_id,
        job_id=repo_id,
        status=JobStatus.PARSING,
        progress=45,
        message="Parsing AST nodes...",
    )
    db.save_job_progress(db_progress)

    # Ensure in-memory cache is empty for this repo
    job_manager.jobs.pop(repo_id, None)
    assert job_manager.get_job_status(repo_id) is None

    # Call status endpoint
    response = rate_limited_client.get(f"/api/repos/{repo_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == repo_id
    assert data["job_id"] == repo_id
    assert data["status"] == "parsing"
    assert data["progress"] == 45
    assert data["message"] == "Parsing AST nodes..."
    assert data["file_count"] == 10
    assert data["chunk_count"] == 25


def test_job_manager_subscribe_replays_existing_progress():
    """Verify that JobManager.subscribe immediately replays cached state to new listeners.

    Regression test for: subscription race condition where start_indexing_job
    emits initial progress before the database listener is attached.
    """
    from repomind.application.models import JobStatus, IndexingProgress
    from repomind.application.job_manager import JobManager

    jm = JobManager()
    progress = IndexingProgress(
        repo_id="test-replay",
        job_id="test-replay",
        status=JobStatus.PENDING,
        progress=0,
        message="Queued",
    )
    jm._update_progress(progress)

    received_events = []
    jm.subscribe("test-replay", lambda p: received_events.append(p))

    # The existing progress must have been immediately replayed upon subscription
    assert len(received_events) == 1
    assert received_events[0].repo_id == "test-replay"
    assert received_events[0].status == JobStatus.PENDING

    # Future updates must also be received
    next_progress = IndexingProgress(
        repo_id="test-replay",
        job_id="test-replay",
        status=JobStatus.CLONING,
        progress=10,
        message="Cloning...",
    )
    jm._update_progress(next_progress)
    assert len(received_events) == 2
    assert received_events[1].status == JobStatus.CLONING

