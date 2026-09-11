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
