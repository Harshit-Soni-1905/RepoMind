"""Repository management endpoints."""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from repomind.api.models.requests import IndexRepositoryRequest, QueryRepositoryRequest
from repomind.api.models.responses import (
    IndexRepositoryResponse,
    JobStatusResponse,
    QueryRepositoryResponse,
    ToolExecutionResponse,
    ErrorResponse,
)
from repomind.application.models import JobStatus, RepositoryInfo, IndexingProgress
from repomind.application.query_service import QueryService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/index", response_model=IndexRepositoryResponse)
async def index_repository(
    request: Request,
    body: IndexRepositoryRequest,
):
    """Start indexing a repository.

    Args:
        request: FastAPI request
        body: Repository indexing request

    Returns:
        IndexRepositoryResponse with job ID
    """
    repo_service = request.app.state.repo_service
    job_manager = request.app.state.job_manager
    db = request.app.state.db

    # Validate URL
    repo_url = str(body.repo_url)
    is_valid, error = repo_service.validate_repository_url(repo_url)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Generate repo ID
    repo_id = repo_service.generate_repo_id()

    # Save initial repository record
    repo_info = RepositoryInfo(
        repo_id=repo_id,
        repo_url=repo_url,
        branch=body.branch or "main",
        status=JobStatus.PENDING,
    )
    db.save_repository(repo_info)

    # Persist initial PENDING progress to the database *before* the
    # background job starts, so the status endpoint can always find it
    # even if the process restarts before any listener fires.
    initial_progress = IndexingProgress(
        repo_id=repo_id,
        job_id=repo_id,
        status=JobStatus.PENDING,
        progress=0,
        message="Job queued for processing",
    )
    db.save_job_progress(initial_progress)

    # Subscribe to job updates to persist to database.
    # Must be registered BEFORE start_indexing_job so the listener
    # catches every update from the background thread.
    def persist_progress(progress):
        db.save_job_progress(progress)
        # Update repository status
        repo_info.status = progress.status
        db.save_repository(repo_info)

    # Start background job
    job_id = job_manager.start_indexing_job(
        repo_url=repo_url,
        repo_id=repo_id,
        branch=body.branch or "main",
    )

    job_manager.subscribe(job_id, persist_progress)

    return IndexRepositoryResponse(
        repo_id=repo_id,
        job_id=job_id,
        status=JobStatus.PENDING.value,
        message="Repository indexing job started",
    )


@router.get("/{repo_id}/status", response_model=JobStatusResponse)
async def get_repository_status(
    repo_id: str,
    request: Request,
):
    """Get indexing job status.

    Args:
        repo_id: Repository identifier
        request: FastAPI request

    Returns:
        JobStatusResponse with current status
    """
    job_manager = request.app.state.job_manager
    db = request.app.state.db

    # Try in-memory first
    progress = job_manager.get_job_status(repo_id)

    # Fall back to database
    if not progress:
        progress = db.get_job_progress(repo_id)

    if not progress:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Get repository info for counts
    repo_info = db.get_repository(repo_id)

    return JobStatusResponse(
        repo_id=progress.repo_id,
        job_id=progress.job_id,
        status=progress.status.value,
        progress=progress.progress,
        message=progress.message,
        error=progress.error,
        created_at=progress.created_at,
        updated_at=progress.updated_at,
        file_count=repo_info.file_count if repo_info else None,
        chunk_count=repo_info.chunk_count if repo_info else None,
    )


@router.post("/{repo_id}/ask")
async def query_repository(
    repo_id: str,
    body: QueryRepositoryRequest,
    request: Request,
):
    """Query an indexed repository with SSE streaming.

    Args:
        repo_id: Repository identifier
        body: Query request
        request: FastAPI request

    Returns:
        Server-Sent Events stream of agent tool traces
    """
    repo_service = request.app.state.repo_service
    db = request.app.state.db

    # Verify repository is indexed
    repo_info = db.get_repository(repo_id)
    if not repo_info:
        raise HTTPException(status_code=404, detail="Repository not found")

    if repo_info.status != JobStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Repository is not ready (status: {repo_info.status.value})",
        )

    # Get repository path
    repo_path = repo_service.storage_root / repo_id
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository files not found")

    # Load graph
    graph = repo_service.load_graph(repo_path)

    # Execute query with SSE streaming
    async def event_generator():
        """Generate SSE events during query execution."""
        try:
            # Stream start
            yield {"event": "start", "data": json.dumps({"repo_id": repo_id})}

            # Query execution with streaming callback
            events = []

            def stream_callback(event):
                events.append(event)

            query_service = QueryService()
            result = query_service.execute_query(
                repo_path=repo_path,
                repo_id=repo_id,
                query=body.question,
                graph=graph,
                provider_name=body.provider,
                model_name=body.model,
                top_k=body.top_k,
                depth=body.depth,
                max_iterations=body.max_iterations,
                stream_callback=stream_callback,
            )

            # Stream buffered events
            for event in events:
                if event["type"] == "tool_start":
                    yield {
                        "event": "tool_start",
                        "data": json.dumps({
                            "tool": event["tool"],
                            "arguments": event["arguments"],
                        })
                    }
                elif event["type"] == "tool_result":
                    yield {
                        "event": "tool_result",
                        "data": json.dumps({
                            "tool": event["tool"],
                            "summary": event["summary"],
                            "order": event["order"],
                        })
                    }
                elif event["type"] == "answer":
                    yield {
                        "event": "answer",
                        "data": json.dumps({"text": event["text"]})
                    }
                elif event["type"] == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": event["message"]})
                    }

            # Final result
            if result.success:
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "answer": result.answer,
                        "tool_count": len(result.tool_executions),
                    })
                }
            else:
                yield {
                    "event": "error",
                    "data": json.dumps({"message": result.error or "Query failed"})
                }

        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())
