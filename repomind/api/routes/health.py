"""Health check endpoint."""

from fastapi import APIRouter
from datetime import datetime, timezone

from repomind.api.models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        components={
            "api": "ok",
            "vectorstore": "ok",
            "graph": "ok",
        },
    )
