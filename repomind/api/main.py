"""FastAPI application entry point."""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from repomind.api.routes import health, repos
from repomind.api.middleware import RateLimitMiddleware, request_id_middleware
from repomind.application.job_manager import JobManager
from repomind.application.repo_service import RepositoryService
from repomind.api.database import Database
from repomind.config import config
from repomind.utils import log_memory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting RepoMind API server")
    log_memory("API_STARTUP")

    # Initialize database
    db_path = Path(config.DATA_DIR) / "repomind.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    app.state.db = Database(db_path)

    # Initialize services
    app.state.repo_service = RepositoryService()
    app.state.job_manager = JobManager(repo_service=app.state.repo_service)

    logger.info(f"Database initialized at {db_path}")
    logger.info(f"Storage root: {app.state.repo_service.storage_root}")

    yield

    # Shutdown
    logger.info("Shutting down RepoMind API server")
    app.state.job_manager.executor.shutdown(wait=True)


def create_app(requests_per_minute: int = 60) -> FastAPI:
    """Create a FastAPI app instance with configurable rate limiting.

    Args:
        requests_per_minute: Maximum requests per IP per minute

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="RepoMind API",
        description="AI-powered codebase understanding for Python repositories",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://repomind-frontend-o43b.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware, requests_per_minute=requests_per_minute)

    # Request ID middleware
    app.middleware("http")(request_id_middleware)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if app.debug else "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(repos.router, prefix="/api/repos", tags=["Repositories"])

    return app


# Create default app instance
app = create_app(requests_per_minute=60)


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "repomind.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    start_server(reload=os.getenv("REPOMIND_RELOAD", "false").lower() == "true")
