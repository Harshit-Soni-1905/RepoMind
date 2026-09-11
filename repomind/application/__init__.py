"""Application service layer shared between CLI and API.

This module extracts reusable business logic from the CLI layer (Stage 8)
so both the CLI and the new web API (Stage 10) can consume the same
repository indexing and query orchestration logic.
"""

from repomind.application.repo_service import RepositoryService
from repomind.application.query_service import QueryService

__all__ = ["RepositoryService", "QueryService"]
