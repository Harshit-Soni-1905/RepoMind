"""API response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    components: Dict[str, str] = Field(default_factory=dict)


class IndexRepositoryResponse(BaseModel):
    """Response for repository indexing request."""
    repo_id: str = Field(..., description="Unique repository identifier")
    job_id: str = Field(..., description="Job identifier for status tracking")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    repo_id: str
    job_id: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    file_count: Optional[int] = None
    chunk_count: Optional[int] = None


class ToolExecutionResponse(BaseModel):
    """Tool execution in agent trace."""
    tool: str
    arguments: Dict[str, Any]
    summary: str
    order: int


class QueryRepositoryResponse(BaseModel):
    """Response for repository query."""
    answer: str
    tool_executions: List[ToolExecutionResponse] = Field(default_factory=list)
    success: bool
    error: Optional[str] = None