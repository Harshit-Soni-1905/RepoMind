"""API request models."""

from typing import Optional
from pydantic import BaseModel, HttpUrl, Field


class IndexRepositoryRequest(BaseModel):
    """Request to index a repository."""
    repo_url: HttpUrl = Field(
        ...,
        description="GitHub or GitLab repository URL to index",
        examples=["https://github.com/user/repo"],
    )
    branch: Optional[str] = Field(
        "main",
        description="Branch to index",
        examples=["main", "develop"],
    )


class QueryRepositoryRequest(BaseModel):
    """Request to query an indexed repository."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask about the repository",
        examples=["How does the authentication flow work?"],
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider to use",
        examples=["gemini"],
    )
    model: Optional[str] = Field(
        None,
        description="LLM model name",
        examples=["gemini-1.5-flash"],
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Number of semantic search results",
    )
    depth: Optional[int] = Field(
        None,
        ge=0,
        le=5,
        description="Graph traversal depth",
    )
    max_iterations: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Maximum agent iterations",
    )