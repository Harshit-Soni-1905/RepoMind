"""API middleware for rate limiting and request tracking."""

import time
import uuid
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            app: FastAPI application
            requests_per_minute: Max requests per IP per minute
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        """Check rate limit before processing request."""
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old requests (older than 60 seconds)
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < 60
        ]

        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute",
                },
            )
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(int(now + 60))
            return response

        # Record this request
        self.requests[client_ip].append(now)

        # Process request
        response = await call_next(request)
        # Add rate limit headers to all responses
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.requests_per_minute - len(self.requests[client_ip])))
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))
        return response


async def request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
