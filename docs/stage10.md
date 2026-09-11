# Stage 10: Web API + Interactive Demo

## Overview

Stage 10 transforms RepoMind from a CLI-only tool into a full-stack web application with:
- FastAPI REST backend with async job processing
- React + TypeScript frontend with Server-Sent Events
- Multi-repository isolation (concurrent indexing)
- Docker deployment with docker-compose
- Comprehensive API and integration tests

---

## Architecture

### Backend (FastAPI + SQLite)

```
┌─────────────────────────────────────────┐
│           FastAPI Application            │
├─────────────────────────────────────────┤
│  Routes:                                 │
│  - POST /api/repos/index                 │
│  - GET  /api/repos/{id}/status           │
│  - POST /api/repos/{id}/ask (SSE)        │
│  - GET  /health                          │
├─────────────────────────────────────────┤
│  Middleware:                             │
│  - CORS (allow localhost:5173)           │
│  - Rate limiting (60 req/min per IP)     │
│  - Request ID tracking                   │
│  - Global exception handler              │
└─────────────────────────────────────────┘
         │                    │
         ↓                    ↓
┌──────────────┐    ┌──────────────────┐
│  JobManager  │    │  RepositoryService│
│  (ThreadPool)│    │  (Clone + Index)  │
└──────────────┘    └──────────────────┘
         │                    │
         ↓                    ↓
┌──────────────────────────────────────┐
│          SQLite Database              │
│  - repositories (metadata)            │
│  - jobs (progress tracking)           │
└──────────────────────────────────────┘
```

### Frontend (React + TypeScript + Vite)

```
┌──────────────────────────────────────┐
│            React App                 │
├──────────────────────────────────────┤
│  Components:                         │
│  - IndexForm (repo URL submission)   │
│  - StatusPanel (polling, progress)   │
│  - QueryPanel (SSE streaming)        │
├──────────────────────────────────────┤
│  Services:                           │
│  - API client (fetch + SSE)          │
│  - TypeScript types                  │
└──────────────────────────────────────┘
         ↓ (HTTP + SSE)
┌──────────────────────────────────────┐
│       Vite Dev Proxy → API           │
│       /api/* → http://localhost:8000 │
└──────────────────────────────────────┘
```

### Multi-Repository Isolation

Each indexed repository gets:
1. **Unique repo_id** (UUID)
2. **Dedicated ChromaDB collection**: `repomind_{repo_id}`
3. **Separate graph file**: `.repomind_graph.pickle` in repo directory
4. **Local clone**: `~/.repomind_data/repos/{repo_id}/`

This allows concurrent indexing and querying of multiple repositories without conflicts.

---

## API Endpoints

### POST /api/repos/index

Start indexing a repository.

**Request:**
```json
{
  "repo_url": "https://github.com/user/repo",
  "branch": "main"
}
```

**Response:**
```json
{
  "repo_id": "a1b2c3d4-...",
  "job_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Repository indexing job started"
}
```

### GET /api/repos/{repo_id}/status

Poll job status.

**Response:**
```json
{
  "repo_id": "a1b2c3d4-...",
  "job_id": "a1b2c3d4-...",
  "status": "embedding",
  "progress": 65,
  "message": "Generating embeddings...",
  "error": null,
  "created_at": "2026-09-02T06:00:00Z",
  "updated_at": "2026-09-02T06:02:00Z",
  "file_count": 100,
  "chunk_count": 500
}
```

**Status values:** `pending`, `cloning`, `scanning`, `parsing`, `chunking`, `embedding`, `graph_building`, `ready`, `failed`

### POST /api/repos/{repo_id}/ask

Query repository with Server-Sent Events streaming.

**Request:**
```json
{
  "question": "How does authentication work?",
  "provider": "gemini",
  "model": "gemini-1.5-flash",
  "top_k": 10,
  "depth": 2,
  "max_iterations": 10
}
```

**SSE Stream Events:**
```
event: start
data: {"repo_id": "a1b2c3d4-..."}

event: tool_start
data: {"tool": "semantic_search", "arguments": {...}}

event: tool_result
data: {"tool": "semantic_search", "summary": "Found 5 relevant chunks", "order": 0}

event: answer
data: {"text": "Authentication is handled by..."}

event: done
data: {"answer": "...", "tool_count": 3}
```

### GET /health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-09-02T06:00:00Z",
  "components": {
    "api": "ok",
    "vectorstore": "ok",
    "graph": "ok"
  }
}
```

---

## Security & Rate Limiting

### Security Measures

1. **URL validation**: Only GitHub and GitLab HTTPS URLs accepted
2. **Repository size limits**: 
   - Max 1000 Python files
   - Max 50MB total size
3. **Rate limiting**: 60 requests/minute per IP
4. **CORS**: Restricted to `localhost:5173`, `localhost:3000`
5. **Clone timeout**: 5-minute limit per repository
6. **Request tracking**: Unique request IDs in headers

### Environment Variables

```bash
# LLM Configuration
GEMINI_API_KEY=your-api-key-here
REPOMIND_LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-flash

# API Configuration
REPOMIND_API_HOST=0.0.0.0
REPOMIND_API_PORT=8000
REPOMIND_DATA_DIR=/app/.repomind_data
REPOMIND_MAX_WORKERS=4
REPOMIND_RATE_LIMIT=60

# Embedding Configuration
REPOMIND_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Configuration
REPOMIND_VECTOR_TOP_K=10
REPOMIND_GRAPH_DEPTH=2
REPOMIND_AGENT_MAX_ITER=10
```

---

## Development

### Running Locally

**Backend:**
```bash
# Install dependencies
pip install -e ".[dev]"

# Start API server
python -m repomind.api.main

# Or with auto-reload
uvicorn repomind.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend

# Install dependencies
npm install

# Start dev server (with API proxy)
npm run dev
```

### Running with Docker

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Remove volumes (clears data)
docker-compose down -v
```

---

## Testing

Stage 10 adds comprehensive API and integration tests:

```bash
# Run all tests including new API tests
pytest

# Run only Stage 10 tests
pytest tests/test_api.py tests/test_database.py tests/test_repo_service.py tests/test_integration.py

# Run with coverage
pytest --cov=repomind --cov-report=term-missing

# Run integration tests
pytest tests/test_integration.py -v
```

**Test coverage:**
- Database persistence (save/retrieve repositories, jobs)
- Repository service (URL validation, size checks, collection naming)
- API routes (index, status, health)
- Middleware (rate limiting, request IDs, CORS)
- Error handling (validation, 404s, rate limits)
- Integration workflow (end-to-end request flow)

---

## Deployment Notes

### Docker Volumes

- `repomind_data`: Persists cloned repositories, vector store, graph files, SQLite database

### Port Mapping

- `8000`: API server (FastAPI + Uvicorn)
- `5173`: Frontend (Nginx serving React build)

### Health Checks

Docker Compose includes health checks for the API service:
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Design Decisions

### Why ThreadPoolExecutor over AsyncIO?

Indexing is CPU-bound (AST parsing, embedding generation). ThreadPoolExecutor provides:
- Simple concurrency model
- Natural integration with synchronous Stage 1-9 code
- Bounded parallelism (configurable workers)

### Why SQLite over Redis/Postgres?

SQLite is sufficient for this use case:
- Single-node deployment
- Moderate write volume (one write per indexing stage)
- Simple schema (repositories + jobs)
- Zero operational overhead

### Why SSE over WebSockets?

Server-Sent Events are simpler for unidirectional streaming:
- Standard EventSource API
- Automatic reconnection
- HTTP-based (works through proxies)
- No need for bidirectional communication

---

## Known Limitations

1. **Single-node only**: No horizontal scaling (SQLite, in-memory rate limiter)
2. **No authentication**: Public API, rate-limited by IP only
3. **No repository updates**: Must re-index to update (no incremental sync)
4. **Clone depth=1**: No git history analysis
5. **Public repos only**: No SSH or authenticated HTTPS clones

These are acceptable for a portfolio project demonstrating the architecture.

---

## Future Enhancements (Out of Scope)

- User authentication and multi-tenancy
- Repository update/refresh without re-cloning
- Horizontal scaling with Redis job queue
- GitHub/GitLab OAuth for private repositories
- Incremental indexing (only changed files)
- Repository comparison ("what changed between v1 and v2?")

---

*Last updated: Stage 10 completion (2026-09-02)*
