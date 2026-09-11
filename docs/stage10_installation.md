# Stage 10 Installation Guide

## Dependencies

Stage 10 adds the following runtime dependencies to `pyproject.toml`:

```toml
dependencies = [
    "pathspec>=0.11.0",
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    "networkx>=3.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "sse-starlette>=1.6.0",
]
```

## Installation Steps

### Option 1: Fresh Install

```bash
# Clone repository
git clone https://github.com/yourusername/repomind.git
cd repomind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e ".[dev]"
```

This installs:
- Core Stage 0-9 dependencies (pathspec, sentence-transformers, chromadb, networkx)
- Stage 10 web API dependencies (fastapi, uvicorn, pydantic, sse-starlette)
- Development dependencies (pytest, pytest-cov)

### Option 2: Upgrade Existing Installation

If you have Stages 0-9 already installed:

```bash
# Activate your existing virtual environment
source venv/bin/activate

# Install Stage 10 dependencies
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.5.0 sse-starlette>=1.6.0

# Or reinstall with updated pyproject.toml
pip install -e ".[dev]"
```

### Option 3: Docker (No Local Install Required)

```bash
# Set API key
export GEMINI_API_KEY="your-key"

# Build and run
docker-compose up --build
```

Docker installs all dependencies automatically inside containers.

## Verification

### Verify API Dependencies

```python
# Run this in Python to verify all packages are installed
import fastapi
import uvicorn
import pydantic
import sse_starlette

print("✓ All Stage 10 dependencies installed")
```

### Run Tests

```bash
# Run Stage 10 tests
pytest tests/test_database.py tests/test_repo_service.py tests/test_api.py tests/test_integration.py -v

# Run full test suite
pytest --cov=repomind
```

### Start API Server

```bash
# Start the API server
python -m repomind.api.main

# Or with uvicorn directly
uvicorn repomind.api.main:app --reload --host 0.0.0.0 --port 8000
```

Should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Frontend

```bash
cd frontend

# Install Node dependencies
npm install

# Start dev server
npm run dev
```

Should see:
```
VITE v5.0.8  ready in 450 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
➜  press h + enter to show help
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'fastapi'`:

```bash
# Ensure virtual environment is activated
which python  # Should show venv path

# Reinstall dependencies
pip install -e ".[dev]"
```

### Port Conflicts

If port 8000 or 5173 is in use:

```bash
# API: Override port
uvicorn repomind.api.main:app --port 8001

# Frontend: Edit frontend/vite.config.ts
# Change port: 5173 → 5174
```

### Docker Build Issues

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

## Development Workflow

1. **Backend changes**: API auto-reloads with `uvicorn --reload`
2. **Frontend changes**: Vite auto-reloads on file save
3. **Python changes**: Run `pip install -e .` to pick up new modules

## Next Steps

After installation:
1. Set `GEMINI_API_KEY` environment variable
2. Start backend: `python -m repomind.api.main`
3. Start frontend: `cd frontend && npm run dev`
4. Open browser: `http://localhost:5173`
5. Index a repository and start asking questions!
