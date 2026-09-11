# CLAUDE.md

Instructions for Claude Code when working on RepoMind.

---

## Project Context

RepoMind is a portfolio project demonstrating AI-powered codebase understanding for Python repositories. It combines AST-based parsing, code-aware chunking, hybrid retrieval (vector + graph), and a custom ReAct agent—without using LangChain or similar frameworks.

**Key constraint**: ₹0 API budget. Uses Google Gemini free tier for LLM, local embeddings, local vector store.

**Educational goal**: Build a production-quality system that a third-year CS student can understand deeply for technical interviews.

---

## Architecture Principles

1. **Modular**: Each subsystem is isolated with clear responsibilities
2. **Explicit over magical**: Implement core logic ourselves rather than hiding it in frameworks
3. **Code-aware, not text-aware**: Chunk on AST boundaries, not arbitrary token windows
4. **Evaluated**: Measure retrieval quality, compare vector-only vs. hybrid
5. **Configurable**: LLM provider and retrieval strategy are pluggable

See `docs/architecture.md` for full design rationale.

---

## Implementation Stages

RepoMind is being built incrementally. **Do not implement stages out of order.**

| Stage | Status | Deliverable |
|-------|--------|-------------|
| 0 | ✅ Complete | Project skeleton |
| 1 | ✅ Complete | Ingestion (repo scanning, file reading) |
| 2 | ✅ Complete | AST parsing (extract functions, classes, imports) |
| 3 | ✅ Complete | Code-aware chunking |
| 4 | ✅ Complete | Vector store (embeddings + ChromaDB) |
| 5 | ✅ Complete | Graph store (NetworkX, imports) |
| 6 | ✅ Complete | Hybrid retrieval |
| 7 | ✅ Complete | ReAct agent (Gemini SDK) |
| 8 | ✅ Complete | CLI interface |
| 9 | ✅ Complete | Evaluation harness |

**Before implementing a new stage**:
1. Confirm the previous stage is complete and tested
2. Present the proposed design for the new stage
3. Wait for explicit approval before writing implementation code

---

## Code Style Guidelines

### General Python Style
- Follow PEP 8
- Type hints on all function signatures
- Docstrings for all public functions/classes (Google style)
- Prefer explicit over clever
- Functions should do one thing
- Modules should not exceed ~300 lines

### Comments
- Explain *why*, not *what*
- Add comments for non-obvious logic only
- Do not comment trivial code

### Example:
```python
def chunk_function(func_node: ast.FunctionDef, source: str) -> CodeChunk:
    """Extract a single function as a code chunk.
    
    Args:
        func_node: AST node representing the function
        source: Original source code for extracting text
        
    Returns:
        CodeChunk with function text, metadata, and line range
    """
    # Include leading decorator lines in the chunk
    # (they're part of the function's semantic meaning)
    start_line = func_node.decorator_list[0].lineno if func_node.decorator_list else func_node.lineno
    
    text = extract_text(source, start_line, func_node.end_lineno)
    return CodeChunk(text=text, metadata={...})
```

---

## Testing Requirements

Every subsystem must have tests before the stage is considered complete.

- Unit tests for core logic
- Integration tests where subsystems interact
- Tests live in `tests/` mirroring `repomind/` structure
- Aim for >80% coverage on core modules

Example test structure:
```python
def test_scanner_finds_python_files():
    """scanner.scan() should find all .py files and skip venv/."""
    # Arrange: create temp directory with .py files and venv/
    # Act: scan the directory
    # Assert: correct files found, venv skipped
```

---

## Dependency Management

### Adding Dependencies

Dependencies should be added **only when actually needed** during implementation, not speculatively.

When adding a dependency:
1. Explain why it's needed and what it does
2. Add to `pyproject.toml` under `dependencies` (runtime) or `dev` (testing/dev only)
3. Document the choice in the relevant subsystem's docstring or comment

### Expected Dependencies (to be added during implementation)

| Stage | Dependency | Purpose |
|-------|-----------|---------|
| 1 (Ingestion) | `pathspec` | `.gitignore` pattern matching |
| 4 (Vector Store) | `sentence-transformers`, `chromadb` | Embeddings and vector search |
| 5 (Graph) | `networkx` | Graph data structure |
| 7 (Agent) | `google-genai` | Official Gemini API client |

Do not add these until their respective stages.

---

## Configuration

All configuration goes in `repomind/config.py`:
- Model names (embedding model, LLM)
- API keys (via environment variables, never hardcoded)
- Retrieval parameters (top-k, chunk size, etc.)
- File paths (vector store, graph store persistence)

Configuration should have sensible defaults and support environment variable overrides.

---

## LLM Provider Abstraction

The agent (Stage 7) must support pluggable LLM providers.

Design pattern:
```python
# repomind/agent/llm_provider.py
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion given a prompt."""
        pass

class GeminiProvider(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        # Gemini-specific API call
        ...

# In agent.py:
def __init__(self, provider: LLMProvider):
    self.provider = provider
```

This allows switching from Gemini to OpenAI/Anthropic/local models without touching core agent logic.

---

## Git Workflow

- Work on a feature branch for each stage (e.g., `stage-1-ingestion`)
- Commit frequently with clear messages
- Squash commits before merging to main
- Tag each completed stage (e.g., `v0.1.0-stage-1`)

---

## When Implementing a Stage

1. **Read `docs/architecture.md`** for the subsystem's design
2. **Present the plan**:
   - Files to create
   - Key functions/classes
   - Dependencies to add
   - Data structures
3. **Wait for approval**
4. **Implement**:
   - Write the code
   - Write tests
   - Update `pyproject.toml` if adding dependencies
   - Update `docs/architecture.md` if design evolved
5. **Run tests**: Ensure all tests pass
6. **Demo**: Show the subsystem working in isolation
7. **Mark stage complete**: Update this file

---

## Current Stage Status

**Current stage: Stage 10 (Web API + Interactive Demo) — COMPLETE**

All stages 0–10 are fully implemented:
- Stage 0-8: Core functionality (ingestion → CLI) 
- Stage 9: Evaluation harness with CALLS edges — 93% coverage
- Stage 10: FastAPI backend + React frontend + Docker — COMPLETE

**Stage 10 deliverables:**
- 11 backend modules (FastAPI, SQLite, JobManager, services)
- 10 frontend files (React + TypeScript + Vite)
- 28 new tests (database, repo service, API, integration)
- Docker deployment (Dockerfile + docker-compose.yml)
- Complete documentation (docs/stage10.md, installation guide)

**To verify Stage 10:**
```bash
# Install dependencies
pip install fastapi uvicorn pydantic sse-starlette

# Run Stage 10 tests
pytest tests/test_database.py tests/test_repo_service.py tests/test_api.py tests/test_integration.py -v

# Start application
docker-compose up --build
# or: python -m repomind.api.main (backend) + cd frontend && npm run dev (frontend)
```

See `STAGE10_COMPLETE.md` for full implementation report.

---

## Questions During Development

If a design decision is ambiguous:
1. Check `docs/architecture.md` first
2. If still unclear, ask for clarification before implementing
3. Document the decision in the code or architecture doc

Prefer asking over guessing when it comes to:
- Chunk size and overlap strategies
- Graph edge types
- Retrieval ranking algorithms
- Agent tool definitions

---

## Resources

- Python `ast` module: https://docs.python.org/3/library/ast.html
- ChromaDB docs: https://docs.trychroma.com/
- NetworkX docs: https://networkx.org/documentation/stable/
- Gemini API docs: https://ai.google.dev/docs
- ReAct paper: https://arxiv.org/abs/2210.03629

---

*Last updated: Stage 9 completion (2026-08-31)*
