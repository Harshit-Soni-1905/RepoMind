# RepoMind

**AI-powered codebase understanding agent for Python repositories**

RepoMind is a portfolio project demonstrating advanced code analysis techniques, hybrid retrieval (semantic + graph), and agentic reasoning over codebases—without relying on heavyweight frameworks like LangChain.

---

## What Makes RepoMind Different

Unlike generic "chat with your code" RAG systems that chunk code arbitrarily:

- **Code-aware chunking**: Chunks follow syntactic boundaries (functions, classes) using Python's AST
- **Hybrid retrieval**: Combines semantic search with graph traversal over imports and call relationships
- **Transparent agent**: Implements ReAct loop from scratch using raw LLM SDK calls—no black-box orchestration
- **Zero API cost**: Uses Google Gemini's free tier for reasoning, local embeddings, local vector store
- **Evaluated**: Measures retrieval quality and compares vector-only vs. hybrid approaches

---

## Architecture Overview

```
Ingestion → AST Parsing → Code-aware Chunking
                ↓
        ┌───────┴────────┐
        ↓                ↓
   Vector Store     Graph Store
   (embeddings)     (imports/calls)
        ↓                ↓
        └───────┬────────┘
                ↓
        Hybrid Retrieval
                ↓
           ReAct Agent
        (think→act→observe)
                ↓
            Answer
```

See [docs/architecture.md](docs/architecture.md) for detailed design rationale.

---

## Current Status

**Stage 10: Web API + Interactive Demo** ✅ (All stages 0–10 complete)

Completed stages:
- Stage 0: Project Setup
- Stage 1: Ingestion (repo scanning, file reading)
- Stage 2: AST Parsing (extract functions, classes, imports, calls)
- Stage 3: Code-aware Chunking
- Stage 4: Vector Store (local embeddings + ChromaDB)
- Stage 5: Graph Store (NetworkX, import/call edges)
- Stage 6: Hybrid Retrieval
- Stage 7: Agent (ReAct loop, tools, Gemini SDK)
- Stage 8: CLI Interface
- Stage 9: Evaluation (vector-only vs. hybrid)
- **Stage 10: Web API + Interactive Demo**

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/repomind.git
cd repomind

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode (includes dev dependencies: pytest, pytest-cov)
pip install -e ".[dev]"
```

**Requires**: Python ≥ 3.9. No GPU required—all components run on CPU.

### Optional: Gemini Free Tier

For LLM-powered reasoning (the default), set your Google AI Studio API key:

```bash
export GEMINI_API_KEY="your_key_here"  # Linux/macOS
$env:GEMINI_API_KEY="your_key_here"    # Windows PowerShell
```

If unset, RepoMind falls back to a **zero-cost local provider** (`free-local`) that emits deterministic stub answers—useful for demos and CI without any API key.

---

## CLI Usage

After installation, the `repomind` command is available:

```bash
repomind --help
```

Two subcommands:

| Command | Purpose |
|---------|---------|
| `repomind index <repo_path>` | Scan, parse, chunk, embed, and build graph index |
| `repomind ask <repo_path> "<question>"` | Ask a natural language question about the codebase |

Both support `-v/--verbose` and `--debug` for detailed output.

---

## The `index` Command

Builds a complete searchable index of a Python repository.

```bash
# Basic usage
repomind index ./my-python-project

# With verbose progress
repomind index ./my-python-project -v

# With debug tracebacks on error
repomind index ./my-python-project --debug
```

**What it does**:
1. Scans for all `.py` files (respects `.gitignore` via `pathspec`)
2. Reads each file and parses its AST to extract functions, classes, imports
3. Chunks code on syntactic boundaries (no mid-function splits)
4. Embeds chunks with a local `sentence-transformers` model (default: `all-MiniLM-L6-v2`)
5. Stores embeddings in a local ChromaDB vector store (default: `./.repomind_data/vectorstore`)
6. Builds an import/call dependency graph using NetworkX
7. Persists the graph as a pickle file at `<repo>/.repomind_graph.pickle`

**Output example** (`-v`):
```
[INFO] Validating repository at: /path/to/my-python-project
[INFO] Stage 1: Scanning Python files...
[INFO] Found 47 Python files.
[INFO] Stage 2 & 3: Reading, parsing AST, and chunking code...
[INFO] Extracted 213 code chunks across 42 files.
[INFO] Stage 4: Generating embeddings and populating vector store...
[INFO] Vector store populated. Total items: 213
[INFO] Stage 5: Building code dependency graph...
[INFO] Graph store built with 187 nodes and 342 edges.
[SUCCESS] Successfully indexed repository: /path/to/my-python-project
```

**Auto-indexing**: The `ask` command automatically runs `index` if the vector store is empty, so explicit `index` is only needed for re-indexing after code changes.

---

## The `ask` Command

Query the indexed codebase with natural language.

```bash
# Basic usage
repomind ask ./my-python-project "How does the authentication flow work?"

# Override LLM provider
repomind ask ./my-python-project "What does the UserService do?" --provider free-local

# Override model (Gemini only)
repomind ask ./my-python-project "Explain the database schema." --model gemini-1.5-pro

# Tune retrieval
repomind ask ./my-python-project "Find all API routes." --top-k 20 --depth 3

# Limit agent loop iterations
repomind ask ./my-python-project "Debug this error." --max-iterations 5

# Verbose: see each tool call the agent makes
repomind ask ./my-python-project "How is config loaded?" -v
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `gemini` | LLM provider: `gemini`, `free-local` |
| `--model` | `gemini-1.5-flash` | Model name (Gemini only) |
| `--top-k` | `10` | Number of semantic search results |
| `--depth` | `2` | Graph traversal depth |
| `--max-iterations` | `10` | Maximum ReAct loop iterations |
| `-v, --verbose` | off | Show agent tool traces |
| `--debug` | off | Show full error tracebacks |

---

## Web Interface (Stage 10)

RepoMind includes a complete web application with real-time progress tracking and streaming responses.

### Running with Docker

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Start the application
docker-compose up

# Access the web interface
open http://localhost:5173
```

The web UI provides:
- **Repository indexing** with real-time progress updates
- **Job status polling** with visual progress bars
- **Query interface** with Server-Sent Events streaming
- **Agent tool trace** showing which tools were used
- **Multi-repository support** (concurrent indexing and querying)

### Development Mode

Run backend and frontend separately for development:

```bash
# Terminal 1: Start API server
python -m repomind.api.main

# Terminal 2: Start frontend dev server
cd frontend
npm install
npm run dev
```

API runs on `http://localhost:8000`, frontend on `http://localhost:5173` with automatic proxy to the API.

### API Documentation

See [docs/stage10.md](docs/stage10.md) for complete API reference including:
- REST endpoints (`/api/repos/index`, `/api/repos/{id}/status`, `/api/repos/{id}/ask`)
- SSE streaming protocol
- Security measures and rate limiting
- Docker deployment guide

---

## Example Questions

| Flag | Description | Default |
|------|-------------|---------|
| `--provider {gemini,free-local}` | LLM provider override | `gemini` (from config) |
| `--model <name>` | LLM model name override | `gemini-1.5-flash` (from config) |
| `--top-k <int>` | Semantic search result count | `10` (from config) |
| `--depth <int>` | Graph traversal depth (0 = vector only) | `2` (from config) |
| `--max-iterations <int>` | Max ReAct loop iterations | `10` (from config) |
| `-v, --verbose` | Print each agent tool step | off |
| `--debug` | Full exception tracebacks on error | off |

### Output example

```bash
$ repomind ask ./my-python-project "What does the hello function do?" -v
[INFO] Analyzing repo: my-python-project
[INFO] LLM Provider: gemini (model: gemini-1.5-flash)
[INFO] Retrieval config: top_k=10, depth=2, max_iterations=10
[INFO] Starting ReAct agent query: 'What does the hello function do?'
[1] semantic_search -> Summary length: 247
[2] get_definition -> Summary length: 183
============================================================
Answer:
============================================================
The `hello()` function in `demo.py` returns a greeting string.
It takes a single `name` parameter and returns "Hello, <name>!".
============================================================
```

---

## Configuration

All configuration lives in `repomind/config.py` and supports environment variable overrides:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `LLM_PROVIDER` | `REPOMIND_LLM_PROVIDER` | `gemini` | `gemini` or `free-local` |
| `GEMINI_MODEL` | `GEMINI_MODEL` | `gemini-1.5-flash` | Free-tier model name |
| `GEMINI_API_KEY` | `GEMINI_API_KEY` | *(none)* | Required for Gemini |
| `VECTOR_STORE_PATH` | `REPOMIND_VECTOR_STORE_PATH` | `./.repomind_data/vectorstore` | ChromaDB persistence dir |
| `VECTOR_SEARCH_TOP_K` | `REPOMIND_VECTOR_TOP_K` | `10` | Default semantic results |
| `GRAPH_TRAVERSAL_DEPTH` | `REPOMIND_GRAPH_DEPTH` | `2` | Default graph expansion depth |
| `AGENT_MAX_ITERATIONS` | `REPOMIND_AGENT_MAX_ITER` | `10` | Default ReAct loop limit |

**Precedence**: CLI flags > Environment variables > Hardcoded defaults.

**Example** (non-interactive CI run):
```bash
REPOMIND_LLM_PROVIDER=free-local \
REPOMIND_VECTOR_TOP_K=5 \
repomind ask ./repo "What does X do?"
```

---

## Examples

### Index a local project and ask about it

```bash
cd /path/to/your/python/project
repomind index . -v
repomind ask . "What are the main entry points?"
repomind ask . "How is the database connection managed?"
```

### Use free-local provider (no API key needed)

```bash
repomind ask ./repo "Explain the class hierarchy." --provider free-local
```

### Tune retrieval for large codebases

```bash
repomind ask ./large-repo "Find all error handling patterns." \
  --top-k 20 --depth 3 --max-iterations 15 -v
```

### Re-index after changes

```bash
# After editing code, re-run index
repomind index ./repo -v

# Then ask again
repomind ask ./repo "What changed in the auth module?"
```

---

## Troubleshooting

### "Repository path does not exist"
Ensure the path is absolute or relative to your current working directory and points to a directory (not a file).

### "No Python files found in repository"
The directory must contain at least one `.py` file. Check that `.gitignore` isn't excluding your source (RepoMind respects `.gitignore`).

### "Gemini provider is unavailable... Falling back to 'free-local'"
Either `GEMINI_API_KEY` is not set, or the `google-genai` package is not installed. Set the key or install the package, or explicitly use `--provider free-local`.

### "Vector store is empty. Auto-indexing repository first..."
This is normal on first `ask`. The CLI runs `index` automatically. If it fails, run `index` manually with `-v` to see the error.

### Slow first run
The embedding model (`all-MiniLM-L6-v2`) downloads on first use (~90 MB). Subsequent runs are fast. A Hugging Face token (`HF_TOKEN`) speeds up the download but is not required.

### Graph file missing on `ask`
If `.repomind_graph.pickle` was deleted, `ask` falls back to vector-only retrieval. Re-run `index` to rebuild the graph.

---

## Current Limitations

- **Python only**: RepoMind parses `.py` files using the stdlib `ast` module. Other languages are not supported.
- **No incremental indexing**: `index` rebuilds the entire vector store and graph. For large repos, this can take minutes.
- **Local-only storage**: ChromaDB and the graph pickle live on the local filesystem—no remote/shared backend.
- **Free-tier rate limits**: Gemini free tier has RPM/TPM limits. Heavy use may hit quotas.
- **Local provider is a stub**: `--provider free-local` returns canned answers; it does not actually reason over code.
- **No auth**: The CLI has no authentication; anyone with filesystem access can query an indexed repo.
- **Stage 9 (Evaluation) pending**: No built-in benchmark to compare vector-only vs. hybrid retrieval quality yet.

---

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_scanner.py
pytest tests/test_cli.py

# Run with verbose output
pytest -v

# CLI-only coverage
pytest tests/test_cli.py --cov=repomind.cli --cov-report=term-missing
```

---

## Design Principles

1. **Modular**: Each subsystem is isolated and testable
2. **Understandable**: Code reads like the surrounding code—production quality but clear
3. **Explicit over magical**: No hidden abstractions; implement core logic ourselves
4. **Local-first**: Works without external API dependencies for embeddings/storage
5. **Configurable**: LLM provider, embedding model, and retrieval strategy are pluggable

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Google Gemini (free tier) | Zero API cost |
| Embeddings | sentence-transformers | Local, free, swappable |
| Vector Store | ChromaDB | Local persistence, ANN search |
| Graph | NetworkX | Standard library, no vendor lock-in |
| AST | Python `ast` stdlib | Core language feature |
| Agent | Custom ReAct loop | Educational, transparent |
| Testing | pytest | Standard |

---

## Project Structure

```
RepoMind/
├── repomind/              # Main package
│   ├── ingestion/         # Repository scanning
│   ├── parsing/           # AST → structured code elements
│   ├── chunking/          # Code-aware segmentation
│   ├── vectorstore/       # Embeddings + similarity search
│   ├── graph/             # Code relationship graph
│   ├── retrieval/         # Hybrid retrieval
│   ├── agent/             # ReAct agent + tools
│   ├── eval/              # Evaluation harness
│   └── config.py          # Central configuration
├── tests/                 # Mirrors repomind/ structure
├── docs/                  # Architecture documentation
└── eval_data/             # Evaluation datasets (added later)
```

---

## Contributing

This is a portfolio project built for learning and demonstration. Contributions that improve clarity, add tests, or enhance documentation are welcome.

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

Built as a demonstration of:
- Production-quality Python architecture
- Hybrid retrieval techniques
- Transparent LLM agent design
- Evaluation-driven development

For questions or discussion, open an issue or reach out via the repository.

---

## Design Principles

1. **Modular**: Each subsystem is isolated and testable
2. **Understandable**: Code reads like the surrounding code—production quality but clear
3. **Explicit over magical**: No hidden abstractions; implement core logic ourselves
4. **Local-first**: Works without external API dependencies for embeddings/storage
5. **Configurable**: LLM provider, embedding model, and retrieval strategy are pluggable

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Google Gemini (free tier) | Zero API cost |
| Embeddings | sentence-transformers | Local, free, swappable |
| Vector Store | ChromaDB | Local persistence, ANN search |
| Graph | NetworkX | Standard library, no vendor lock-in |
| AST | Python `ast` stdlib | Core language feature |
| Agent | Custom ReAct loop | Educational, transparent |
| Testing | pytest | Standard |

---

## Project Structure

```
RepoMind/
├── repomind/              # Main package
│   ├── ingestion/         # Repository scanning
│   ├── parsing/           # AST → structured code elements
│   ├── chunking/          # Code-aware segmentation
│   ├── vectorstore/       # Embeddings + similarity search
│   ├── graph/             # Code relationship graph
│   ├── retrieval/         # Hybrid retrieval
│   ├── agent/             # ReAct agent + tools
│   ├── eval/              # Evaluation harness
│   └── config.py          # Central configuration
├── tests/                 # Mirrors repomind/ structure
├── docs/                  # Architecture documentation
└── eval_data/             # Evaluation datasets (added later)
```

---

## Configuration

Configuration will be managed through `repomind/config.py` with support for:
- Environment variables
- Configuration file overrides
- Sensible defaults

LLM provider is abstracted to allow switching from Gemini to other providers without changing core agent logic.

---

## Contributing

This is a portfolio project built for learning and demonstration. Contributions that improve clarity, add tests, or enhance documentation are welcome.

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

Built as a demonstration of:
- Production-quality Python architecture
- Hybrid retrieval techniques
- Transparent LLM agent design
- Evaluation-driven development

For questions or discussion, open an issue or reach out via the repository.
