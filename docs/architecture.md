# RepoMind Architecture

This document captures the architectural decisions, component responsibilities, and data flow for RepoMind.

---

## Design Philosophy

RepoMind is built to answer the question: **"How do we build a code understanding system that goes beyond naive RAG?"**

### Core Principles

1. **Code-aware, not text-aware**: We chunk on syntactic boundaries (functions, classes), not arbitrary token windows
2. **Hybrid retrieval**: Semantic search finds conceptually related code; graph traversal finds structurally related code
3. **Transparent agent**: We implement the ReAct loop ourselves to understand every decision the agent makes
4. **Zero external cost**: Free-tier LLM, local embeddings, local storage
5. **Evaluated**: We measure what we build—retrieval quality and answer faithfulness

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       REPOMIND SYSTEM                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   INGESTION     │
                    │  - scanner.py   │
                    │  - file_reader.py│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   AST PARSING   │
                    │  - ast_parser.py│
                    │  - models.py    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   CHUNKING      │
                    │  - chunker.py   │
                    └────────┬────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
        ┌─────────────┐             ┌─────────────┐
        │ VECTOR STORE│             │ GRAPH STORE │
        │- embedder.py│             │- builder.py │
        │- store.py   │             │- store.py   │
        └──────┬──────┘             └──────┬──────┘
               │                           │
               └─────────────┬─────────────┘
                             ▼
                    ┌─────────────────┐
                    │ HYBRID RETRIEVAL│
                    │  - hybrid.py    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   REACT AGENT   │
                    │  - agent.py     │
                    │  - tools.py     │
                    │  - prompts.py   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   INTERFACE     │
                    │  (CLI / eval)   │
                    └─────────────────┘
```

---

## Component Responsibilities

### 1. Ingestion (`repomind/ingestion/`)

**Purpose**: Find all Python files in a repository, read them safely.

- `scanner.py`: Walk directory tree, discover `.py` files, respect `.gitignore`
- `file_reader.py`: Read files with encoding detection, size limits, error handling

**Key decisions**:
- Uses `pathspec` for `.gitignore` matching (complex edge cases)
- Skips common non-source directories (`venv/`, `__pycache__/`, `.git/`)
- Fails gracefully on binary files, encoding errors, permission issues

---

### 2. Parsing (`repomind/parsing/`)

**Purpose**: Convert Python source code into structured representations.

- `ast_parser.py`: Use Python's `ast` module to extract functions, classes, imports
- `models.py`: Data classes representing parsed elements (FunctionDef, ClassDef, ImportInfo)

**Key decisions**:
- We parse, not tokenize—we care about structure, not syntax highlighting
- Extract docstrings, signatures, decorators for rich metadata
- Capture line ranges so we can map chunks back to source

**Why not use existing tools?**  
Tools like `jedi` or `rope` are for IDE features (completion, refactoring). We need a simpler, more transparent AST visitor pattern that students can read and understand in 30 minutes.

---

### 3. Chunking (`repomind/chunking/`)

**Purpose**: Segment parsed code into semantically meaningful units.

- `chunker.py`: Create chunks from parsed elements with rich metadata

**Chunking strategy**:
- One chunk per function
- One chunk per class (includes class docstring + method signatures, but not full method bodies unless small)
- Module-level docstrings and imports as context chunks
- Each chunk knows its file, line range, type, and relationships

**Why this matters**:  
Generic RAG systems chunk on token count. If a function spans the boundary, half the logic is in one chunk, half in another. Semantic search fails because neither chunk is coherent.

---

### 4. Vector Store (`repomind/vectorstore/`)

**Purpose**: Turn text into vectors, search by similarity.

- `embedder.py`: Generate embeddings using `sentence-transformers`
- `store.py`: Index and query embeddings using ChromaDB

**Key decisions**:
- Model: `all-MiniLM-L6-v2` (384 dims, fast, local, free)
- ChromaDB for persistence and ANN search (HNSW index)
- Metadata filtering: can restrict search to specific files or types

**Why ChromaDB?**  
FAISS is faster but requires manual persistence and has no metadata filtering. Pinecone/Weaviate are cloud services (we want local). ChromaDB is the sweet spot: local, persistent, metadata-aware.

---

### 5. Graph Store (`repomind/graph/`)

**Purpose**: Model structural relationships in the codebase.

- `models.py`: Data models for nodes (`NodeType`), edges (`EdgeType`), and `GraphNode`.
- `builder.py`: Build directed graph from import, definition, and containment relationships.
- `traversal.py`: Query graph for dependencies, dependents, and direct neighbors (`GraphTraverser`).

**Graph structure**:
- Nodes: files (`FILE`), classes (`CLASS`), top-level functions (`FUNCTION`), class methods (`METHOD`)
- Edges: `DEFINES` (file defines class/function), `CONTAINS` (class contains method), `IMPORTS` (module imports module)
- Node IDs: Deterministic POSIX paths (`"path/to/file.py"`, `"path/to/file.py::Symbol"`, `"path/to/file.py::Class.method"`)

**Key decisions**:
- NetworkX `DiGraph` for graph data structure
- Import resolution handles absolute package paths and relative dot imports (`.module`, `..pkg`) while safely ignoring unresolved stdlib/external packages
- `GraphTraverser` provides BFS traversal with edge type filtering and depth constraints (`depth=1`, `depth=2`, `depth=None`)

**Why a graph?**  
If you search for "authentication", you find `auth/login.py`. But what about the crypto utilities it imports? Or the database models it writes to? The graph lets the agent traverse dependencies rather than relying on keyword overlap.

---

### 6. Hybrid Retrieval (`repomind/retrieval/`)

**Purpose**: Combine vector search with graph traversal for better recall.

- `hybrid.py`: Merge ranked results from vector store and graph store

**Algorithm** (Reciprocal Rank Fusion or weighted merge):
1. Vector search returns top-k chunks by cosine similarity
2. Graph traversal returns nodes related to top vector results
3. Merge and re-rank using RRF: `score = 1/(rank_vector + k) + 1/(rank_graph + k)`

**Why hybrid?**  
Vector-only: misses structurally related code without semantic overlap  
Graph-only: no semantic understanding, just follows imports blindly  
Hybrid: finds both "what mentions auth" and "what does auth.py depend on"

---

### 7. Agent (`repomind/agent/`)

**Purpose**: Reason about which tools to call and when, synthesize final answer.

- `agent.py`: ReAct loop (Think → Act → Observe → repeat)
- `tools.py`: Tool definitions (semantic_search, graph_traverse, read_file)
- `prompts.py`: System/user prompt templates

**ReAct loop structure**:
```
while not done:
    thought = llm("What should I do next?")
    if should_answer(thought):
        answer = llm("Provide final answer")
        break
    action, args = parse_action(thought)
    observation = tools[action](**args)
    add_to_context(thought, action, observation)
```

**Key decisions**:
- LLM: Google Gemini free tier (configurable via provider abstraction)
- Prompt includes tool descriptions, conversation history, and current observation
- Max 10 iterations to prevent infinite loops
- Tools return structured data, not raw text dumps

**Why implement this ourselves?**  
LangChain abstracts the loop, making it hard to debug or understand. Implementing ReAct in ~150 lines teaches you exactly how agents work—critical for interviews.

---

### 8. CLI Interface (`repomind/cli/`)

**Purpose**: User-facing command line for indexing and querying repositories.

- `main.py`: Argument parsing (`argparse` with subcommands) and dispatch
- `commands.py`: Orchestration logic for `index` and `ask` commands
- `formatter.py`: Structured terminal output (info/warning/error/step/answer)

**Commands**:

| Command | Description |
|---------|-------------|
| `repomind index <path>` | Full index build: scan → parse → chunk → embed → graph |
| `repomind ask <path> "<question>"` | Query indexed repo via ReAct agent |

**Key design decisions**:

1. **Subcommand structure** — `add_subparsers(dest="command")` intentionally not `required=True` so `main()` can print help and exit 1 on no-argument invocation. This matches standard CLI conventions (`docker`, `git`, `aws`).

2. **Thin orchestration layer** — `commands.py` does no heavy lifting. It wires existing Stages 1–7 subsystems together:
   - `index` → `RepoScanner` → `FileReader` → `parse_source` → `CodeChunker` → `CodeEmbedder` + `VectorStore` → `GraphBuilder` → `save_graph`
   - `ask` → `VectorStore` + `GraphTraverser` → `HybridRetriever` → `LLMProvider` + `ToolRegistry` + `ReActAgent`

3. **Configuration precedence** — CLI flags override environment variables override hardcoded defaults (`config.py`). All knobs (provider, model, top-k, depth, max-iterations) are independently overridable.

4. **Graph persistence** — NetworkX 3.x removed `read_gpickle`/`write_gpickle`. The CLI serializes the `DiGraph` directly with `pickle` because node attributes carry `GraphNode` dataclasses and `NodeType` enums that are not JSON-serializable. Per-repo file: `<repo>/.repomind_graph.pickle`.

5. **Auto-indexing** — If `ask` finds an empty vector store (`count() == 0`), it silently runs `index` first. This makes the CLI zero-config for new users.

6. **LLM provider abstraction** — `LLMProvider` ABC with two implementations:
   - `GeminiProvider` (free tier, requires `GEMINI_API_KEY`)
   - `FreeLocalProvider` (zero cost, deterministic stub answers)
   The CLI falls back to `FreeLocalProvider` if Gemini is unavailable, so demos/CI never require an API key.

7. **Error handling** — Exceptions are caught at the command boundary, formatted via `CLIFormatter.error()`, and return exit code 1. `--debug` prints full tracebacks.

8. **Syntax error reporting** — `ast_parser.parse` signals syntax errors by returning a `ParsedFile` with `has_syntax_error=True` rather than raising. The CLI checks this flag and warns the user, rather than silently indexing the file as empty.

**Output formatting** (`CLIFormatter`):
- `[INFO]` — progress messages
- `[WARNING]` — non-fatal issues (skipped files, fallback provider)
- `[SUCCESS]` — command completed
- `[ERROR]` — fatal errors (with optional `--debug` traceback)
- `[N] tool_name -> summary` — verbose agent steps
- `=== Answer ===` block — final answer

---

### 9. Evaluation (`repomind/eval/`)

**Purpose**: Measure retrieval quality and answer correctness.

- `dataset.py`: Load/manage evaluation Q&A pairs
- `metrics.py`: Compute retrieval metrics (precision@k, recall@k, MRR) and answer quality

**Evaluation approach**:
1. Create ground-truth Q&A pairs with known relevant chunks
2. Compare retrieval results: vector-only vs. hybrid
3. Measure: precision, recall, mean reciprocal rank
4. Optionally: LLM-as-judge for answer faithfulness

---

## Data Flow: Ingestion to Answer

### Phase A: Indexing (one-time per repo)

```
1. scanner.scan("path/to/repo")
   → ["main.py", "utils/helpers.py", ...]

2. For each file:
   file_reader.read(path) → source_code
   ast_parser.parse(source_code) → [FunctionDef(...), ClassDef(...), ImportInfo(...)]

3. chunker.chunk(all_parsed_elements)
   → [CodeChunk(text="def foo()...", metadata={...}), ...]

4. For each chunk:
   embedder.embed(chunk.text) → vector
   vector_store.add(vector, metadata)

5. graph_builder.build(all_parsed_elements)
   → NetworkX DiGraph(nodes=modules/functions, edges=imports)
```

### Phase B: Query (per question)

```
6. User asks: "How does authentication work?"

7. Agent ReAct loop:

   THINK: "I should search for auth-related code"
   ACTION: semantic_search("authentication login user")
   
   → hybrid_retrieval:
       - embedder.embed(query) → query_vector
       - vector_store.query(query_vector, top_k=10) → vector_results
       - For top vector results, graph_store.neighbors(result.module) → graph_results
       - merge(vector_results, graph_results) → ranked_chunks
   
   OBSERVE: [chunk from auth/login.py, chunk from models/user.py, ...]
   
   THINK: "auth/login.py looks important. What does it import?"
   ACTION: graph_traverse("auth.login", depth=1)
   
   → graph_store.neighbors("auth.login")
   
   OBSERVE: [auth.login → models.user, auth.login → utils.crypto]
   
   THINK: "I should read utils/crypto.py to understand hashing"
   ACTION: read_file("utils/crypto.py")
   
   OBSERVE: <file contents>
   
   THINK: "I have enough context now"
   ANSWER: "Authentication works by..."
```

---

## Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LLM** | Google Gemini (free tier) | Zero API cost, good reasoning ability, configurable |
| **Embeddings** | `sentence-transformers` | Local, no API calls, good quality, easy to swap |
| **Vector store** | ChromaDB | Local persistence, metadata filtering, ANN search |
| **Graph library** | NetworkX | Standard, simple API, serialization built-in |
| **Chunking** | Custom AST-based | Core differentiator—must implement ourselves |
| **Agent framework** | None (raw SDK) | Educational goal—understand the loop, don't hide it |
| **Testing** | pytest | Standard Python testing |

---

## Build vs. Use Decisions

### We Build:
- AST visitor for extracting code elements
- Chunking strategy (syntactic boundaries)
- Graph construction from imports/calls
- Hybrid retrieval merging logic
- ReAct agent loop
- Evaluation harness

### We Use Libraries:
- `ast` (stdlib): parsing Python source
- `sentence-transformers`: embedding generation
- `chromadb`: vector indexing and search
- `networkx`: graph data structure
- `google-genai`: Official Gemini API client
- `pathspec`: `.gitignore` parsing

**Why this split?**  
We implement what teaches us about code understanding and agent design. We use libraries for infrastructure (embeddings, storage) where reimplementation adds no insight.

---

## Staged Implementation

| Stage | Deliverable | Why This Order |
|-------|-------------|----------------|
| 0 | Project skeleton | Foundation for all work |
| 1 | Ingestion | Can't parse what we can't read |
| 2 | AST Parsing | Structured representation required for chunking |
| 3 | Chunking | Need chunks before we can embed them |
| 4 | Vector Store | Enables first retrieval method |
| 5 | Graph Store | Enables second retrieval method |
| 6 | Hybrid Retrieval | Combines both methods |
| 7 | Agent | Uses retrieval to answer questions |
| 8 | CLI | User-facing interface |
| 9 | Evaluation | Validates the whole system |

---

### Stage 8: CLI Interface (Complete)

The CLI exposes the full pipeline through two subcommands:

- **`repomind index <repo>`** — Scans `.py` files, parses AST, chunks on syntactic boundaries, embeds with `sentence-transformers`, stores in ChromaDB, builds import graph with NetworkX, persists graph as pickle.
- **`repomind ask <repo> "<question>"`** — Runs the ReAct agent with hybrid retrieval (vector + graph). Auto-indexes if the vector store is empty.

**Key implementation details**:
- `argparse` subparsers with `dest="command"` (not `required`) so `main()` handles the no-command case
- Configuration precedence: CLI flags > env vars > `config.py` defaults
- Graph persisted with `pickle` (NetworkX 3.x removed `read/write_gpickle`; node attrs carry `GraphNode` dataclasses/enums)
- `FreeLocalProvider` fallback when Gemini unavailable — demos/CI work with $0 API budget
- Syntax errors reported via `ParsedFile.has_syntax_error` flag (parser returns it, doesn't raise)

---

## Open Questions (to revisit during implementation)

1. **Chunk size**: How much context to include around each function/class?
2. **Graph edges**: Should we model function calls in addition to imports?
3. **Retrieval fusion**: RRF vs. weighted linear combination?
4. **Agent tools**: Should we add a "find_definition" tool?
5. **Evaluation dataset**: How many Q&A pairs give us statistical confidence?

These will be answered as we build each subsystem.

---

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Reciprocal Rank Fusion for hybrid search](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- Python `ast` module documentation
- ChromaDB documentation
- NetworkX documentation

---

*Last updated: Stage 8 Completion (2026-08-30)*
