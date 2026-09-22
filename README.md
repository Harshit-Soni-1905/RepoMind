# RepoMind

**AI-powered codebase understanding and retrieval system for Python repositories and Jupyter Notebooks.**

RepoMind helps developers understand unfamiliar codebases using AST-based code analysis, code-aware chunking, semantic search, code graphs, hybrid retrieval, and an LLM-powered ReAct agent.

Instead of dumping an entire repository into an LLM's context window, RepoMind builds a structured, searchable representation of the codebase and retrieves only the relevant context before generating an answer.

---

## ✨ Features

- **Code-aware indexing** — parses Python repos and Jupyter Notebooks, respects `.gitignore`
- **AST-based analysis** — extracts functions, classes, imports, calls, decorators, docstrings, source locations
- **Notebook support** — processes code + Markdown cells, handles IPython magics, ignores outputs/attachments
- **Logical chunking** — splits code around meaningful structures, not arbitrary text boundaries
- **Semantic retrieval** — FastEmbed embeddings, local ONNX Runtime inference, ChromaDB vector store
- **Code graph** — NetworkX graph of files, symbols, imports, calls, and definitions
- **Hybrid retrieval** — combines semantic search with graph-based structural retrieval
- **ReAct agent** — tool-using reasoning loop for retrieving and analyzing repo context
- **Web app** — React + TypeScript frontend, FastAPI backend, real-time indexing, streaming responses
- **CLI** — index repos and ask natural-language questions, with debug/verbose modes
- **Retrieval evaluation** — Precision@k, Recall@k, MRR, nDCG@k, R-Precision
- **Docker support** — containerized backend with Docker Compose
- **Automated tests** — ingestion, parsing, retrieval, graph, agent, API, CLI, evaluation

---

## 🏗️ Architecture
