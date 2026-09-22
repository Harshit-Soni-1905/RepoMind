# RepoMind

**AI-powered codebase understanding and retrieval system for Python repositories and Jupyter Notebooks.**

RepoMind helps developers understand unfamiliar codebases using **AST-based code analysis, code-aware chunking, semantic search, code graphs, hybrid retrieval, and an LLM-powered ReAct agent**.

Instead of sending an entire repository to an LLM, RepoMind builds a structured and searchable representation of the codebase and retrieves relevant context before generating an answer.

---

## ✨ Features

- **Code-aware repository indexing**
  - Supports Python repositories and Jupyter Notebooks
  - Respects `.gitignore` rules

- **AST-based code analysis**
  - Extracts functions, classes, imports, calls, decorators, docstrings, and source locations

- **Jupyter Notebook support**
  - Processes code and Markdown cells
  - Handles IPython magics
  - Ignores notebook outputs and attachments

- **Logical code chunking**
  - Chunks code around meaningful structures instead of arbitrary text boundaries

- **Semantic retrieval**
  - FastEmbed-based embeddings
  - Local ONNX Runtime inference
  - ChromaDB vector storage

- **Code graph**
  - NetworkX-based graph of files, symbols, imports, calls, definitions, and relationships

- **Hybrid retrieval**
  - Combines semantic search with structural graph information

- **ReAct agent**
  - Tool-based reasoning loop for retrieving and analyzing repository context

- **Web application**
  - React + TypeScript frontend
  - FastAPI backend
  - Real-time indexing progress
  - Streaming query responses

- **CLI**
  - Index repositories
  - Ask natural-language questions about codebases
  - Debug and verbose modes

- **Retrieval evaluation**
  - Precision@k
  - Recall@k
  - MRR
  - nDCG@k
  - R-Precision

- **Docker support**
  - Containerized backend
  - Docker Compose setup

- **Automated testing**
  - Tests covering ingestion, parsing, retrieval, graph construction, agent behavior, API, CLI, and evaluation

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │   Python Repo    │
                         │    / Notebook    │
                         └────────┬─────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │ Repository Scanner │
                        └─────────┬──────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              Python Parser              Notebook Parser
                 AST                    Code + Markdown
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         Code-Aware Chunks
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
               FastEmbed                  Code Graph
                    │                     NetworkX
                    ▼                           │
                ChromaDB                      │
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         Hybrid Retrieval
                                  │
                                  ▼
                            ReAct Agent
                                  │
                                  ▼
                             Gemini LLM
                                  │
                                  ▼
                              Answer
