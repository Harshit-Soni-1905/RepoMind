# 🧠 REPOMIND | AI Codebase Understanding Agent

### Agentic RAG System for Python Repositories and Jupyter Notebooks

<p align="center">
  An AI-powered system that understands codebases using AST parsing, code graphs, hybrid retrieval, and an LLM-powered ReAct agent.
</p>

<p align="center">
  <a href="https://github.com/Harshit-Soni-1905/RepoMind">
    📂 <b>Repository</b>
  </a>
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">

<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">

<img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">

<img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript">

<img src="https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini">

<img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB">

<img src="https://img.shields.io/badge/NetworkX-3776AB?style=for-the-badge&logo=graphql&logoColor=white" alt="NetworkX">

<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">

<img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">

<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">

</p>

---

## 📌 Overview

**RepoMind** helps developers understand unfamiliar codebases without dumping an entire repository into an LLM's context window. It builds a structured, searchable representation of a codebase — combining AST-based parsing, semantic embeddings, and a code graph — and retrieves only the relevant context before an LLM-powered agent answers a question.

```text
Repository
  ↓
Parsing (AST + Notebooks)
  ↓
Code-Aware Chunking
  ↓
Embeddings + Code Graph
  ↓
Hybrid Retrieval
  ↓
ReAct Agent
  ↓
Answer
```

---

## 🎯 Objective

The objective is to move beyond naive "paste the whole repo into the prompt" approaches and build a system that retrieves *precisely relevant* code context using two complementary signals:

```text
Semantic Retrieval  → conceptually related code (embeddings)
Graph Retrieval      → structurally related code (imports/calls/definitions)
```

---

## 🏗️ Architecture

```text
Python Repo / Notebook
        │
        ▼
  Repository Scanner
        │
   ┌────┴────┐
   ▼         ▼
Python     Notebook
Parser     Parser
(AST)   (Code+Markdown)
   └────┬────┘
        ▼
  Code-Aware Chunks
        │
   ┌────┴────┐
   ▼         ▼
FastEmbed   Code Graph
   │        (NetworkX)
   ▼            │
ChromaDB        │
   └────┬───────┘
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
```

The ReAct agent reasons about the question, calls retrieval tools (semantic search + graph traversal), observes the results, and iterates before producing a final grounded answer — rather than relying on one large uncontrolled context dump.

---

## 🧠 Hybrid Retrieval

| Strategy | Finds | How |
|---|---|---|
| **Semantic Retrieval** | Conceptually related code | Question → embedding → vector similarity search |
| **Graph Retrieval** | Structurally related code | File/symbol graph: imports, calls, definitions |

Combining both gives more reliable context than either alone — semantic search catches conceptual matches even with different wording, while graph retrieval catches direct dependencies semantic search would miss.

---

## ✨ Features

- **AST-based code analysis** — functions, classes, imports, calls, decorators, docstrings
- **Jupyter Notebook support** — code + Markdown cells, IPython magics, ignores outputs
- **Logical chunking** — splits around meaningful code structures, not arbitrary text
- **Code graph** — NetworkX graph of files, symbols, imports, calls, definitions
- **ReAct agent** — tool-based reasoning loop for retrieval and analysis
- **Web app** — React + TypeScript frontend, FastAPI backend, real-time streaming
- **CLI** — index repos and ask natural-language questions
- **Retrieval evaluation** — Precision@k, Recall@k, MRR, nDCG@k, R-Precision
- **Docker support** — containerized backend with Docker Compose

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Backend | FastAPI |
| Frontend | React + TypeScript (Vite) |
| LLM | Google Gemini |
| Embeddings | FastEmbed + ONNX Runtime |
| Vector Store | ChromaDB |
| Code Graph | NetworkX |
| Code Parsing | Python `ast` |
| Agent | Custom ReAct agent |
| Real-time Updates | Server-Sent Events |
| Testing | pytest |
| Containerization | Docker |

---

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/Harshit-Soni-1905/RepoMind.git
cd RepoMind
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

Set up your environment:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Index a repository and ask a question:

```bash
repomind index ./my-project
repomind ask ./my-project "How does authentication work?"
```

Or run the web app:

```bash
python -m repomind.api.main        # backend → localhost:8000
cd frontend && npm install && npm run dev
```

Or with Docker:

```bash
docker compose up --build
```

---

## 🔌 API Overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Backend health check |
| `/api/repos/index` | POST | Start repository indexing |
| `/api/repos/{repo_id}/status` | GET | Get indexing status |
| `/api/repos/{repo_id}/ask` | POST | Ask a question about a repository |

Real-time updates stream via Server-Sent Events. Full interactive docs at `http://localhost:8000/docs`.

---

## 📊 Evaluation

| Metric | Measures |
|---|---|
| Precision@k | Relevant results among retrieved results |
| Recall@k | Coverage of all relevant results |
| MRR | Rank of the first relevant result |
| nDCG@k | Ranking quality |
| R-Precision | Precision at the number of relevant documents |

---

## ⚠️ Limitations

- Parsing pipeline currently focuses on **Python** and Python-based notebooks
- **No incremental indexing** — full re-index on changes, not just changed files
- Vector and graph data are stored **locally**
- Answer quality depends on Gemini's availability, quotas, and the quality of retrieved context
- Retrieved context improves grounding but does not guarantee fully correct answers

---

## 🔮 Future Improvements

- Incremental repository indexing
- Support for additional programming languages
- Retrieval reranking
- Persistent managed vector storage
- Authentication and authorization for public deployments
- Line-level answer grounding
- Git history-aware code understanding

---

## 🎓 Key Learnings

This project provided hands-on experience with:

- AST-based static code analysis
- Building and querying a code dependency graph
- Semantic search with embeddings + vector databases
- Designing a hybrid (semantic + structural) retrieval system
- Building a ReAct-style tool-using agent
- FastAPI + Server-Sent Events for streaming responses
- Retrieval evaluation methodology (Precision/Recall/MRR/nDCG)

---

## 📜 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Harshit Soni**

B.Tech Computer Science & Engineering

Interested in Artificial Intelligence, Machine Learning and AI Agents.

---

<p align="center">
  Built with Python • FastAPI • React • Gemini
</p>
