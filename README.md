
---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- Node.js and npm
- Docker (optional)
- Gemini API key

No GPU required.

### 1. Clone the repository

```bash
git clone https://github.com/Harshit-Soni-1905/RepoMind.git
cd RepoMind
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

> Never commit `.env` files or API keys to Git.

---

## 💻 CLI Usage

```bash
repomind --help
```

**Index a repository**
```bash
repomind index ./my-project
repomind index ./my-project -v        # verbose
repomind index ./my-project --debug   # debug
```

**Ask questions**
```bash
repomind ask ./my-project "How does authentication work?"
repomind ask ./my-project "Where is the database initialized?"
repomind ask ./my-project "Which functions call process_payment()?"
```

---

## 🌐 Web Application

Start the backend:
```bash
python -m repomind.api.main
```
Runs at `http://localhost:8000` (docs at `/docs`).

Start the frontend:
```bash
cd frontend
npm install
npm run dev
```

The web app supports repository submission, real-time indexing progress, natural-language queries, and streaming agent responses.

---

## 🐳 Docker

```bash
docker compose up --build
```

Stop services:
```bash
docker compose down
```

---

## 🔌 API Overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Backend health check |
| `/api/repos/index` | POST | Start repository indexing |
| `/api/repos/{repo_id}/status` | GET | Get indexing status |
| `/api/repos/{repo_id}/ask` | POST | Ask a question about a repository |

Real-time updates are delivered via Server-Sent Events. Full interactive docs at `http://localhost:8000/docs`.

---

## 🧪 Testing

```bash
pytest                                          # run all tests
pytest -v                                       # verbose
pytest tests/test_scanner.py                    # single file
pytest --cov=repomind --cov-report=term-missing # coverage
```

---

## 📊 Evaluation

RepoMind includes a retrieval evaluation framework:

| Metric | Measures |
|---|---|
| Precision@k | Relevant results among retrieved results |
| Recall@k | Coverage of all relevant results retrieved |
| MRR | Rank of the first relevant result |
| nDCG@k | Ranking quality |
| R-Precision | Precision at the number of relevant documents |

---

## ⚠️ Current Limitations

- **Python-focused** — parsing pipeline currently targets Python and Python-based notebooks
- **No incremental indexing** — full re-index on changes, not just changed files
- **Local storage** — vector and graph data are stored locally
- **LLM dependency** — reasoning/generation relies on Gemini availability and quotas
- **Retrieval quality** depends on chunking, embedding quality, query phrasing, top-k, and repo structure
- **Possible hallucination** — grounded context improves accuracy but doesn't guarantee it

---

## 🗺️ Future Improvements

- Incremental repository indexing
- Support for additional languages
- Retrieval reranking
- Persistent managed vector storage
- Authentication and authorization
- Line-level answer grounding
- Git history-aware code understanding

---

## 🔒 Security

- Never commit API keys or `.env` files
- Repository contents are treated as untrusted input
- Repository code is not executed during indexing
- Public deployments should add authentication and durable storage
- Prompt-injection defenses should be considered for untrusted repo content

---

## 📄 License

Released under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Harshit Soni**
B.Tech Computer Science & Engineering, ABES Engineering College, Ghaziabad
GitHub: [@Harshit-Soni-1905](https://github.com/Harshit-Soni-1905)

---

⭐ If you find RepoMind useful, consider starring the repo or opening an issue with feedback.
