# Stage 9: Evaluation and Benchmarking — Implementation Plan

> **Status**: Planning phase only — DO NOT IMPLEMENT until explicitly approved.  
> **Target**: Provide measurable evidence that RepoMind's hybrid retrieval (vector + graph) outperforms vector-only retrieval, and measure agent reasoning quality.

---

## 1. Evaluation Dataset Design

### 1.1 Repository Selection

Use **3–5 diverse, real-world Python repositories** as evaluation corpora:
- **Small** (~50–150 files): A focused library (e.g., `click`, `requests` subset)
- **Medium** (~200–500 files): A web framework component (e.g., `django` submodule, `fastapi`)
- **Large** (~1000+ files): A substantial codebase (e.g., `numpy` core, `pandas` internals)

All repos must be:
- Pure Python (no C extensions required for import)
- Have meaningful import/call graphs
- Licensed permissively (MIT/BSD/Apache-2.0)
- Pinned to a specific commit hash for reproducibility

### 1.2 Question Types & Distribution

Create **50–75 questions per repository** (150–375 total), categorized:

| Category | % | Description | Example |
|----------|---|-------------|---------|
| **Symbol lookup** | 20% | "Where is `AuthMiddleware` defined?" | Single symbol, direct answer |
| **Call chain** | 20% | "What calls `validate_token()`?" | Requires graph traversal upstream |
| **Dependency trace** | 15% | "What does `UserService` import?" | Graph traversal downstream |
| **Cross-file reasoning** | 20% | "How does login flow work across `auth/` and `api/`?" | Multi-hop, needs both vector + graph |
| **Behavioral / "How" questions** | 15% | "How are database migrations applied?" | Synthesized answer from multiple chunks |
| **Negative / "Not found"** | 10% | "Is there a WebSocket handler?" | Tests precision (should return empty) |

**Why this distribution?**
- Pure symbol lookup favors vector-only (easy baseline)
- Call chain/dependency trace *requires* graph edges
- Cross-file reasoning is where hybrid should shine
- Negative cases prevent gaming recall at precision's expense

### 1.3 Ground Truth Representation

**For retrieval evaluation** (deterministic, chunk-level):
```json
{
  "question_id": "repo1_042",
  "question": "What calls validate_token()?",
  "relevant_chunk_ids": [
    "auth/middleware.py::validate_token",
    "auth/handlers.py::LoginHandler.post",
    "api/routes.py::login_endpoint"
  ],
  "relevant_filepaths": ["auth/middleware.py", "auth/handlers.py", "api/routes.py"],
  "category": "call_chain",
  "difficulty": "medium"
}
```

**For answer evaluation** (requires judgment):
```json
{
  "question_id": "repo1_042",
  "reference_answer": "validate_token() is called by LoginHandler.post() in auth/handlers.py and by the login_endpoint in api/routes.py. It is not called directly by any middleware.",
  "key_facts": [
    "validate_token defined in auth/middleware.py",
    "called by LoginHandler.post",
    "called by login_endpoint",
    "not called by middleware"
  ],
  "required_citations": [
    "auth/middleware.py:15-25",
    "auth/handlers.py:40-55",
    "api/routes.py:10-18"
  ]
}
```

### 1.4 Avoiding "Too Easy" Evaluation Sets

1. **No trivial name matches**: Exclude questions where the answer is the exact function name appearing in the query
2. **Require multi-hop**: At least 40% of questions need ≥2 graph hops or ≥2 distinct files
3. **Adversarial distractors**: Include semantically similar but irrelevant code (e.g., `validate_user` vs `validate_token`)
4. **Hold-out validation**: Split questions into dev (30%) / test (70%) — dev for iteration, test for final report
5. **Human verification**: Every ground-truth chunk verified by a human (not LLM) against actual source

---

## 2. Retrieval Evaluation

### 2.1 Systems Under Comparison

| System | Retrieval Components | Configuration |
|--------|---------------------|---------------|
| **Vector-Only** | `VectorStore.search(query, top_k=k)` | `depth=0`, `top_k ∈ {5, 10, 20}` |
| **Hybrid** | `HybridRetriever.retrieve(query, top_k=k, depth=d)` | `top_k ∈ {5, 10, 20}`, `depth ∈ {1, 2, 3}` |

Both use **identical**:
- Vector store (same embeddings, same collection)
- Graph (same persisted pickle)
- Embedder model (`all-MiniLM-L6-v2`)
- Top-k limit per query

### 2.2 Metrics

| Metric | Formula | Deterministic? | Use Case |
|--------|---------|----------------|----------|
| **Precision@k** | \|relevant ∩ retrieved@k\| / k | ✅ Yes | "Of top k, how many are correct?" |
| **Recall@k** | \|relevant ∩ retrieved@k\| / \|relevant\| | ✅ Yes | "Of all relevant, how many found?" |
| **MRR (Mean Reciprocal Rank)** | mean(1 / rank of first relevant) | ✅ Yes | "How early is the first hit?" |
| **nDCG@k** | Normalized Discounted Cumulative Gain | ✅ Yes | Rank-aware, rewards ordering |
| **R-Precision** | Precision@R where R = \|relevant\| | ✅ Yes | Single-number summary |

**Primary metrics for comparison**: Precision@10, Recall@10, MRR
**Secondary**: nDCG@10, R-Precision

### 2.3 Evaluation Procedure

For each (repo, question, system, config):
1. Run retrieval → get ranked list of `HybridResultNode` (or `RetrievalResult` for vector-only)
2. Map retrieved `chunk_id` / `node_id` → ground-truth `relevant_chunk_ids`
3. Compute metrics at k ∈ {5, 10, 20}
4. Aggregate per-repo, per-category, and overall

---

## 3. Answer Evaluation

### 3.1 Metrics Taxonomy

| Metric | What It Measures | Deterministic? | Method |
|--------|------------------|----------------|--------|
| **Exact Match (EM)** | Answer string equals reference | ✅ | String normalize + compare |
| **F1 / ROUGE-L** | Token overlap with reference | ✅ | Standard NLP metrics |
| **Citation Precision** | % of cited sources actually relevant | ✅ | Check cited file:line vs ground truth |
| **Citation Recall** | % of required citations present | ✅ | Check required_citations covered |
| **Faithfulness** | Answer doesn't hallucinate beyond sources | ⚠️ LLM judge | "Does answer claim X not in sources?" |
| **Correctness** | Answer is factually correct | ⚠️ LLM judge | "Is answer right per code?" |
| **Completeness** | Answer covers all key_facts | ⚠️ LLM judge | "Are all key facts addressed?" |

### 3.2 Deterministic Metrics (No LLM Required)

**Citation Precision / Recall**: Exact match of cited `filepath:start_line-end_line` against ground-truth `required_citations`.

**Token Overlap (ROUGE-L)**: Compare generated answer tokens to reference answer tokens — no semantic understanding, but correlates with quality.

**Coverage of Key Facts (deterministic variant)**: Check if each `key_fact` string appears (substring match) in answer. Conservative but fully automatic.

### 3.3 LLM-Judge Metrics (Optional, Free-Tier)

Use **Gemini free-tier** as judge with a strict rubric prompt:

```text
You are an evaluator. Given a QUESTION, a GENERATED ANSWER, and the SOURCE CODE CHUNKS provided to the agent, score:
- FAITHFULNESS (1-5): Does the answer only claim what is supported by the provided chunks? (1 = hallucinates, 5 = fully grounded)
- CORRECTNESS (1-5): Is the answer factually correct per the actual repository code? (Requires you to know the code — use provided chunks)
- COMPLETENESS (1-5): Does the answer address all aspects of the question?

Return JSON: {"faithfulness": int, "correctness": int, "completeness": int, "reasoning": "..."}
```

**Constraint**: Judge uses **only the retrieved chunks** (not full repo) — evaluates whether the *agent's reasoning* from its context was sound, not whether it had perfect knowledge.

---

## 4. Agent Evaluation

### 4.1 Metrics

| Metric | Description | Deterministic? |
|--------|-------------|----------------|
| **Task Success Rate** | Agent produces a final answer (not max-iterations exhaustion or error) | ✅ |
| **Tool Selection Accuracy** | % of tool calls that are appropriate (semantic_search vs graph_traverse vs read_file) | ⚠️ LLM judge or heuristic |
| **Tool Call Efficiency** | Total tool calls / successful queries | ✅ |
| **Iterations to Answer** | Mean/median iterations before final answer | ✅ |
| **Early Termination Rate** | % ending in error / max-iterations / empty answer | ✅ |
| **Graph Tool Usage** | % of queries where `graph_traverse` was called (hybrid only) | ✅ |
| **Redundant Calls** | Repeated same tool with same args | ✅ |

### 4.2 Evaluation Procedure

Run the full **ReAct agent** (not just retrieval) on the same question set with both:
- `provider=free-local` (deterministic stub — for structural metrics only)
- `provider=gemini` (if API key available — for quality metrics)

Log every `AgentState` transition to JSONL for post-hoc analysis.

---

## 5. Benchmark Methodology

### 5.1 Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| Repository | Same commit hash per repo | Identical codebase |
| Embedding model | `all-MiniLM-L6-v2` | Same vector space |
| Vector store | Same ChromaDB collection | Same index |
| Graph | Same persisted pickle | Same structure |
| Top-k | Matched per experiment (5/10/20) | Fair comparison |
| Questions | Identical question set | Paired comparison |
| Random seed | Fixed for any stochasticity | Reproducibility |

### 5.2 Experimental Design

**Paired design**: Every question evaluated under *both* systems → paired t-test / Wilcoxon signed-rank.

**Conditions matrix**:
| Condition | Retrieval | Top-k | Depth |
|-----------|-----------|-------|-------|
| V5 | vector-only | 5 | 0 |
| V10 | vector-only | 10 | 0 |
| V20 | vector-only | 20 | 0 |
| H5d1 | hybrid | 5 | 1 |
| H10d2 | hybrid | 10 | 2 |
| H20d3 | hybrid | 20 | 3 |

**Primary comparison**: V10 vs H10d2 (matched top-k, typical depth)

---

## 6. Reproducibility

### 6.1 Experiment Manifest (JSON)

Every run produces a `manifest.json`:
```json
{
  "experiment_id": "eval_20260830_142301",
  "timestamp": "2026-08-30T14:23:01Z",
  "git_commit": "a1b2c3d4",
  "config": {
    "repos": [{"name": "fastapi", "commit": "abc123", "path": "eval_data/repos/fastapi"}],
    "questions_file": "eval_data/questions/fastapi_v1.jsonl",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_top_k": 10,
    "graph_depth": 2,
    "fusion_weights": {"vector": 0.7, "graph": 0.3}
  },
  "systems": ["vector_only", "hybrid"],
  "metrics_computed": ["precision@10", "recall@10", "mrr", "ndcg@10"],
  "environment": {
    "python": "3.11.9",
    "chromadb": "0.5.5",
    "sentence_transformers": "3.0.1",
    "networkx": "3.3"
  }
}
```

### 6.2 Raw Results (JSONL)

One line per (question, system):
```jsonl
{"question_id": "fastapi_001", "system": "hybrid", "top_k": 10, "depth": 2, "retrieved_ids": ["main.py::app", "routing.py::APIRouter"], "relevant_ids": ["main.py::app", "routing.py::APIRouter", "deps.py::get_db"], "precision_at_10": 0.2, "recall_at_10": 0.67, "mrr": 0.5, "latency_ms": 42}
{"question_id": "fastapi_001", "system": "vector_only", "top_k": 10, "depth": 0, "retrieved_ids": ["main.py::app", "routing.py::APIRouter", "middleware.py::CORSMiddleware"], "relevant_ids": ["main.py::app", "routing.py::APIRouter", "deps.py::get_db"], "precision_at_10": 0.2, "recall_at_10": 0.67, "mrr": 0.5, "latency_ms": 18}
```

### 6.3 Aggregated Summary (CSV)

```csv
system,repo,category,precision_at_10,recall_at_10,mrr,ndcg_at_10,latency_ms_mean
vector_only,fastapi,overall,0.32,0.58,0.41,0.38,18
hybrid,fastapi,overall,0.41,0.72,0.58,0.52,42
vector_only,fastapi,call_chain,0.18,0.35,0.22,0.25,18
hybrid,fastapi,call_chain,0.45,0.78,0.65,0.59,42
```

---

## 7. Results Output Formats

| Format | Purpose | Audience |
|--------|---------|----------|
| **JSONL (raw)** | Full per-query detail, reproducible analysis | Researchers, debugging |
| **CSV (summary)** | Per-system/repo/category aggregates | Comparison tables, plots |
| **Markdown Report** | Human-readable benchmark report | Documentation, README |
| **HTML (optional)** | Interactive tables/sortable | Presentation |

### 7.1 Markdown Report Template

```markdown
# RepoMind Retrieval Benchmark — {timestamp}

## Configuration
- Repositories: fastapi (abc123), click (def456)
- Embedding: all-MiniLM-L6-v2
- Vector top-k: 10 | Graph depth: 2 | Fusion: 0.7/0.3

## Overall Results

| System | Precision@10 | Recall@10 | MRR | nDCG@10 | Latency (ms) |
|--------|-------------|-----------|-----|---------|-------------|
| Vector-Only | 0.32 | 0.58 | 0.41 | 0.38 | 18 |
| Hybrid | **0.41** | **0.72** | **0.58** | **0.52** | 42 |

**Improvement**: Hybrid +28% precision, +24% recall, +41% MRR

## By Category

| Category | System | P@10 | R@10 | MRR |
|----------|--------|------|------|-----|
| call_chain | Vector-Only | 0.18 | 0.35 | 0.22 |
| call_chain | Hybrid | **0.45** | **0.78** | **0.65** |
| cross_file | Vector-Only | 0.25 | 0.48 | 0.33 |
| cross_file | Hybrid | **0.38** | **0.69** | **0.54** |

## Statistical Significance
- Paired Wilcoxon signed-rank test (n=150): p < 0.001 for all primary metrics
- Effect sizes (Cohen's d): Precision 0.62 (medium), Recall 0.71 (medium-large)

## Agent Evaluation (FreeLocalProvider)
- Task Success: Vector 92% / Hybrid 94%
- Mean Iterations: Vector 4.2 / Hybrid 3.8
- Graph Tool Usage (Hybrid): 67% of queries
```

---

## 8. Statistical Interpretation Guidelines

### 8.1 What to Report
- **Point estimates** with **95% confidence intervals** (bootstrap, 1000 resamples)
- **Paired test p-values** (Wilcoxon for non-normal, t-test if normal)
- **Effect sizes** (Cohen's d, Cliff's delta) — not just p-values
- **Per-category breakdown** — overall averages hide where hybrid helps/hurts

### 8.2 What NOT to Claim
- ❌ "Hybrid is better" — instead: "Hybrid shows statistically significant improvement on call_chain and cross_file categories (p<0.01)"
- ❌ "X% improvement" without CI — instead: "Precision@10 improved by 0.09 [95% CI: 0.05–0.13]"
- ❌ Generalization beyond tested repos — instead: "On 3 repos (fastapi, click, requests)..."
- ❌ Causality from correlation — graph helps *these* queries; not proven for all codebases

### 8.3 Minimum Reporting Standard
Every claim in the benchmark report must reference a specific table row or statistical test.

---

## 9. Project Structure

```
RepoMind/
├── repomind/
│   └── eval/
│       ├── __init__.py
│       ├── dataset.py          # Dataset loading, Question/GROUNDTRUTH dataclasses
│       ├── metrics.py          # Deterministic metrics (P@k, R@k, MRR, nDCG, citation P/R)
│       ├── judge.py            # Optional LLM judge (Gemini free-tier)
│       ├── runner.py           # Orchestration: run retrieval/agent on dataset
│       ├── reporter.py         # JSONL, CSV, Markdown output
│       └── statistical.py      # Bootstrap CI, paired tests, effect sizes
├── eval_data/
│   ├── repos/                  # Cloned/pinned repo copies (git submodules or tarballs)
│   │   ├── fastapi/            # Pinned commit
│   │   ├── click/
│   │   └── requests/
│   ├── questions/
│   │   ├── fastapi_v1.jsonl    # Questions + retrieval ground truth
│   │   ├── click_v1.jsonl
│   │   └── requests_v1.jsonl
│   ├── answers/
│   │   ├── fastapi_v1.jsonl    # Questions + reference answers + citations
│   │   ├── click_v1.jsonl
│   │   └── requests_v1.jsonl
│   └── fixtures/               # Small synthetic repos for unit tests
│       ├── tiny_repo/
│       └── medium_repo/
├── eval_outputs/
│   ├── runs/                   # One subdir per experiment run
│   │   └── eval_20260830_142301/
│   │       ├── manifest.json
│   │       ├── raw_results.jsonl
│   │       ├── summary.csv
│   │       └── report.md
│   └── latest -> runs/eval_20260830_142301/  # Symlink to latest
└── scripts/
    └── run_evaluation.py       # CLI entry: python -m scripts.run_evaluation
```

**Note**: `eval_data/repos/` not committed to git — fetched by script with pinned commit hashes. `eval_data/questions/` and `eval_data/answers/` **are** committed (small, human-curated).

---

## 10. Testing

### 10.1 Unit Tests (`tests/test_eval_*.py`)

| Test Module | Coverage |
|-------------|----------|
| `test_dataset.py` | Load JSONL, validate schema, split dev/test, filter by category |
| `test_metrics.py` | Precision@k, Recall@k, MRR, nDCG, citation P/R — known inputs → expected outputs |
| `test_statistical.py` | Bootstrap CI, Wilcoxon, Cohen's d — synthetic data with known properties |
| `test_reporter.py` | JSONL/CSV/Markdown round-trip, formatting |
| `test_judge.py` | Judge prompt formatting, JSON parsing, score bounds |

### 10.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_retrieval_eval_end_to_end` | Index fixture repo → run vector-only + hybrid retrieval on 5 questions → metrics match hand-computed |
| `test_agent_eval_structure` | Run agent (free-local) on fixture → verify state logging, tool calls recorded |
| `test_reproducibility` | Same manifest + same inputs → byte-identical raw_results.jsonl |

### 10.3 Test Fixtures

```
eval_data/fixtures/
├── tiny_repo/
│   ├── __init__.py
│   ├── a.py        # def foo(): pass; def bar(): foo()
│   └── b.py        # from a import bar; def baz(): bar()
├── tiny_questions.jsonl   # 5 questions with known answers
└── tiny_answers.jsonl     # Reference answers
```

All deterministic — no network, no API keys, runs in <5 seconds.

---

## 11. Proposed Implementation Phases

| Phase | Deliverable | Est. Effort | Depends On |
|-------|-------------|-------------|------------|
| **9.1** | Dataset infrastructure: `dataset.py`, JSONL schemas, fixture repos/questions | 2–3 days | — |
| **9.2** | Deterministic retrieval metrics: `metrics.py` (P@k, R@k, MRR, nDCG, citation P/R) | 2 days | 9.1 |
| **9.3** | Retrieval runner: `runner.py` — vector-only + hybrid on dataset, emit JSONL | 2 days | 9.2 |
| **9.4** | Statistical analysis: `statistical.py` — bootstrap CI, paired tests, effect sizes | 1–2 days | 9.3 |
| **9.5** | Reporting: `reporter.py` — CSV summary, Markdown report | 1 day | 9.4 |
| **9.6** | Answer evaluation (deterministic): citation P/R, ROUGE-L, key_fact coverage | 2 days | 9.1 |
| **9.7** | Optional LLM judge: `judge.py` — Gemini free-tier, strict rubric | 1–2 days | 9.6 |
| **9.8** | Agent evaluation: run ReAct agent, log state, compute structural metrics | 2 days | 9.3 |
| **9.9** | CLI entry point: `scripts/run_evaluation.py` with `--system`, `--repo`, `--output-dir` | 1 day | 9.5 |
| **9.10** | Unit/integration tests + CI integration | 2 days | All |
| **9.11** | Create initial evaluation datasets (3 repos, ~50 q each) | 3–5 days | 9.1 |
| **9.12** | Run first benchmark, produce report, document findings | 1–2 days | 9.9–9.11 |

**Total**: ~20–25 days (can be parallelized; core metrics + runner ≈ 1 week)

---

## 12. Risks & Weaknesses

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Ground truth is subjective** | Medium | High | Human verification of every chunk; multiple annotators for dev set; inter-annotator agreement |
| **Vector-only baseline is weak** | Low | Medium | Use strong embedding model (all-MiniLM-L6-v2), tune top-k fairly |
| **Hybrid config not optimal** | Medium | Medium | Grid search fusion weights (0.5/0.5, 0.7/0.3, 0.3/0.7) and depth (1/2/3) in dev |
| **LLM judge unreliable** | Medium | Low | Judge is optional; core claims rest on deterministic metrics |
| **Dataset too small for stats** | Medium | Medium | Target 150+ questions; bootstrap CI works down to n≈30 |
| **Reproducibility drift (chromadb, sentence-transformers versions)** | Low | High | Pin versions in manifest; record exact env; use `pip freeze` |
| **Graph construction non-deterministic** | Low | High | Graph built once per repo, persisted; same pickle used for all runs |
| **FreeLocalProvider too weak for agent eval** | High | Low | Agent eval with free-local only measures structure (tool calls, iterations); quality needs Gemini |

---

## 13. Summary: What Success Looks Like

A **complete Stage 9** delivers:

1. **Runnable benchmark**: `python -m scripts.run_evaluation --repo fastapi --system both --out eval_outputs/runs/`
2. **Reproducible artifacts**: manifest.json + raw_results.jsonl + summary.csv + report.md per run
3. **Statistically sound comparison**: Paired tests, CIs, effect sizes — no p-hacking
4. **Clear answer**: "Hybrid improves Recall@10 by X% [CI] on call-chain questions; no significant difference on symbol-lookup"
5. **Extensible framework**: New metrics, new repos, new systems plug in without rewrite

---

*Plan complete. Awaiting approval to begin Phase 9.1 implementation.*