# FastAPI Real-Benchmark Preparation Report

**Date:** 2026-08-31  
**Status:** Dataset validated, indexing verified, all tests passing. **Awaiting approval to run the actual vector-vs-hybrid benchmark comparison.**

---

## 1. Exact FastAPI Commit Used

| Field | Value |
|-------|-------|
| **Upstream repo** | https://github.com/tiangolo/fastapi.git |
| **Upstream tag** | 0.141.1 |
| **Upstream commit** | `95f8322ee1dcda7ceace7b1c4f6c9915b36d748f` |
| **Commit date** | 2026-07-29T17:15:38+00:00 |
| **Commit subject** | Release version 0.141.1 (#16106) |
| **License** | MIT |

Recorded in `eval_data/repos/fastapi_corpus/PROVENANCE.json` and injected into the experiment manifest via `config.corpus_provenance` in `scripts/run_evaluation.py`.

---

## 2. Actual Repository Statistics (Measured, Not Estimated)

### 2.1 Source Files

| Scope | .py files |
|-------|-----------|
| Full cloned repo (including tests, docs) | 1136 |
| `fastapi/` package only | **48** |
| `tests/` (excluded from corpus) | 593 |
| `docs_src/` (excluded) | ~500 |

**Corpus used:** The `fastapi/` package only (48 files). Scoping is applied identically to both vector_only and hybrid systems, so it cannot bias the comparison.

### 2.2 Indexing Results (from `build_repo_index` with `depth=3`)

| Metric | Value |
|--------|-------|
| Python files scanned | **48** |
| Files successfully parsed | **48** |
| Syntax errors | **0** |
| **AST chunks extracted** | **385** |
| Vector store chunks (after `clear()` + `add_chunks`) | **385** |
| Graph nodes | **518** |
| Graph edges | **558** |

### 2.3 Chunk Type Breakdown

| Chunk type | Count |
|------------|-------|
| function | 150 |
| class | 109 |
| method | 73 |
| module_context | 44 |
| class_header | 6 |
| file | 3 |
| **Total** | **385** |

### 2.4 Graph Structure

| Node type | Count |
|-----------|-------|
| METHOD | 205 |
| FUNCTION | 150 |
| CLASS | 115 |
| FILE | 48 |
| **Total nodes** | **518** |

| Edge type | Count |
|-----------|-------|
| DEFINES | 265 |
| CONTAINS | 205 |
| IMPORTS | 88 |
| **Total edges** | **558** |

### 2.5 Graph Depth Characteristics (undirected projection, matching HybridRetriever traversal)

| Metric | Value |
|--------|-------|
| Weakly connected components | 15 |
| Largest WCC size | 502 nodes |
| Isolated nodes | 13 |
| Diameter | 8 |
| Average shortest path | 4.39 |

### 2.6 Intra-Repo Import Subgraph (FILE nodes + IMPORTS edges only)

| Metric | Value |
|--------|-------|
| File nodes | 48 |
| IMPORTS edges | 88 |
| Components | 15 |
| Largest component | 33 files |
| Diameter | 5 |
| Average shortest path | 2.305 |

### 2.7 Cross-File Symbol Distance Distribution (deterministic 400-pair sample, cutoff 8)

| Distance (hops) | Pairs |
|-----------------|-------|
| 3 | 14,632 |
| 4 | 55,525 |
| 5 | 64,100 |
| 6 | 24,291 |
| 7 | 3,995 |
| 8 | 40 |
| **< 3** | **0** |

**Key finding:** No cross-file symbol pair is closer than **3 hops** in the undirected graph. `graph_depth >= 3` is therefore **necessary** for any cross-file symbol corroboration. This is an empirical measurement, not an assertion.

### 2.8 Top Files by Total Degree (all edge types)

| File | Degree |
|------|--------|
| routing.py | 53 |
| openapi/models.py | 51 |
| dependencies/utils.py | 40 |
| openapi/utils.py | 27 |
| _compat/v2.py | 24 |
| exceptions.py | 22 |
| dependencies/models.py | 22 |
| utils.py | 16 |
| _compat/shared.py | 16 |
| params.py | 15 |

### 2.9 Chunks per File (top 12)

| File | Chunks |
|------|--------|
| routing.py | 93 |
| openapi/models.py | 42 |
| dependencies/utils.py | 29 |
| applications.py | 25 |
| _compat/v2.py | 22 |
| dependencies/models.py | 17 |
| _compat/shared.py | 16 |
| openapi/utils.py | 13 |
| exceptions.py | 12 |
| params.py | 12 |
| param_functions.py | 10 |
| security/oauth2.py | 8 |

---

## 3. Evaluation Dataset Statistics

### 3.1 Dataset Files

- **Questions:** `eval_data/questions/fastapi_corpus_v1.jsonl` (60 questions)
- **Answers:** `eval_data/answers/fastapi_corpus_v1.jsonl` (60 answers)

### 3.2 Category Mix (proportional redistribution after removing call_chain)

| Category | Count | Share | Target (100/80 scaling) |
|----------|-------|-------|------------------------|
| symbol_lookup | 15 | 25.00% | 25.00% |
| cross_file | 15 | 25.00% | 25.00% |
| dependency_trace | 11 | 18.33% | 18.75% |
| behavioral | 11 | 18.33% | 18.75% |
| negative | 8 | 13.33% | 12.50% |
| **TOTAL** | **60** | **100%** | **100%** |

**Rationale:** Proportional scaling of the five remaining categories (×1.25) was chosen deliberately. Reallocating call_chain's 20% toward cross_file or dependency_trace would favor the hybrid system by construction, violating the constraint *"Do not change the benchmark methodology to make hybrid look better."*

### 3.3 Difficulty Mix

| Difficulty | Count |
|------------|-------|
| easy | 11 |
| medium | 30 |
| hard | 19 |

### 3.4 Ground-Truth Span

| Metric | Value |
|--------|-------|
| Answerable questions (non-negative) | 52 |
| Multi-chunk ground truth | 29 |
| Multi-file (cross-file) ground truth | 25 |
| Chunks per question (min/mean/max) | 1 / 2.33 / 7 |
| Total ground-truth chunk references | 121 |
| Distinct chunks referenced | 78 |

### 3.5 Graph Visibility of Ground Truth (Diagnostic)

| Metric | Count |
|--------|-------|
| Refs that ARE graph nodes | 105 |
| Refs with NO graph node | 16 |
| &nbsp;&nbsp;→ module_context | 16 |

**Implication:** 16 ground-truth references are `module_context` chunks, which have **no graph node**. They can never receive graph corroboration. This asymmetry is reported transparently and NOT engineered around.

---

## 4. Graph Depth Configuration

**Configured depth:** `3` (for both indexing and retrieval)

**Justification:**
- Measured minimum cross-file symbol distance = 3 hops
- Depth 2 reaches **zero** cross-file symbol pairs (empirically verified)
- Depth 3 is the minimal viable setting for graph corroboration on multi-hop questions
- Both systems use the same index; only the hybrid retriever traverses the graph

---

## 5. Issues Discovered (Documented, Not Fixed)

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | 53 vector chunks (`module_context` ×44, `class_header` ×6, `file` ×3) have **no graph node** | Graph corroboration cannot reach them; hybrid recall depressed for import-related questions | Documented, NOT fixed |
| 2 | 186 graph nodes (48 FILE nodes + 138 unchunked symbols) are **not in vector store** | Hybrid may surface nodes that vector store never sees; hybrid precision@k artificially lowered when file nodes rank | Documented, NOT fixed |
| 3 | Negative questions: metrics return constants (precision 0, recall 1, nDCG 1) for empty ground truth | Non-discriminative ties for negative category | Preserved per methodology |
| 4 | `AgentRunner` has API mismatches vs real agent (`create_default_registry` signature, `llm_provider` vs `provider`, `analyze` vs `run`) | Affects only `--agent` path; retrieval benchmark unaffected | Documented, out of scope |
| 5 | Two negative candidates invalidated: `templating.py` exists, CSRF appears in docstrings | Reduced negative pool; replaced with `graphql`, `jwt`, `bcrypt`, `prometheus` | Updated |

All issues are reported honestly. **No changes to methodology were made to hide or mitigate them.**

---

## 6. Tests Passed

```
============================= test session starts ==============================
...
TOTAL                                2233    146    93%
Coverage HTML written to dir htmlcov
================= 356 passed, 2 skipped, 4 warnings in 33.99s =================
```

All Stage 1–9 tests pass with **93% coverage**. No tests were weakened. The two skipped tests are pre-existing.

---

## 7. Reproducibility Artifacts

| Artifact | Location |
|----------|----------|
| Corpus provenance (upstream commit, tag, scope rationale) | `eval_data/repos/fastapi_corpus/PROVENANCE.json` |
| Corpus statistics (chunk inventory, import edges, depth metrics) | `eval_data/corpus_stats/fastapi/*.json` |
| Evaluation questions | `eval_data/questions/fastapi_corpus_v1.jsonl` |
| Evaluation answers (ground truth + citations) | `eval_data/answers/fastapi_corpus_v1.jsonl` |
| Validation script | `scripts/validate_dataset.py` |
| Index build script (same code path as benchmark) | `scripts/build_index.py` |
| Benchmark driver (updated to record corpus provenance) | `scripts/run_evaluation.py` |

---

## 8. Next Step (Requires Explicit Approval)

**Run the actual vector-vs-hybrid benchmark comparison:**

```bash
python -m scripts.run_evaluation \
    --repo eval_data/repos/fastapi_corpus \
    --system both \
    --questions eval_data/questions/fastapi_corpus_v1.jsonl \
    --answers eval_data/answers/fastapi_corpus_v1.jsonl \
    --out eval_outputs/fastapi_benchmark_20260831 \
    --top-k 10 \
    --depth 3 \
    --experiment-id fastapi_v1
```

This will produce:
- `raw_results.jsonl` — per-question, per-system retrieval records
- `summary.csv` — flattened metrics for analysis
- `manifest.json` — full experiment provenance (including upstream commit)
- `report.md` — markdown report with statistical comparison (Wilcoxon, Cohen's d)

**Awaiting your approval before executing.**

---

## 9. Constraints Honored

| Constraint | Status |
|------------|--------|
| Do not change methodology to make hybrid look better | ✅ Proportional redistribution |
| Do not tune questions to produce desired result | ✅ Questions authored from source only |
| Do not fabricate benchmark results | ✅ All numbers measured and reported |
| Preserve reproducibility | ✅ Provenance recorded at manifest level |
| Keep experiment zero-cost | ✅ Local embeddings, no paid APIs |
| Do not proceed to Rich repository yet | ✅ Not cloned |
| Do not create Stage 10 / redesign harness | ✅ No new architecture |
| Do not make arbitrary changes to Stages 1–9 | ✅ Only driver fix for provenance |

---

**Prepared for approval to proceed with the benchmark comparison.**