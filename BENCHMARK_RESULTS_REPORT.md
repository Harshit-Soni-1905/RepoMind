# FastAPI Benchmark Results: Vector-Only vs Hybrid Retrieval

**Experiment ID:** `fastapi_v1`  
**Date:** 2026-08-31  
**Corpus:** FastAPI 0.141.1 (commit `95f8322ee1dcda7ceace7b1c4f6c9915b36d748f`)  
**Configuration:** `top_k=20`, `graph_depth=3`, fusion weights vector=0.7 / graph=0.3  
**Dataset:** 60 questions (15 symbol_lookup, 15 cross_file, 11 dependency_trace, 11 behavioral, 8 negative)

---

## 1. Overall Results

| Metric | Vector-Only | Hybrid | Absolute Δ | Relative Δ | Winner |
|--------|-------------|--------|------------|------------|--------|
| **MRR** | 0.518 | 0.438 | **-0.080** | **-15.4%** | Vector |
| **R-Precision** | 0.535 | 0.461 | **-0.074** | **-13.8%** | Vector |
| **Precision@5** | 0.147 | 0.130 | **-0.017** | **-11.4%** | Vector |
| **Recall@5** | 0.594 | 0.537 | **-0.057** | **-9.6%** | Vector |
| **nDCG@5** | 0.578 | 0.509 | **-0.069** | **-12.0%** | Vector |
| **Precision@10** | 0.093 | 0.082 | **-0.012** | **-12.5%** | Vector |
| **Recall@10** | 0.673 | 0.641 | **-0.032** | **-4.8%** | Vector |
| **nDCG@10** | 0.612 | 0.546 | **-0.066** | **-10.8%** | Vector |
| **Precision@20** | 0.053 | 0.053 | 0.000 | 0.0% | Tie |
| **Recall@20** | 0.730 | 0.730 | 0.000 | 0.0% | Tie |
| **nDCG@20** | 0.629 | 0.574 | **-0.055** | **-8.7%** | Vector |
| **Latency (ms)** | 9.1 | 12.8 | +3.7 | +40.6% | Vector |

**Summary:** **Vector-only outperforms hybrid on every ranking metric at every k.** The differences are consistent and non-trivial (9–15% relative degradation). Hybrid is also ~40% slower.

---

## 2. Per-Category Results

### Symbol Lookup (15 questions)

| Metric | Vector | Hybrid | Δ |
|--------|--------|--------|---|
| MRR | **0.802** | 0.624 | -0.178 |
| R-Prec | **0.733** | 0.567 | -0.167 |
| P@5 | **0.187** | 0.160 | -0.027 |
| R@5 | **0.867** | 0.667 | -0.200 |
| nDCG@5 | **0.809** | 0.610 | -0.199 |

**Vector wins decisively.** Hybrid's graph expansion pulls in related symbols (siblings, imports) that dilute the top-ranked exact match.

### Cross-File Reasoning (15 questions)

| Metric | Vector | Hybrid | Δ |
|--------|--------|--------|---|
| MRR | **0.717** | 0.626 | -0.091 |
| R-Prec | **0.421** | 0.378 | -0.043 |
| P@5 | **0.267** | 0.227 | -0.040 |
| R@5 | **0.443** | 0.400 | -0.043 |
| nDCG@5 | **0.487** | 0.428 | -0.059 |

**Vector wins.** These questions require multi-hop reasoning (e.g., handler → utils, dependant → path params), but the graph corroboration is not strong enough to overcome vector's correct first hits.

### Dependency Trace (11 questions)

| Metric | Vector | Hybrid | Δ |
|--------|--------|--------|---|
| MRR | **0.109** | 0.060 | -0.049 |
| R-Prec | **0.074** | 0.000 | -0.074 |
| P@5 | **0.036** | 0.018 | -0.018 |
| R@5 | **0.045** | 0.023 | -0.023 |
| nDCG@5 | **0.033** | 0.014 | -0.019 |

**Vector wins.** Ground truth targets are `module_context` chunks (import blocks) — which have **no graph nodes**. Graph expansion cannot reach them, so hybrid gains nothing and loses precision from false positives.

### Behavioral (11 questions)

| Metric | Vector | Hybrid | Δ |
|--------|--------|--------|---|
| MRR | **0.643** | 0.622 | -0.021 |
| R-Prec | **0.545** | 0.500 | -0.045 |
| P@5 | 0.145 | **0.164** | +0.018 |
| R@5 | 0.682 | **0.727** | +0.045 |
| nDCG@5 | **0.624** | 0.619 | -0.006 |

**Mixed.** Hybrid slightly edges vector on recall@5 but loses on MRR, R-Prec, and nDCG. The wins are on questions where the answer spans multiple symbols (e.g., OAuth2 scopes merged across dependencies).

### Negative (8 questions)

| Metric | Vector | Hybrid |
|--------|--------|--------|
| All metrics | Identical | Identical |

Both return empty results; metrics return constant values (P=0, R=1, nDCG=1) per implementation. Non-discriminative.

---

## 3. Per-Question Paired Analysis (MRR)

| Category | Hybrid Better | Vector Better | Tied |
|----------|---------------|---------------|------|
| symbol_lookup | 1 | 4 | 10 |
| cross_file | 3 | 3 | 9 |
| dependency_trace | 4 | 7 | 0 |
| behavioral | 5 | 4 | 2 |
| **Total** | **13** | **18** | **29** |

**Vector wins on more questions** (18 vs 13), but **29 questions are tied** (MRR identical, typically both 0 or both 1). The magnitude of vector's wins is larger: average MRR drop when hybrid loses = 0.37; average MRR gain when hybrid wins = 0.12.

---

## 4. Statistical Significance (Paired Tests, Hybrid – Vector)

| Metric | Mean Δ | 95% CI | Wilcoxon p | t-test p | Cohen's d | Cliff's δ | Significance |
|--------|--------|--------|------------|----------|-----------|-----------|--------------|
| precision@10 | -0.012 | [-0.025, 0.000] | 0.124 | 0.084 | -0.22 | negligible | NS |
| recall@10 | -0.032 | [-0.096, 0.027] | 0.192 | 0.273 | -0.15 | negligible | NS |
| MRR | -0.080 | [-0.152, -0.014] | **0.008** | **0.012** | **-0.39** | small | **Significant** |
| nDCG@10 | -0.066 | [-0.126, -0.014] | **0.011** | **0.017** | **-0.36** | small | **Significant** |
| R-Precision | -0.074 | [-0.138, -0.015] | **0.009** | **0.013** | **-0.37** | small | **Significant** |

**MRR, nDCG@10, and R-Precision show statistically significant degradation** for hybrid (p < 0.05, paired tests). Effect sizes are small (Cohen's d ≈ -0.35 to -0.39). Precision and recall differences at k=10 are not statistically significant, though directionally negative.

---

## 5. Why Hybrid Underperforms (Root Cause Analysis)

### 5.1 Ground-Truth Asymmetry (Documented Pre-Benchmark)

| Issue | Vector | Hybrid |
|-------|--------|--------|
| 53 chunks have no graph node (44 module_context, 6 class_header, 3 file) | ✅ Retrieved normally | ❌ Invisible to graph expansion |
| 186 graph nodes absent from vector store (48 FILE, 138 unchunked symbols) | ❌ Never retrieved | ✅ Can surface via traversal |

**Consequence:** 16 ground-truth references are `module_context` chunks (import blocks). Hybrid **cannot corroborate** these, while vector retrieves them directly. Meanwhile, hybrid surfaces FILE nodes and unchunked symbols that are **not valid ground truth**, depressing precision.

### 5.2 Fusion Formula Behavior

Current fusion (post-fix):  
`score = 0.7 × norm_semantic + 0.3 × gph_score`  
where `gph_score = 1 − 1/(1 + support)` and `support = Σ 1/(1+d)`.

- **support** saturates quickly: even a single 1-hop neighbor gives `support=0.5 → gph=0.33`
- Vector scores (cosine) typically 0.3–0.8 for relevant chunks
- Graph term is small relative to vector weight; it mainly re-ranks **within** the vector top-k
- When vector's top hit is correct, graph expansion adds neighbors that push it down

### 5.3 Category-Specific Mechanisms

| Category | Why Vector Wins |
|----------|-----------------|
| symbol_lookup | Exact name match is already top-1; graph adds siblings/imports that dilute |
| cross_file | Vector finds the target directly (semantic match); graph path length ≥3 gives weak support |
| dependency_trace | Ground truth = `module_context` (no graph node); hybrid cannot reach it |
| behavioral | Mixed; hybrid helps when answer spans multiple related symbols |

---

## 6. Confidence Intervals (Bootstrap, 95%)

Bootstrap resampling (10,000 iterations, paired):

| Metric | Vector CI | Hybrid CI | Δ CI |
|--------|-----------|-----------|------|
| MRR | [0.448, 0.588] | [0.362, 0.518] | [-0.152, -0.014] |
| R-Prec | [0.452, 0.618] | [0.365, 0.558] | [-0.138, -0.015] |
| nDCG@10 | [0.548, 0.676] | [0.472, 0.620] | [-0.126, -0.014] |
| Recall@10 | [0.598, 0.748] | [0.558, 0.722] | [-0.096, +0.027] |

Vector's intervals are consistently higher; hybrid's intervals are consistently lower. The Δ CIs for MRR, R-Prec, nDCG@10 exclude zero.

---

## 7. Effect Sizes

| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| MRR | -0.39 | Small, favors vector |
| R-Precision | -0.37 | Small, favors vector |
| nDCG@10 | -0.36 | Small, favors vector |
| Recall@10 | -0.15 | Negligible |
| Precision@10 | -0.22 | Small, favors vector |

All significant effects are **small but consistent** in favor of vector-only.

---

## 8. Latency

| System | Mean Latency (ms) | Std Dev |
|--------|-------------------|---------|
| Vector-only | 9.1 | ~2.1 |
| Hybrid | 12.8 | ~3.4 |

Graph traversal adds ~3.7 ms/query (40% overhead) with no accuracy benefit.

---

## 9. Artifacts Generated

| File | Description |
|------|-------------|
| `eval_outputs/fastapi_benchmark_20260831/fastapi_v1/raw_results.jsonl` | 120 records (60 questions × 2 systems) with full retrieved lists and per-question metrics |
| `eval_outputs/fastapi_benchmark_20260831/fastapi_v1/summary.csv` | 17 rows: overall + per-category per-system metrics |
| `eval_outputs/fastapi_benchmark_20260831/fastapi_v1/manifest.json` | Full experiment provenance (corpus commit, config, systems, metrics) |
| `eval_outputs/fastapi_benchmark_20260831/fastapi_v1/report.md` | Markdown report with tables and statistical analysis |

---

## 10. Tests Passed

```
================= 356 passed, 2 skipped, 4 warnings in 21.08s =================
TOTAL coverage: 93%
```

All Stage 1–9 tests pass after the benchmark run.

---

## 11. Conclusions

### Primary Finding

**Vector-only retrieval outperforms hybrid vector+graph retrieval on this FastAPI corpus and question set.** The difference is statistically significant for MRR (p=0.008), R-Precision (p=0.009), and nDCG@10 (p=0.011), with small effect sizes (d ≈ -0.35 to -0.39). Hybrid is also ~40% slower.

### Why This Is Not a Bug

The benchmark was designed to measure the *current* system honestly. The asymmetry between vector and graph coverage was **documented before the run** (16 ground-truth chunks have no graph node; 186 graph nodes have no vector representation). This is a property of the current graph construction (no CALLS edges, module_context chunks excluded), not an evaluation error.

### What Would Change the Result

| Change | Expected Effect |
|--------|-----------------|
| Add CALLS edges (function→function calls) | Would create shorter cross-file paths, stronger graph corroboration |
| Include module_context as graph nodes | Would allow hybrid to corroborate import-block answers |
| Increase graph weight (e.g., 0.5/0.5) | Would amplify graph signal but also false positives |
| Tune questions to multi-hop dependencies | Would favor hybrid by construction (violates methodology) |

### Honest Assessment

The current RepoMind graph model (DEFINES/CONTAINS/IMPORTS only) provides **insufficient structural signal** to improve over a strong semantic vector baseline on this corpus. The graph is sparse (88 IMPORTS edges for 48 files), cross-file distances are long (≥3 hops), and key answer types (import blocks) are graph-invisible.

**Recommendation:** Before investing in graph retrieval improvements, add CALLS edges and/or include module_context in the graph. Re-evaluate on the same dataset to measure the actual gain.

---

## 12. Constraints Honored

| Constraint | Status |
|------------|--------|
| Same corpus, questions, top-k, depth for both systems | ✅ |
| No dataset modification after seeing results | ✅ |
| No parameter tuning to favor hybrid | ✅ |
| Paired experimental design preserved | ✅ |
| k=5, 10, 20 all reported | ✅ |
| All required metrics collected | ✅ |
| Bootstrap CIs, paired tests, effect sizes | ✅ |
| Honest reporting (hybrid worse) | ✅ |
| No silent algorithm changes | ✅ |
| Tests re-run after benchmark | ✅ |
| Rich benchmark NOT started | ✅ |

---

**Experiment complete. Ready for review before any next steps.**