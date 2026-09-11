# RepoMind Retrieval Benchmark — fastapi_improved_hybrid_20260901

## Configuration
- **Repositories**: ['C:\\MyWork\\Projects\\RepoMind\\eval_data\\repos\\fastapi_corpus']
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector top-k**: 10
- **Graph depth**: 2
- **Fusion weights**: vector=0.7, graph=0.3

## Overall Results

| System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_only | 0.516 | 0.535 | 0.147 | 0.594 | 0.578 | 0.093 | 0.673 | 0.612 | 0.047 | 0.673 | 0.612 | 14.1 |
| hybrid | 0.498 | 0.513 | 0.153 | 0.604 | 0.574 | 0.093 | 0.673 | 0.602 | 0.050 | 0.691 | 0.609 | 15.5 |

## By Category

| Category | System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbol_lookup | combined | 0.781 | 0.700 | 0.187 | 0.867 | 0.796 | 0.100 | 0.900 | 0.809 | 0.050 | 0.900 | 0.809 |
| dependency_trace | combined | 0.105 | 0.062 | 0.036 | 0.045 | 0.032 | 0.073 | 0.149 | 0.090 | 0.039 | 0.156 | 0.093 |
| cross_file | combined | 0.702 | 0.418 | 0.280 | 0.463 | 0.492 | 0.147 | 0.482 | 0.503 | 0.078 | 0.513 | 0.514 |
| behavioral | combined | 0.636 | 0.545 | 0.145 | 0.682 | 0.624 | 0.100 | 0.909 | 0.706 | 0.050 | 0.909 | 0.706 |
| negative | combined | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |

## Statistical Significance

### precision_at_10
- Mean difference (A - B): 0.0000
- 95% CI: [0.0000, 0.0000]
- Wilcoxon p-value: 1.0000
- Paired t-test p-value: 1.0000
- Cohen's d: 0.000 (Cohen's d: negligible, Cliff's delta: negligible)

### recall_at_10
- Mean difference (A - B): 0.0000
- 95% CI: [0.0000, 0.0000]
- Wilcoxon p-value: 1.0000
- Paired t-test p-value: 1.0000
- Cohen's d: 0.000 (Cohen's d: negligible, Cliff's delta: negligible)

### mrr
- Mean difference (A - B): -0.0179
- 95% CI: [-0.0433, -0.0001]
- Wilcoxon p-value: 0.1263
- Paired t-test p-value: 0.1260
- Cohen's d: -0.198 (Cohen's d: negligible, Cliff's delta: negligible)

### ndcg_at_10
- Mean difference (A - B): -0.0098
- 95% CI: [-0.0236, 0.0008]
- Wilcoxon p-value: 0.2213
- Paired t-test p-value: 0.1651
- Cohen's d: -0.179 (Cohen's d: negligible, Cliff's delta: negligible)

### r_precision
- Mean difference (A - B): -0.0222
- 95% CI: [-0.0597, 0.0083]
- Wilcoxon p-value: 0.2733
- Paired t-test p-value: 0.2287
- Cohen's d: -0.155 (Cohen's d: negligible, Cliff's delta: negligible)


---
*Report generated: 2026-08-31T18:57:38.018073Z*
*Git commit: unknown*