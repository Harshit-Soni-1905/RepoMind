# RepoMind Retrieval Benchmark — dryrun_fastapi_20260831

## Configuration
- **Repositories**: ['C:\\MyWork\\Projects\\RepoMind\\eval_data\\repos\\fastapi_corpus']
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector top-k**: 5
- **Graph depth**: 3
- **Fusion weights**: vector=0.7, graph=0.3

## Overall Results

| System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_only | 0.495 | 0.530 | 0.147 | 0.594 | 0.578 | 0.073 | 0.594 | 0.578 | 0.037 | 0.594 | 0.578 | 8.8 |

## By Category

| Category | System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbol_lookup | combined | 0.789 | 0.733 | 0.187 | 0.867 | 0.809 | 0.093 | 0.867 | 0.809 | 0.047 | 0.867 | 0.809 |
| dependency_trace | combined | 0.053 | 0.045 | 0.036 | 0.045 | 0.033 | 0.018 | 0.045 | 0.033 | 0.009 | 0.045 | 0.033 |
| cross_file | combined | 0.706 | 0.421 | 0.267 | 0.443 | 0.487 | 0.133 | 0.443 | 0.487 | 0.067 | 0.443 | 0.487 |
| behavioral | combined | 0.609 | 0.545 | 0.145 | 0.682 | 0.624 | 0.073 | 0.682 | 0.624 | 0.036 | 0.682 | 0.624 |
| negative | combined | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |


---
*Report generated: 2026-08-31T07:22:50.365174Z*
*Git commit: unknown*