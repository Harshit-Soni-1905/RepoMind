# RepoMind Retrieval Benchmark — eval_20260831_052902_fe12fa

## Configuration
- **Repositories**: ['C:\\MyWork\\Projects\\RepoMind\\eval_data\\fixtures\\tiny_repo']
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector top-k**: 10
- **Graph depth**: 2
- **Fusion weights**: vector=0.7, graph=0.3

## Overall Results

| System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_only | 0.700 | 0.800 | 0.240 | 1.000 | 0.926 | 0.120 | 1.000 | 0.926 | 0.060 | 1.000 | 0.926 | 36.4 |
| hybrid | 0.700 | 0.800 | 0.240 | 1.000 | 0.926 | 0.120 | 1.000 | 0.926 | 0.060 | 1.000 | 0.926 | 14.8 |

## By Category

| Category | System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbol_lookup | combined | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 0.100 | 1.000 | 1.000 | 0.050 | 1.000 | 1.000 |
| call_chain | combined | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 0.100 | 1.000 | 1.000 | 0.050 | 1.000 | 1.000 |
| dependency_trace | combined | 0.500 | 0.000 | 0.200 | 1.000 | 0.631 | 0.100 | 1.000 | 0.631 | 0.050 | 1.000 | 0.631 |
| cross_file | combined | 1.000 | 1.000 | 0.600 | 1.000 | 1.000 | 0.300 | 1.000 | 1.000 | 0.150 | 1.000 | 1.000 |
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
- Mean difference (A - B): 0.0000
- 95% CI: [0.0000, 0.0000]
- Wilcoxon p-value: 1.0000
- Paired t-test p-value: 1.0000
- Cohen's d: 0.000 (Cohen's d: negligible, Cliff's delta: negligible)

### ndcg_at_10
- Mean difference (A - B): 0.0000
- 95% CI: [0.0000, 0.0000]
- Wilcoxon p-value: 1.0000
- Paired t-test p-value: 1.0000
- Cohen's d: 0.000 (Cohen's d: negligible, Cliff's delta: negligible)

### r_precision
- Mean difference (A - B): 0.0000
- 95% CI: [0.0000, 0.0000]
- Wilcoxon p-value: 1.0000
- Paired t-test p-value: 1.0000
- Cohen's d: 0.000 (Cohen's d: negligible, Cliff's delta: negligible)


---
*Report generated: 2026-08-31T05:30:12.523448Z*
*Git commit: unknown*