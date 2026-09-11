# RepoMind Retrieval Benchmark — fastapi_v1

## Configuration
- **Repositories**: ['C:\\MyWork\\Projects\\RepoMind\\eval_data\\repos\\fastapi_corpus']
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector top-k**: 20
- **Graph depth**: 3
- **Fusion weights**: vector=0.7, graph=0.3

## Overall Results

| System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_only | 0.518 | 0.535 | 0.147 | 0.594 | 0.578 | 0.093 | 0.673 | 0.612 | 0.053 | 0.730 | 0.629 | 9.1 |
| hybrid | 0.438 | 0.461 | 0.130 | 0.537 | 0.509 | 0.082 | 0.641 | 0.546 | 0.053 | 0.730 | 0.574 | 12.8 |

## By Category

| Category | System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbol_lookup | combined | 0.713 | 0.650 | 0.173 | 0.767 | 0.709 | 0.100 | 0.883 | 0.749 | 0.057 | 1.000 | 0.778 |
| dependency_trace | combined | 0.085 | 0.037 | 0.027 | 0.034 | 0.023 | 0.045 | 0.094 | 0.057 | 0.050 | 0.202 | 0.103 |
| cross_file | combined | 0.671 | 0.399 | 0.247 | 0.422 | 0.457 | 0.143 | 0.476 | 0.485 | 0.077 | 0.504 | 0.496 |
| behavioral | combined | 0.633 | 0.523 | 0.155 | 0.705 | 0.622 | 0.100 | 0.909 | 0.692 | 0.055 | 1.000 | 0.714 |
| negative | combined | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |

## Statistical Significance

### precision_at_10
- Mean difference (A - B): -0.0117
- 95% CI: [-0.0250, 0.0000]
- Wilcoxon p-value: 0.1235
- Paired t-test p-value: 0.0844
- Cohen's d: -0.223 (Cohen's d: small, Cliff's delta: negligible)

### recall_at_10
- Mean difference (A - B): -0.0321
- 95% CI: [-0.0958, 0.0244]
- Wilcoxon p-value: 0.3139
- Paired t-test p-value: 0.3189
- Cohen's d: -0.129 (Cohen's d: negligible, Cliff's delta: negligible)

### mrr
- Mean difference (A - B): -0.0799
- 95% CI: [-0.1593, -0.0085]
- Wilcoxon p-value: 0.0572
- Paired t-test p-value: 0.0338
- Cohen's d: -0.274 (Cohen's d: small, Cliff's delta: negligible)

### ndcg_at_10
- Mean difference (A - B): -0.0664
- 95% CI: [-0.1332, -0.0091]
- Wilcoxon p-value: 0.0386
- Paired t-test p-value: 0.0392
- Cohen's d: -0.266 (Cohen's d: small, Cliff's delta: negligible)

### r_precision
- Mean difference (A - B): -0.0743
- 95% CI: [-0.1469, -0.0093]
- Wilcoxon p-value: 0.0653
- Paired t-test p-value: 0.0385
- Cohen's d: -0.267 (Cohen's d: small, Cliff's delta: negligible)


---
*Report generated: 2026-08-31T07:29:15.099624Z*
*Git commit: unknown*