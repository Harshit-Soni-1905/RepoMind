# RepoMind Retrieval Benchmark — fastapi_calls_final

## Configuration
- **Repositories**: ['C:\\MyWork\\Projects\\RepoMind\\eval_data\\repos\\fastapi_corpus']
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector top-k**: 10
- **Graph depth**: 2
- **Fusion weights**: vector=0.7, graph=0.3

## Overall Results

| System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_only | 0.167 | 0.262 | 0.043 | 0.274 | 0.263 | 0.035 | 0.307 | 0.280 | 0.018 | 0.307 | 0.280 | 12.7 |
| hybrid | 0.178 | 0.258 | 0.047 | 0.279 | 0.269 | 0.035 | 0.307 | 0.282 | 0.025 | 0.362 | 0.301 | 12.0 |

## By Category

| Category | System | mrr | r_precision | precision_at_5 | recall_at_5 | ndcg_at_5 | precision_at_10 | recall_at_10 | ndcg_at_10 | precision_at_20 | recall_at_20 | ndcg_at_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbol_lookup | combined | 0.279 | 0.233 | 0.067 | 0.300 | 0.268 | 0.040 | 0.333 | 0.281 | 0.023 | 0.383 | 0.295 |
| dependency_trace | combined | 0.105 | 0.062 | 0.036 | 0.045 | 0.032 | 0.073 | 0.149 | 0.090 | 0.041 | 0.163 | 0.096 |
| cross_file | combined | 0.200 | 0.094 | 0.060 | 0.106 | 0.105 | 0.033 | 0.117 | 0.111 | 0.025 | 0.167 | 0.130 |
| behavioral | combined | 0.182 | 0.182 | 0.036 | 0.182 | 0.182 | 0.018 | 0.182 | 0.182 | 0.009 | 0.182 | 0.182 |
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
- Mean difference (A - B): 0.0113
- 95% CI: [-0.0003, 0.0304]
- Wilcoxon p-value: 0.1842
- Paired t-test p-value: 0.1910
- Cohen's d: 0.169 (Cohen's d: negligible, Cliff's delta: negligible)

### ndcg_at_10
- Mean difference (A - B): 0.0023
- 95% CI: [-0.0015, 0.0086]
- Wilcoxon p-value: 0.8339
- Paired t-test p-value: 0.4414
- Cohen's d: 0.099 (Cohen's d: negligible, Cliff's delta: negligible)

### r_precision
- Mean difference (A - B): -0.0042
- 95% CI: [-0.0125, 0.0000]
- Wilcoxon p-value: 1.0000
- Paired t-test p-value: 0.3173
- Cohen's d: -0.129 (Cohen's d: negligible, Cliff's delta: negligible)


---
*Report generated: 2026-09-01T16:21:35.660163Z*
*Git commit: unknown*