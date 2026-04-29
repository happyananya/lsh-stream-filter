# SIFT1M Phase 2 Report

## Outputs
- Results CSV: `results/phase2/sift1m_phase2_results.csv`
- Plot: `results/phase2/sift1m_phase2_recall_qps.png`

## Headline
- Best recall method: `hnswlib` (0.9828)
- Best QPS method: `hnswlib` (2772.94)

## Table
| method                   | dataset   |   recall |      qps |   p50_us |   p95_us |   p99_us |   mean_us |   n_queries | json_path                            |
|:-------------------------|:----------|---------:|---------:|---------:|---------:|---------:|----------:|------------:|:-------------------------------------|
| hnswlib                  | sift1m    |  0.98276 | 2772.94  |  358.833 |  500.252 |  662.799 |   360.629 |       10000 | results/phase2/hnsw_sift1m.json      |
| faiss_index_lsh_fallback | sift1m    |  0.23861 |  448.528 | 2204.46  | 2307.75  | 2413.29  |  2229.52  |       10000 | results/phase2/faiss_lsh_sift1m.json |
