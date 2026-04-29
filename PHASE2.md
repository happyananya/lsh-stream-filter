# Phase 2 Reproducibility Baselines

This phase reproduces SIFT1M ANN baselines and freezes the timing harness for later phases.

## Scope

- Reproduce HNSW (`hnswlib`) on SIFT1M
- Reproduce FALCONN cross-polytope LSH on SIFT1M
- Use one shared timing harness (`scripts/phase2/benchmark_harness.py`)

## Prerequisites

Ensure Phase 1 SIFT files exist:

- `data/sift1m/base.npy`
- `data/sift1m/query.npy`
- `data/sift1m/groundtruth.npy`

Install libraries:

```bash
pip install hnswlib falconn
```

## 1) Reproduce HNSW on SIFT1M

```bash
python scripts/phase2/reproduce_hnsw_sift1m.py \
  --data-dir data/sift1m \
  --k 10 \
  --M 16 \
  --ef-construction 200 \
  --ef-search 100
```

Output:

- `results/phase2/hnsw_sift1m.json`

Expected sanity range:

- Recall@10 around `0.94-0.97`
- QPS depends heavily on hardware; compare against your machine baseline.

## 2) Reproduce FALCONN on SIFT1M

```bash
python scripts/phase2/reproduce_falconn_sift1m.py \
  --data-dir data/sift1m \
  --k 10 \
  --num-hash-tables 20 \
  --num-hash-bits 18 \
  --num-probes 64
```

Output:

- `results/phase2/falconn_sift1m.json`

Tune `num_hash_tables`, `num_hash_bits`, and `num_probes` to match published points.

## Fallback if FALCONN fails to build

On some macOS/Python combinations, `falconn` may fail to compile. Use a temporary
classical LSH fallback with FAISS:

```bash
python scripts/phase2/reproduce_faiss_lsh_sift1m.py \
  --data-dir data/sift1m \
  --k 10 \
  --nbits 256
```

Output:

- `results/phase2/faiss_lsh_sift1m.json`

This keeps Phase 2 moving, but keep in mind it is a fallback baseline (weaker than FALCONN).

## 3) Summarize results

```bash
python scripts/phase2/summarize_phase2.py --results-dir results/phase2
```

## Harness policy (frozen for later phases)

- Warm-up queries before measurement
- Per-query timing with `time.perf_counter_ns`
- Single-threaded defaults via environment variables
- Recall computed against ground truth for each query

## Optional quick debug mode

Use `--max-queries 1000` on reproduce scripts for fast iteration before full runs.
