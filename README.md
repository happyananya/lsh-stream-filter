# LSH Stream Filter

An experimental streaming novelty filter that uses SimHash-style Locality Sensitive Hashing (LSH) to reject near-duplicate embeddings in real time.

The project compares an approximate LSH admission filter against a brute-force cosine-similarity oracle, then evaluates precision/recall/compression across synthetic and real-text embedding streams.

## What This Project Does

In a high-volume embedding stream, storing every vector is expensive and often redundant.
This repository implements:

- `LSHAdmissionFilter`: fast approximate novelty filter using random hyperplane hashes.
- `BruteForceOracle`: exact ground truth novelty detector (`O(n)` per insert).
- Stream generators for:
  - synthetic clustered vectors
  - real MS MARCO passage embeddings (MiniLM)
- Evaluation utilities and experiment scripts to measure:
  - novelty precision
  - novelty recall
  - compression ratio
  - false admit / false reject rates

## Core Idea

Each incoming vector is L2-normalized and hashed into `L` hash tables, each table using `k` random hyperplanes (bit signature length `k`).

Admission decision:

1. Look up bucket-mates in every table.
2. For each candidate, compute true cosine similarity.
3. Reject if any candidate similarity is `>= similarity_threshold`.
4. Otherwise admit and insert into all tables.

This hybrid approach keeps LSH's speed benefits while reducing accidental false rejections from random bucket collisions.

## Repository Structure

- `lsh.py` - `LSHAdmissionFilter` implementation + smoke tests
- `oracle.py` - exact `BruteForceOracle` baseline + smoke tests
- `stream.py` - synthetic/real stream generation, embedding diagnostics
- `evaluate.py` - metrics + FIFO/random-eviction baselines
- `experiments/day3_synthetic.py` - synthetic integration run
- `experiments/day4_real_data.py` - real data run at multiple redundancy levels
- `experiments/day5_sweep.py` - parameter sweep + plots + CSV outputs
- `results/sweep_results.csv` - saved sweep table (sample run artifact)
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+ recommended
- macOS/Linux/WSL (tested in standard shell environments)
- Internet connection for first real-data run (downloads dataset/model)

Python packages used:

- `numpy`, `scipy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `datasets`
- `sentence-transformers`
- `tqdm`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1) Run module smoke tests

```bash
python lsh.py
python oracle.py
python stream.py
python evaluate.py
```

Each script includes assertions and prints success if internal checks pass.

### 2) Run synthetic experiment (fast)

```bash
python experiments/day3_synthetic.py
```

Expected behavior:

- Oracle stores exactly 1 per cluster (`10` total in this setup).
- LSH filter stores a small multiple of that (still strongly compressed).
- Printed table shows precision/recall/compression and confusion counts.

### 3) Run real-data experiment

```bash
python experiments/day4_real_data.py
```

On first run this will:

- stream MS MARCO passages
- encode with `sentence-transformers/all-MiniLM-L6-v2`
- cache embeddings to `embeddings.npy`

Later runs reuse the cache for speed.

### 4) Run full sweep + figures

```bash
python experiments/day5_sweep.py
```

Outputs saved under `results/`:

- `sweep_results.csv`
- `recall_heatmap.png`
- `precision_heatmap.png`
- `theory_vs_empirical.png`
- `memory_over_time.png`

## Metrics

The project evaluates filter decisions against oracle decisions per stream element.

- `TP`: filter admits, oracle admits (correct novel keep)
- `FP`: filter admits, oracle rejects (false admit of redundant item)
- `FN`: filter rejects, oracle admits (false reject of novel item)
- `TN`: filter rejects, oracle rejects (correct redundant reject)

Derived metrics:

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `compression_ratio = admitted_count / stream_length`

Interpretation:

- Higher precision -> fewer redundant vectors stored.
- Higher recall -> fewer novel vectors lost.
- Lower compression ratio -> less memory footprint.

## Main Parameters and Their Effects

- `k` (hash bits per table):
  - higher `k` -> stricter bucket matching, usually fewer accidental collisions
- `L` (number of tables):
  - higher `L` -> more chances to find true near-duplicates
- `similarity_threshold`:
  - higher threshold -> more tolerant, more admits
  - lower threshold -> stricter novelty filtering
- `redundancy` in `real_stream`:
  - controls duplicate fraction of generated stream

Rule of thumb:

- Start at `k=8`, `L=8`, `similarity_threshold=0.9`.
- Increase `k` if false admits are too high.
- Increase `L` if recall is low (missing novel/duplicate discrimination trade-off depends on stream geometry).

## Reproducibility

Most scripts use fixed seeds (`seed=42` by default), so runs are stable given:

- same dependency versions
- same model version/cache
- same hardware math behavior

If you need strict reproducibility across machines, pin package versions in `requirements.txt` and keep the same cached embeddings file.

## Example Existing Sweep Results

The checked-in `results/sweep_results.csv` already contains runs for:

- `k ∈ {4, 8, 16}`
- `L ∈ {4, 8, 16}`
- redundancy levels `{20%, 50%, 80%}`

Typical pattern in this artifact:

- `k=4` with moderate/high `L` gives near-perfect precision/recall.
- very high `k` (e.g., `16`) can increase false admits in some settings due to collision dynamics and stream structure.
- compression tracks redundancy: more redundant streams lead to lower optimal keep fraction.

## Troubleshooting

- Real-data script is slow on first run:
  - expected; model inference + dataset loading happen once.
- Out-of-memory concerns:
  - reduce `N_PASSAGES` in experiment scripts.
- No cache reuse:
  - confirm `embeddings.npy` path and write permissions.
- Plot generation fails in headless environments:
  - script already forces non-interactive backend (`matplotlib.use("Agg")`).

## Notes and Limitations

- `LSHAdmissionFilter` still verifies cosine on bucket candidates, so it is not purely hash-only.
- Real-stream "duplicates" are injected via noisy copies, which approximates practical near-duplicates but is still synthetic augmentation.
- The project is research/prototyping oriented rather than a packaged production library.

## Next Improvements (Optional)

- Add command-line interfaces for all experiments.
- Convert configs (`k`, `L`, threshold, dataset size) to argparse flags.
- Add unit tests under `tests/` and CI workflow.
- Add latency benchmarking (time per insert) and memory profiling.