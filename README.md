# LSH Stream Filter

A research implementation of **Locality-Sensitive Hashing (LSH)-based stream filtering** for high-dimensional embedding streams. The core contribution is a **Bucket-Occupancy Retention Policy** that uses LSH bucket occupancy as an intrinsic novelty signal to decide which items to keep under a bounded memory budget — discarding semantically redundant items while preserving diverse, query-relevant ones.

## Motivation

Large-scale systems that ingest continuous streams of dense embeddings (e.g., conversational AI memory, document ingestion pipelines, real-time search corpora) must operate under strict memory constraints. Naive approaches like FIFO or uniform reservoir sampling waste budget on redundant, near-duplicate items.

This project evaluates a family of **streaming retention policies** that aim to answer:

> *Given a stream of N embeddings and a memory budget of B items, which B items should you keep to maximize recall on future queries?*

The key insight: LSH bucket occupancy is a cheap, unsupervised proxy for semantic redundancy. Items that hash to already-crowded buckets are likely near-duplicates; items in sparse buckets are novel and worth keeping.

---

## Project Structure

```
lsh-stream-filter/
├── src/
│   ├── retention_policies.py   # Core BucketOccupancyRetention + abstract base class
│   ├── baselines.py            # FIFO, Reservoir, RandomSampling, SemanticDedup, StreamLSH
│   ├── evaluation.py           # Streaming harness + recall@k evaluation
│   └── jaccard_tests.py        # BucketOccupancy with Jaccard-similarity aggregator
├── added_metrics.py            # Comprehensive metrics: recall, diversity, memory, latency
├── experiments/
│   ├── phase2_characterization.py   # LSH bucket distribution analysis
│   ├── phase2_falconnpp.py          # FALCONN++ baseline
│   ├── phase3_sift1m.py             # Retention policy sweep on SIFT1M
│   ├── phase3_msmarco.py            # Retention policy sweep on MS MARCO
│   ├── phase3_profile_falconn.py    # FALCONN profiling
│   ├── phase4_lsh.py                # Cross-polytope hashing + multi-probe + reranking
│   ├── phase4_sift1m.py             # Phase 4 sweep on SIFT1M
│   ├── phase4_msmarco.py            # Phase 4 sweep on MS MARCO
│   ├── phase4_2_sift1m.py           # Phase 4.2 sweep on SIFT1M
│   ├── phase4_2_msmarco.py          # Phase 4.2 sweep on MS MARCO
│   ├── phase4_bounded_memory.py     # Memory-bounded retention experiments
│   ├── phase5_locomo_qa.py          # End-to-end LLM QA benchmark (LoCoMo)
│   ├── phase6_jl_projection.py      # JL projection to compound memory savings
│   ├── benchmark_harness.py         # Shared QPS/recall benchmark utilities
│   ├── layered_lsh.py               # Layered LSH experiments
│   ├── lsh_hnsw_hybrid.py           # LSH–HNSW hybrid index
│   ├── plot_*.py                    # Result visualization scripts
│   └── run_*.py                     # Convenience runners
├── scripts/
│   ├── phase1/
│   │   ├── prepare_sift1m.py        # Convert SIFT1M HDF5 → .npy
│   │   ├── prepare_msmarco.py       # Download + embed MS MARCO passages
│   │   └── compute_groundtruth.py   # Exact brute-force ground truth
│   ├── phase2/
│   │   ├── benchmark_harness.py     # Core benchmark utilities
│   │   ├── reproduce_*.py           # Reproduce FAISS/FALCONN/HNSW results
│   │   └── run_phase2_sift1m.py     # Phase 2 runner
│   ├── prepare_sift.py              # Alternate SIFT prep script
│   ├── prepare_msmarco.py           # Alternate MS MARCO prep script
│   ├── prepare_streams.py           # Synthetic stream generation (with duplication/drift)
│   ├── compute_groundtruth_gpu.py   # GPU-accelerated ground truth computation
│   ├── analyze_results.py           # Result analysis and summary tables
│   └── build_falconnpp.sh           # Build FALCONN++ from source
├── tests/
│   └── test_memory_bounds.py        # Unit tests: all policies respect capacity B
├── data/
│   ├── sift1m/                      # SIFT1M vectors (base.npy, query.npy, groundtruth.npy)
│   └── msmarco/                     # MS MARCO embeddings + passage/query IDs
├── results/                         # Experiment outputs (CSV, PNG)
├── requirements.txt
└── .gitignore
```

---

## Core Algorithm: Bucket-Occupancy Retention

`BucketOccupancyRetention` (`src/retention_policies.py`) is the main proposed policy.

**How it works:**

1. For each incoming embedding `x`, compute its bucket IDs across `L` independent LSH hash tables using random-projection hashing (SimHash).
2. Look up how many items already occupy those buckets — this is the **occupancy score** `O(x)`, aggregated across tables using `min`, `max`, `mean`, or `median`.
3. Decide whether to keep `x`:
   - **Threshold mode** (`threshold=T`): keep if `O(x) < T`.
   - **Capacity-bounded mode** (default): if under budget `B`, always keep; if at capacity, keep only if `x` is more novel than the most-redundant currently retained item (max-heap eviction).
4. On insertion, increment bucket counts; on eviction, decrement them.

**Jaccard aggregator variant** (`src/jaccard_tests.py`): Instead of raw occupancy counts, compute the maximum Jaccard similarity between the new item's L-element bucket signature and all retained items that share at least one bucket. Uses an inverted index for `O(L × avg_bucket_size)` lookup cost instead of `O(M)`.

### Retention Policy Interface

All policies implement the `RetentionPolicy` abstract base class:

```python
class RetentionPolicy(ABC):
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        """Process one streaming item. Returns True if kept."""

    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (embeddings, item_ids) for all retained items."""

    def kept_count(self) -> int:
        """Number of currently retained items."""

    def stats(self) -> dict:
        """Diagnostics (kept, seen, discarded, bucket stats, ...)."""
```

### Baselines (`src/baselines.py`)

| Policy | Description |
|--------|-------------|
| `OracleRetention` | Keep everything — recall upper bound |
| `FIFORetention` | Keep the most recent `B` items |
| `ReservoirSamplingRetention` | Vitter's Algorithm R — uniform random sample of size `B` |
| `RandomSamplingRetention` | Keep each item with fixed probability `B/N` |
| `SemanticDedupRetention` | Keep items with cosine similarity < `1 - ε` to all retained items (FAISS exact search) |
| `StreamLSHRetention` | Freshness-based retention (Kraus et al., IEEE BigData 2017); equivalent to FIFO under ingestion-only conditions |

---

## Experimental Phases

### Phase 1 — Data Preparation

Converts raw datasets into the `.npy` format used throughout.

```bash
# SIFT1M (download sift-128-euclidean.hdf5 from ann-benchmarks first)
python scripts/phase1/prepare_sift1m.py --hdf5-path sift-128-euclidean.hdf5

# MS MARCO (downloads + embeds with sentence-transformers)
python scripts/phase1/prepare_msmarco.py

# Compute exact brute-force ground truth
python scripts/phase1/compute_groundtruth.py
```

Produces:
- `data/sift1m/`: `base.npy` (1M × 128 float32), `query.npy` (10K × 128), `groundtruth.npy`
- `data/msmarco/`: `base.npy`, `query.npy`, `groundtruth.npy`, `passage_ids.npy`, `query_ids.npy`

### Phase 2 — ANN Baseline Benchmarks

Reproduces standard ANN benchmarks to establish comparison baselines.

```bash
python scripts/phase2/run_phase2_sift1m.py
```

Covers: FAISS flat LSH, FALCONN (cross-polytope), HNSW (hnswlib). Outputs recall@10 vs QPS Pareto curves to `results/`.

### Phase 3 — Retention Policy Sweep

Streams each dataset through every retention policy across a range of memory budgets `B`.

```bash
python experiments/phase3_sift1m.py
python experiments/phase3_msmarco.py
```

Computes recall@10 of the retained set as a function of budget, then plots Pareto frontiers.

### Phase 4 — LSH Optimizations

Implements a layered sequence of optimizations to the LSH index:

| Step | Technique | Benefit |
|------|-----------|---------|
| 4.1 | Cross-polytope hashing (Andoni et al., 2015) | `O(d log d)` rotation vs `O(d²)` dense projection |
| 4.2 | Multi-probe LSH (`T` perturbed buckets per table) | Same recall with ~10× fewer tables |
| 4.3 | PCA preprocessing *(stub)* | Reduce `d` before hashing |
| 4.4 | ITQ rotation *(stub)* | Improved binary code quality |
| 4.5 | Exact-distance reranking | +0.05–0.20 recall lift at fixed budget |
| 4.6 | Bit-packing + SIMD popcount *(stub)* | 5–10× QPS improvement |

```bash
python experiments/phase4_lsh.py --dataset sift1m --step all
python experiments/phase4_lsh.py --dataset msmarco --step 4.5
```

### Phase 5 — End-to-End LLM Memory Benchmark (LoCoMo)

Tests whether retention policies preserve enough conversational context for an LLM to answer questions about earlier sessions.

**Pipeline:**
1. Load LoCoMo multi-session conversations
2. Embed each dialogue turn with `all-MiniLM-L6-v2`
3. Stream turns through each policy under bounded memory
4. For each QA question, retrieve top-K context from the retained set
5. Prompt an LLM (Qwen 2.5 via Ollama) to answer; score against ground truth
6. Compare answer accuracy across policies at different budgets

```bash
python experiments/phase5_locomo_qa.py
```

Requires: Ollama running locally with `qwen2.5:7b`, LoCoMo dataset in `data/locomo/`.

### Phase 6 — Johnson-Lindenstrauss Projection

Tests whether JL projection can compound the memory advantage of `BucketOccupancyRetention` — store more items for the same byte footprint by projecting embeddings to lower dimensionality, at a controlled recall cost.

```bash
python experiments/phase6_jl_projection.py
```

---

## Metrics

`added_metrics.py` provides a multi-tier evaluation suite callable after any experiment:

**Tier 1 — Recall & Latency**
- `recall_at_k_distribution` — per-query recall@k with full distribution (mean, p10/p50/p90, zero-recall fraction)
- `recall_at_multiple_k` — recall@1, @10, @100 in one pass
- `query_relevant_coverage` — fraction of oracle top-k union covered by the kept set
- `latency_distribution` — insertion latency (mean, p50/p95/p99/p999, tail ratio)

**Tier 2 — Diversity (L2-based)**
- `k_center_radius` — worst-case distance from any stream item to its nearest retained representative
- `cluster_coverage` — fraction of stream clusters with at least one representative kept
- `mean_intra_set_distance` — mean nearest-neighbor distance within the kept set

**Tier 2b — Diversity (cosine-based)**
- `mean_pairwise_cosine_similarity` — nearest-neighbor cosine sim distribution in the kept set
- `cosine_coverage_radius` — worst-case angular gap (1 − min cosine sim)
- `cosine_redundancy_score` — stream-level redundancy characterization

**Tier 3 — Memory**
- `memory_footprint_bytes` — total bytes: embeddings + hash metadata + bucket count tables

All metrics are computed together via `evaluate_policy_complete(...)`, returning a flat dict suitable for a DataFrame row.

---

## Installation

**Requirements:** Python 3.11 (recommended for FALCONN compatibility)

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

```
numpy<2.0       # FALCONN requires NumPy 1.x
scipy
matplotlib
seaborn
sentence-transformers
datasets
tqdm
scikit-learn
pandas
h5py
hnswlib
faiss-cpu
tabulate
falconn
```

> **Note:** `falconn` may require building from source on some platforms. See `scripts/build_falconnpp.sh`.

---

## Running Tests

```bash
python tests/test_memory_bounds.py
```

Verifies that all six policies strictly respect the memory budget `B` after streaming `N=1000` items. Expected output:

```
Testing memory bounds for 6 policies (Stream N=1000, Budget B=100)...
  [PASS] FIFO: retained 100 items (<= 100)
  [PASS] Reservoir: retained 100 items (<= 100)
  [PASS] Random: retained ~100 items (<= 100)
  [PASS] SemanticDedup: retained N items (<= 100)
  [PASS] Stream-LSH: retained 100 items (<= 100)
  [PASS] BucketOccupancy: retained 100 items (<= 100)

All policies strictly respect the memory bound!
```

---

## Quick Usage Example

```python
import numpy as np
from src.retention_policies import BucketOccupancyRetention
from src.evaluation import stream_through_policy, evaluate_recall

# Load your data
base = np.load("data/sift1m/base.npy")        # (1M, 128)
queries = np.load("data/sift1m/query.npy")    # (10K, 128)
gt = np.load("data/sift1m/groundtruth.npy")   # (10K, 100)

# Create a policy with a memory budget of 50K items
policy = BucketOccupancyRetention(
    dim=128,
    L=8,            # number of hash tables
    K=10,           # hash bits per table
    capacity=50_000,
    aggregator='median',
)

# Stream all 1M items through the policy
source_ids = np.arange(len(base))
stats = stream_through_policy(policy, base, source_ids)

print(f"Kept {stats['kept']:,} / {stats['stream_size']:,} items")
print(f"Throughput: {stats['throughput_items_per_sec']:,.0f} items/s")

# Evaluate recall@10
results = evaluate_recall(policy, queries, gt, k=10)
print(f"Recall@10: {results['recall']:.4f}")
```

---

## Synthetic Stream Generation

`scripts/prepare_streams.py` generates streams with controlled properties for controlled experiments:

| Stream Type | Description |
|-------------|-------------|
| `CleanStream_0` | No duplicates — baseline stream |
| `HeavyDuplication_50` | 50% of items are near-duplicates of earlier items |
| `TopicDrift` | Stream gradually shifts topic distribution over time |

These synthetic streams are used in Phase 4 framing experiments to measure how each policy responds to different redundancy regimes.

---

## Results Overview

All experiment outputs land in `results/`. Key files:

| File | Contents |
|------|----------|
| `phase3_sift1m.csv` / `_pareto.png` | Retention policy recall vs budget on SIFT1M |
| `phase3_msmarco.csv` / `_pareto.png` | Retention policy recall vs budget on MS MARCO |
| `phase4_sift1m.csv` / `_pareto.png` | Phase 4 optimizations vs Phase 3 baselines (SIFT1M) |
| `phase4_msmarco.csv` / `_pareto.png` | Phase 4 optimizations vs Phase 3 baselines (MS MARCO) |
| `phase5_locomo/` | LLM QA accuracy by policy and memory budget |
| `phase6_jl/` | Recall-memory tradeoff under JL projection |
| `phase2_characterization/` | LSH bucket distribution plots |
| `profiling_*.csv` / `*.png` | Per-step time breakdown (FALCONN profiling) |

---

## References

- Andoni, A. & Indyk, P. (2008). *Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions.* CACM.
- Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., & Schmidt, L. (2015). *Practical and optimal LSH for angular distance.* NeurIPS.
- Kraus, O., Carmel, L., & Keidar, U. (2017). *Stream-LSH: Fast and Memory-Bounded Nearest Neighbor Search in Data Streams.* IEEE BigData.
- Gong, Y. & Lazebnik, S. (2011). *Iterative quantization: A procrustean approach to learning binary codes.* CVPR.
- Johnson, W. & Lindenstrauss, J. (1984). *Extensions of Lipschitz mappings into a Hilbert space.* Contemporary Mathematics.
