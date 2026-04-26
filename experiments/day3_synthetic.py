"""
Day 3 — Synthetic integration test.

10 clusters × 20 vectors = 200 total (95% redundancy), dim=50.
Runs LSHAdmissionFilter (k=8, L=8) and BruteForceOracle on the same stream,
then prints a side-by-side comparison table.

Expected results
----------------
Oracle   : stores exactly 10 vectors (one per cluster)
Filter   : stores 10–30 vectors (some hash collisions may admit extras,
           but should never reach 150+)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lsh    import LSHAdmissionFilter
from oracle import BruteForceOracle
from stream import synthetic_stream
from evaluate import compute_metrics, print_metrics

# ── Parameters ────────────────────────────────────────────────────────────────
N_CLUSTERS   = 10
CLUSTER_SIZE = 20
DIM          = 50
K            = 8      # hash bits per table
L            = 8      # number of tables
THRESHOLD    = 0.9
SEED         = 42

# ── Generate stream ───────────────────────────────────────────────────────────
vectors, labels = synthetic_stream(
    n_clusters=N_CLUSTERS,
    cluster_size=CLUSTER_SIZE,
    dim=DIM,
    noise_sigma=0.02,
    seed=SEED,
)

n_total   = len(vectors)
n_novel   = sum(labels)
n_redund  = n_total - n_novel
print(f"Stream: {n_total} vectors  |  novel={n_novel}  redundant={n_redund}  "
      f"redundancy={n_redund/n_total:.1%}")

# ── Run filter and oracle on the same stream ──────────────────────────────────
filt   = LSHAdmissionFilter(dim=DIM, k=K, L=L,
                             similarity_threshold=THRESHOLD, seed=SEED)
oracle = BruteForceOracle(similarity_threshold=THRESHOLD)

filter_admits = []
oracle_admits = []

for vec in vectors:
    filter_admits.append(filt.insert(vec))
    oracle_admits.append(oracle.insert(vec))

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics = compute_metrics(filter_admits, oracle_admits)

print()
print("=" * 60)
print(f"{'Metric':<25} {'Filter':>10}  {'Oracle':>10}")
print("-" * 60)
print(f"{'Vectors stored':<25} {len(filt):>10}  {len(oracle):>10}")
print(f"{'Compression ratio':<25} {metrics['compression_ratio']:>10.3f}  "
      f"{len(oracle)/n_total:>10.3f}")
print(f"{'Novelty precision':<25} {metrics['precision']:>10.3f}  {'1.000':>10}")
print(f"{'Novelty recall':<25} {metrics['recall']:>10.3f}  {'1.000':>10}")
print(f"{'TP / FP / FN / TN':<25} "
      f"{metrics['tp']:>3}/{metrics['fp']:>3}/{metrics['fn']:>3}/{metrics['tn']:>3}  "
      f"{'':>10}")
print("=" * 60)
print()
print_metrics(metrics, label="LSH filter vs oracle")

# ── Sanity assertions ─────────────────────────────────────────────────────────
assert len(oracle) == N_CLUSTERS, \
    f"Oracle must store exactly {N_CLUSTERS}, got {len(oracle)}"

assert len(filt) <= 150, \
    (f"Filter stored {len(filt)} vectors — k is too low or threshold too loose. "
     f"Expected <= 150.")

assert metrics["recall"] >= 0.5, \
    (f"Recall {metrics['recall']:.3f} is dangerously low — "
     f"filter is rejecting too many novel vectors.")

print(f"\nSanity checks passed.")
print(f"  Oracle stored  : {len(oracle)} (expected exactly {N_CLUSTERS})")
print(f"  Filter stored  : {len(filt)} (expected 10–30, hard limit 150)")
