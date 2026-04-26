"""
Day 4 — Real data pipeline.

Loads (or encodes) 5 k MS MARCO passages, constructs streams at three
redundancy levels, runs the LSH filter and brute-force oracle on each,
and prints a summary table plus embedding diagnostics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from lsh      import LSHAdmissionFilter
from oracle   import BruteForceOracle
from stream   import real_stream, empirical_near_dup_angle, participation_ratio, _load_or_encode
from evaluate import compute_metrics, print_metrics

# ── Parameters ────────────────────────────────────────────────────────────────
REDUNDANCY_LEVELS = [0.2, 0.5, 0.8]
K          = 8
L          = 8
THRESHOLD  = 0.9
SEED       = 42
N_PASSAGES = 5000

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings.npy")

# ── Embedding diagnostics (computed once) ────────────────────────────────────
print("=" * 65)
print("Embedding diagnostics")
print("=" * 65)
embeddings = _load_or_encode(CACHE_PATH, N_PASSAGES)

mean_angle = empirical_near_dup_angle(
    embeddings.astype(np.float64),
    sim_threshold=0.8,
    n_sample=500,
    seed=SEED,
)
pr = participation_ratio(embeddings.astype(np.float64))
print()

# ── Per-redundancy experiments ────────────────────────────────────────────────
results = []

for redundancy in REDUNDANCY_LEVELS:
    vectors, labels = real_stream(
        redundancy=redundancy,
        noise_sigma=0.01,
        seed=SEED,
        cache_path=CACHE_PATH,
        n_passages=N_PASSAGES,
    )

    filt   = LSHAdmissionFilter(dim=embeddings.shape[1], k=K, L=L,
                                 similarity_threshold=THRESHOLD, seed=SEED)
    oracle = BruteForceOracle(similarity_threshold=THRESHOLD)

    filter_admits = [filt.insert(v)   for v in vectors]
    oracle_admits = [oracle.insert(v) for v in vectors]

    m = compute_metrics(filter_admits, oracle_admits)
    m["redundancy"]      = redundancy
    m["n_stream"]        = len(vectors)
    m["filter_stored"]   = len(filt)
    m["oracle_stored"]   = len(oracle)
    results.append(m)

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 75)
print(f"{'Redund':>7}  {'Stream':>7}  {'F-stored':>8}  {'O-stored':>8}  "
      f"{'Precision':>10}  {'Recall':>8}  {'Compress':>9}")
print("-" * 75)
for m in results:
    print(
        f"{m['redundancy']:>7.0%}  "
        f"{m['n_stream']:>7}  "
        f"{m['filter_stored']:>8}  "
        f"{m['oracle_stored']:>8}  "
        f"{m['precision']:>10.3f}  "
        f"{m['recall']:>8.3f}  "
        f"{m['compression_ratio']:>9.3f}"
    )
print("=" * 75)
print()

for m in results:
    print_metrics(m, label=f"redundancy={m['redundancy']:.0%}")

# ── Sanity assertions ─────────────────────────────────────────────────────────
for m in results:
    r = m["redundancy"]
    assert m["recall"] >= 0.5, \
        f"Recall {m['recall']:.3f} too low at redundancy={r:.0%}"
    # Oracle should admit ≈ (1-r) fraction; filter may store a bit more
    oracle_compression = m["oracle_stored"] / m["n_stream"]
    assert oracle_compression <= (1 - r) + 0.05, \
        (f"Oracle storing too much at redundancy={r:.0%}: "
         f"oracle_compression={oracle_compression:.3f}, expected ≈{1-r:.2f}")
    assert m["compression_ratio"] <= oracle_compression * 2.0, \
        (f"Filter storing far more than oracle at redundancy={r:.0%}: "
         f"filter={m['compression_ratio']:.3f} oracle={oracle_compression:.3f}")

print("\nSanity checks passed.")
print(f"  Empirical mean near-dup angle : {mean_angle:.2f}°")
print(f"  Participation ratio           : {pr:.1f} / {embeddings.shape[1]} dims")
