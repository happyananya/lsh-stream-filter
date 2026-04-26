"""
Day 5 — Parameter sweep, heatmaps, memory-over-time, and theory vs empirical.

Outputs (all saved to results/):
  sweep_results.csv
  recall_heatmap.png
  precision_heatmap.png
  theory_vs_empirical.png
  memory_over_time.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from lsh      import LSHAdmissionFilter
from oracle   import BruteForceOracle
from stream   import (real_stream, empirical_near_dup_angle,
                      participation_ratio, _load_or_encode)
from evaluate import compute_metrics, FIFOBaseline, RandomEvictionBaseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE_PATH  = os.path.join(os.path.dirname(__file__), "..", "embeddings.npy")
N_PASSAGES  = 5000
SEED        = 42

K_VALUES          = [4, 8, 16]
L_VALUES          = [4, 8, 16]
REDUNDANCY_LEVELS = [0.2, 0.5, 0.8]
THRESHOLD         = 0.9

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Load embeddings & diagnostics ─────────────────────────────────────────────
print("Loading embeddings …")
embeddings = _load_or_encode(CACHE_PATH, N_PASSAGES)
DIM = embeddings.shape[1]

print()
mean_angle_deg = empirical_near_dup_angle(
    embeddings.astype(np.float64), sim_threshold=0.8, n_sample=500, seed=SEED)
pr = participation_ratio(embeddings.astype(np.float64))
print()


# ── Sweep ─────────────────────────────────────────────────────────────────────
print("Running sweep …")
rows = []

for redundancy in REDUNDANCY_LEVELS:
    # Generate stream and run oracle ONCE per redundancy level
    vectors, labels = real_stream(
        redundancy=redundancy, noise_sigma=0.01,
        seed=SEED, cache_path=CACHE_PATH, n_passages=N_PASSAGES)

    oracle = BruteForceOracle(similarity_threshold=THRESHOLD)
    oracle_admits = [oracle.insert(v) for v in vectors]

    for k in K_VALUES:
        for L in L_VALUES:
            filt = LSHAdmissionFilter(
                dim=DIM, k=k, L=L,
                similarity_threshold=THRESHOLD, seed=SEED)
            filter_admits = [filt.insert(v) for v in vectors]

            m = compute_metrics(filter_admits, oracle_admits)
            rows.append(dict(
                k=k, L=L, redundancy=redundancy,
                precision=m["precision"],
                recall=m["recall"],
                compression=m["compression_ratio"],
                false_admit_rate=m["fp"] / max(m["fp"] + m["tn"], 1),
                false_reject_rate=m["fn"] / max(m["fn"] + m["tp"], 1),
                tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
                filter_stored=len(filt),
                oracle_stored=sum(oracle_admits),
                n_stream=len(vectors),
            ))
            print(f"  k={k:2d}  L={L:2d}  red={redundancy:.0%}  "
                  f"prec={m['precision']:.3f}  recall={m['recall']:.3f}  "
                  f"compress={m['compression_ratio']:.3f}")

df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, "sweep_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")


# ── Heatmaps at redundancy = 0.5 ──────────────────────────────────────────────
df_05 = df[df["redundancy"] == 0.5]

for metric, label, fname in [
    ("recall",    "Novelty Recall",    "recall_heatmap.png"),
    ("precision", "Novelty Precision", "precision_heatmap.png"),
]:
    pivot = df_05.pivot(index="L", columns="k", values=metric)
    # index=L (rows), columns=k (cols) — keep L ascending top-to-bottom
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="YlGnBu", vmin=0.0, vmax=1.0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"{label} vs k and L  (redundancy=50%)", fontsize=12)
    ax.set_xlabel("k  (hash bits per table)")
    ax.set_ylabel("L  (number of tables)")
    path = os.path.join(RESULTS_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Theory vs Empirical ───────────────────────────────────────────────────────
# Theoretical curves at fixed L=8
L_THEORY = 8
k_range  = np.array(K_VALUES)

# Near-duplicate collision probability per bit (SimHash)
theta_dup_rad  = np.radians(mean_angle_deg)
p_dup          = 1.0 - theta_dup_rad / np.pi          # P(same bit | near-dup pair)

# Novel vector collision probability per bit (assume θ≈90°)
p_novel        = 0.5                                   # 1 - (π/2)/π

# Theoretical false admit rate  = P(no collision in any table)
theory_FA = (1.0 - p_dup ** k_range) ** L_THEORY

# Theoretical false reject rate = P(spurious collision in at least one table)
theory_FR = 1.0 - (1.0 - p_novel ** k_range) ** L_THEORY

# Empirical rates at L=8, averaged over redundancy levels
df_L8 = df[df["L"] == L_THEORY].groupby("k")[
    ["false_admit_rate", "false_reject_rate"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

ax.plot(k_range, theory_FA, "b--o",  label="Theory: false admit rate")
ax.plot(k_range, theory_FR, "r--s",  label="Theory: false reject rate")
ax.plot(df_L8["k"], df_L8["false_admit_rate"],  "b-o",
        label="Empirical: false admit rate")
ax.plot(df_L8["k"], df_L8["false_reject_rate"], "r-s",
        label="Empirical: false reject rate")

ax.set_xlabel("k  (hash bits per table)")
ax.set_ylabel("Error rate")
ax.set_title(
    f"Theory vs Empirical error rates  (L={L_THEORY}, θ_dup≈{mean_angle_deg:.1f}°, "
    f"PR={pr:.0f}/{DIM})",
    fontsize=10)
ax.legend(fontsize=8)
ax.set_xticks(k_range)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

path = os.path.join(RESULTS_DIR, "theory_vs_empirical.png")
fig.tight_layout()
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")


# ── Memory over time ──────────────────────────────────────────────────────────
# k=8, L=8, redundancy=0.5 — replay stream, track cumulative stored count
print("\nBuilding memory-over-time traces …")

vectors_05, _ = real_stream(
    redundancy=0.5, noise_sigma=0.01,
    seed=SEED, cache_path=CACHE_PATH, n_passages=N_PASSAGES)

# Run oracle to get true stored count as a reference capacity for baselines
oracle_ref = BruteForceOracle(similarity_threshold=THRESHOLD)
oracle_cumulative = []
for v in vectors_05:
    oracle_ref.insert(v)
    oracle_cumulative.append(len(oracle_ref))

capacity = len(oracle_ref)  # fair baseline capacity = oracle final count

filt_mem   = LSHAdmissionFilter(dim=DIM, k=8, L=8,
                                 similarity_threshold=THRESHOLD, seed=SEED)
fifo       = FIFOBaseline(capacity=capacity)
rand_evict = RandomEvictionBaseline(capacity=capacity, seed=SEED)

filter_cumulative = []
fifo_cumulative   = []
rand_cumulative   = []

for v in vectors_05:
    filt_mem.insert(v)
    fifo.insert(v)
    rand_evict.insert(v)
    # FIFOBaseline/RandomEviction: current buffer size
    filter_cumulative.append(len(filt_mem))
    fifo_cumulative.append(len(fifo))
    rand_cumulative.append(len(rand_evict))

x = np.arange(len(vectors_05))

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

# Thin the x axis for a lighter plot (plot every 50th point)
step = max(1, len(x) // 200)
ax.plot(x[::step], np.array(filter_cumulative)[::step],
        color="steelblue", linewidth=1.5, label="LSH filter (k=8, L=8)")
ax.plot(x[::step], np.array(oracle_cumulative)[::step],
        color="green",     linewidth=1.5, linestyle="--", label="Oracle (ground truth)")
ax.axhline(capacity, color="darkorange", linewidth=1.5,
           linestyle="-.",  label=f"FIFO (cap={capacity})")
ax.axhline(capacity, color="crimson",    linewidth=1.0,
           linestyle=":",   label=f"Random eviction (cap={capacity})")

ax.set_xlabel("Stream position")
ax.set_ylabel("Vectors in memory")
ax.set_title("Memory over time  (redundancy=50%, k=8, L=8)", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

path = os.path.join(RESULTS_DIR, "memory_over_time.png")
fig.tight_layout()
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")


# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 80)
print(f"{'k':>4}  {'L':>4}  {'red':>5}  {'precision':>10}  {'recall':>8}  "
      f"{'compress':>9}  {'FA_rate':>8}  {'FR_rate':>8}")
print("-" * 80)
for _, r in df.sort_values(["redundancy", "k", "L"]).iterrows():
    print(f"{int(r.k):>4}  {int(r.L):>4}  {r.redundancy:>5.0%}  "
          f"{r.precision:>10.3f}  {r.recall:>8.3f}  "
          f"{r.compression:>9.3f}  {r.false_admit_rate:>8.4f}  "
          f"{r.false_reject_rate:>8.4f}")
print("=" * 80)

print(f"\nAll plots saved to {RESULTS_DIR}/")
print(f"  mean near-dup angle : {mean_angle_deg:.2f}°")
print(f"  participation ratio : {pr:.1f} / {DIM} dims")
