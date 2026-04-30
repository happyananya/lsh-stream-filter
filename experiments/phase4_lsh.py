#!/usr/bin/env python3
"""
Phase 4: Layered LSH optimizations — cross-polytope hashing + reranking.

Steps implemented here:
  4.1  Custom cross-polytope hashing (Andoni et al., 2015)
  4.2  Multi-probe LSH (fewer tables, T perturbed buckets per table)
  4.5  Reranking with exact distance (the single biggest recall win)

Stubs ready for subsequent commits:
  4.3  PCA preprocessing      → PCATransform class
  4.4  ITQ rotation            → ITQRotation class
  4.6  Bit-packing + SIMD      → bit_packed_hamming stub
  4.7  Learned projections     → separate file

Run:
  python experiments/phase4_lsh.py --dataset sift1m
  python experiments/phase4_lsh.py --dataset msmarco
  python experiments/phase4_lsh.py --dataset sift1m --step 4.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "phase2"))
from benchmark_harness import benchmark, enforce_single_thread


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def fwht_batch(X: np.ndarray) -> np.ndarray:
    """
    Normalized Fast Walsh-Hadamard Transform for a batch of row vectors.
    X.shape must be (n, d) where d is a power of 2.
    O(n * d * log d) — keeps the rotation at O(d log d) per vector.
    """
    X = X.astype(np.float32, copy=True)
    n, d = X.shape
    h = 1
    while h < d:
        X = X.reshape(n, d // (2 * h), 2, h)
        a = X[:, :, 0, :].copy()
        b = X[:, :, 1, :].copy()
        X[:, :, 0, :] = a + b
        X[:, :, 1, :] = a - b
        X = X.reshape(n, d)
        h *= 2
    return X / np.sqrt(d)


# ---------------------------------------------------------------------------
# Cross-polytope hash table  (Step 4.1)
# ---------------------------------------------------------------------------

class CrossPolytopeHashTable:
    """
    One LSH table using cross-polytope hashing.

    Mapping x ∈ R^d → bucket ∈ [0, 2 * padded_dim):
      1. Pad x to padded_dim = next power-of-2 ≥ d
      2. For each rotation: element-wise sign-flip then FWHT
      3. bucket = argmax_i |z_i| * 2 + (0 if z[argmax] ≥ 0 else 1)

    The O(d log d) rotation (vs O(d²) dense random projection) is what
    makes cross-polytope practical at high dimensions.
    """

    def __init__(self, dim: int, num_rotations: int = 2, seed: int = 0) -> None:
        self.dim = dim
        self.padded_dim = 1 << int(np.ceil(np.log2(max(dim, 2))))
        self.num_rotations = num_rotations

        rng = np.random.default_rng(seed)
        self.sign_matrices = [
            rng.choice([-1.0, 1.0], size=self.padded_dim).astype(np.float32)
            for _ in range(num_rotations)
        ]

        self._table: dict[int, list[int]] = {}
        self._base: Optional[np.ndarray] = None

    def _rotate_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply num_rotations of (sign-flip + FWHT) to each row of X."""
        if X.shape[1] < self.padded_dim:
            X = np.pad(X, ((0, 0), (0, self.padded_dim - X.shape[1])))
        X = X.astype(np.float32, copy=True)
        for signs in self.sign_matrices:
            X = X * signs
            X = fwht_batch(X)
        return X

    def hash_batch(self, X: np.ndarray) -> np.ndarray:
        """Return bucket ids of shape (n,) for a batch of vectors."""
        Z = self._rotate_batch(X)
        argmax = np.argmax(np.abs(Z), axis=1)
        neg_sign = (Z[np.arange(len(Z)), argmax] < 0).astype(np.int32)
        return argmax * 2 + neg_sign

    def build(self, base: np.ndarray) -> None:
        self._base = base
        bucket_ids = self.hash_batch(base)
        self._table = {}
        for idx, bid in enumerate(bucket_ids.tolist()):
            self._table.setdefault(bid, []).append(idx)

    def probe(self, q: np.ndarray, num_probes: int) -> list[int]:
        """
        Return candidates from up to num_probes buckets.

        Probe sequence: coordinates of the rotated query sorted by |z_i|
        descending.  The primary bucket is argmax; secondary probes replace
        that argmax with the 2nd, 3rd, … largest coordinates — exactly the
        highest-probability adjacent buckets under the cross-polytope model.
        """
        Z = self._rotate_batch(q.reshape(1, -1))[0]
        top_coords = np.argsort(np.abs(Z))[::-1][:num_probes]
        neg_signs = (Z[top_coords] < 0).astype(np.int32)
        bucket_ids = top_coords * 2 + neg_signs
        candidates: list[int] = []
        for bid in bucket_ids.tolist():
            candidates.extend(self._table.get(bid, []))
        return candidates


# ---------------------------------------------------------------------------
# Multi-probe index  (Step 4.1 + 4.2)
# ---------------------------------------------------------------------------

class MultiProbeLSHIndex:
    """
    Multi-probe cross-polytope LSH index.

    L tables × T probes per table replaces L=100 tables × 1 probe:
    same recall, ~10× less memory.  Optional exact-distance reranking
    (Step 4.5) is the analog of HNSW's efSearch.
    """

    def __init__(
        self,
        dim: int,
        num_tables: int = 16,
        num_rotations: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.num_tables = num_tables
        # 2 rotations for d ≤ 128, 1 for higher dims (less overhead vs marginal quality gain)
        self.num_rotations = num_rotations if num_rotations is not None else (2 if dim <= 128 else 1)
        self._base: Optional[np.ndarray] = None

        self.tables = [
            CrossPolytopeHashTable(dim, num_rotations=self.num_rotations, seed=seed + i)
            for i in range(num_tables)
        ]

    def build(self, base: np.ndarray) -> None:
        self._base = base
        for i, table in enumerate(self.tables):
            table.build(base)
            if (i + 1) % 4 == 0 or (i + 1) == len(self.tables):
                print(f"    Built {i + 1}/{len(self.tables)} tables", flush=True)

    def _gather(self, q: np.ndarray, num_probes: int) -> np.ndarray:
        """Union of candidates across all tables, deduplicated."""
        seen: set[int] = set()
        for table in self.tables:
            seen.update(table.probe(q, num_probes))
        return np.fromiter(seen, dtype=np.int64, count=len(seen))

    def query(self, q: np.ndarray, k: int, num_probes: int = 64) -> np.ndarray:
        """Top-k by exact L2 over the candidate union (no rerank cap)."""
        candidates = self._gather(q, num_probes)
        if len(candidates) == 0:
            return np.zeros(k, dtype=np.int64)
        vecs = self._base[candidates]
        dists = np.einsum("ij,ij->i", vecs - q, vecs - q)
        top = min(k, len(candidates))
        idx = np.argpartition(dists, top - 1)[:top]
        return candidates[idx[np.argsort(dists[idx])]]

    def query_with_rerank(
        self, q: np.ndarray, k: int, num_probes: int = 64, rerank_m: int = 500
    ) -> np.ndarray:
        """
        Step 4.5 — gather candidates then exact-distance rerank.

        If candidates > rerank_m, keep the rerank_m closest by IP (fast
        pre-filter) before paying for exact L2.  This is the analog of
        HNSW's efSearch and typically gives +0.05–0.20 recall lift.
        """
        candidates = self._gather(q, num_probes)
        if len(candidates) == 0:
            return np.zeros(k, dtype=np.int64)

        vecs = self._base[candidates]

        if len(candidates) > rerank_m:
            # Fast IP pre-filter to trim candidate pool
            ip_scores = vecs @ q
            top_pre = np.argpartition(ip_scores, -rerank_m)[-rerank_m:]
            candidates = candidates[top_pre]
            vecs = vecs[top_pre]

        dists = np.einsum("ij,ij->i", vecs - q, vecs - q)
        top = min(k, len(candidates))
        idx = np.argpartition(dists, top - 1)[:top]
        return candidates[idx[np.argsort(dists[idx])]]


# ---------------------------------------------------------------------------
# Stubs for Steps 4.3 – 4.7
# ---------------------------------------------------------------------------

class PCATransform:
    """
    Step 4.3 — PCA preprocessing.

    Fit on the base set, apply the same projection at query time.
    For MS MARCO: d=384 → d'=128 or 256 cuts hash cost and improves bucket quality.
    For SIFT1M:   d=128 → d'=64.
    """

    def fit(self, X: np.ndarray, n_components: int) -> "PCATransform":
        raise NotImplementedError(
            "Fit PCA on base vectors (or 100k sample).\n"
            "  mean = X.mean(0)\n"
            "  _, _, Vt = np.linalg.svd((X - mean), full_matrices=False)\n"
            "  self.components = Vt[:n_components]  # shape (n_components, d)\n"
            "  self.mean = mean"
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("return (X - self.mean) @ self.components.T")


class ITQRotation:
    """
    Step 4.4 — Iterative Quantization (Gong & Lazebnik, 2011).

    After PCA, learn rotation R minimising ||sign(R @ X.T) - R @ X.T||.
    ~50 iterations of alternating SVD.  Typically yields +1–3 recall points
    at fixed bit budget with negligible extra cost at query time.
    """

    def fit(self, X: np.ndarray, n_iter: int = 50) -> "ITQRotation":
        raise NotImplementedError(
            "  R = np.eye(X.shape[1])\n"
            "  for _ in range(n_iter):\n"
            "      B = np.sign(X @ R.T)          # binarize\n"
            "      U, _, Vt = np.linalg.svd(B.T @ X)  # SVD\n"
            "      R = (U @ Vt).T               # update rotation\n"
            "  self.R = R"
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("return X @ self.R.T")


def bit_packed_hamming_stub(
    query_bits: np.ndarray, base_bits: np.ndarray
) -> np.ndarray:
    """
    Step 4.6 — Bit-packing + SIMD popcount (stub).

    For prototyping use numpy.bitwise_count (NumPy ≥ 2.0).
    For production drop to C via ctypes or pybind11 — pure NumPy leaves
    5–10× QPS on the table.  Profile with py-spy before and after.

    Expected signature:
      query_bits: uint64 array of shape (n_words,)   — packed query signature
      base_bits:  uint64 array of shape (N, n_words) — packed base signatures
      returns:    int32 array of shape (N,)           — Hamming distances
    """
    raise NotImplementedError(
        "Pack signatures: sigs_u64 = np.packbits(sigs).view(np.uint64)\n"
        "Popcount loop:  np.bitwise_count(xor).sum(axis=1)  # NumPy ≥ 2.0\n"
        "Hot path:       ctypes call into popcount_avx512 for production"
    )


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def sweep_step41_42(
    base: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    output_rows: list[dict],
    label: str = "",
) -> None:
    """Step 4.1+4.2: cross-polytope multi-probe, no reranking."""
    print(f"\n[Step 4.1+4.2] Cross-polytope multi-probe {label}")
    dim = base.shape[1]

    for num_tables in [8, 16, 32]:
        print(f"  Building L={num_tables} tables...")
        idx = MultiProbeLSHIndex(dim=dim, num_tables=num_tables)
        t0 = time.time()
        idx.build(base)
        build_time = time.time() - t0
        print(f"  Built in {build_time:.1f}s")

        for num_probes in [4, 16, 64, 256]:
            print(f"    L={num_tables}, T={num_probes}")

            def make_qfn(index: MultiProbeLSHIndex, np_: int):
                def qfn(q: np.ndarray, k: int) -> np.ndarray:
                    return index.query(q, k, num_probes=np_)
                return qfn

            stats = benchmark(make_qfn(idx, num_probes), queries, gt, k=10)
            output_rows.append({
                "Method": f"CP-LSH{label}",
                "Parameters": f"L={num_tables}, T={num_probes}",
                "Step": "4.1+4.2",
                "Build_Time_s": round(build_time, 3),
                **stats.as_dict(),
            })


def sweep_step45(
    base: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    output_rows: list[dict],
    label: str = "",
) -> None:
    """Step 4.5: cross-polytope + exact-distance reranking."""
    print(f"\n[Step 4.5] Reranking sweep {label}")
    dim = base.shape[1]

    # Fix L=16 (best memory/recall tradeoff from 4.2); sweep T and rerank_m
    num_tables = 16
    idx = MultiProbeLSHIndex(dim=dim, num_tables=num_tables)
    t0 = time.time()
    idx.build(base)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")

    for num_probes in [16, 64]:
        for rerank_m in [100, 200, 500, 1000, 2000]:
            print(f"    L={num_tables}, T={num_probes}, rerank_m={rerank_m}")

            def make_qfn(index: MultiProbeLSHIndex, np_: int, rm: int):
                def qfn(q: np.ndarray, k: int) -> np.ndarray:
                    return index.query_with_rerank(q, k, num_probes=np_, rerank_m=rm)
                return qfn

            stats = benchmark(make_qfn(idx, num_probes, rerank_m), queries, gt, k=10)
            output_rows.append({
                "Method": f"CP-LSH+Rerank{label}",
                "Parameters": f"L={num_tables}, T={num_probes}, m={rerank_m}",
                "Step": "4.5",
                "Build_Time_s": round(build_time, 3),
                **stats.as_dict(),
            })


# ---------------------------------------------------------------------------
# Cumulative Pareto plot
# ---------------------------------------------------------------------------

def plot_cumulative_pareto(
    phase3_csv: str,
    phase4_rows: list[dict],
    output_path: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    def pareto_front(df_grp: pd.DataFrame) -> np.ndarray:
        pts = df_grp[["recall", "qps"]].values
        pts = pts[pts[:, 0].argsort()[::-1]]
        front = [pts[0]]
        for p in pts[1:]:
            if p[1] >= front[-1][1]:
                front.append(p)
        arr = np.array(front)
        return arr[arr[:, 0].argsort()]

    fig, ax = plt.subplots(figsize=(11, 8))

    # Phase 3 baselines (dashed, muted)
    df3 = pd.read_csv(phase3_csv)
    colors3 = {
        "HNSW": "#d62728",
        "Vanilla LSH (FAISS)": "#aec7e8",
        "FALCONN": "#ffbb78",
    }
    for method, grp in df3.groupby("Method"):
        front = pareto_front(grp)
        ax.plot(
            front[:, 0], front[:, 1],
            marker="o", linewidth=1.5, linestyle="--", alpha=0.65,
            label=f"{method} (Ph3)", color=colors3.get(method, "#aaa"),
        )

    # Phase 4 steps (solid, bold)
    if phase4_rows:
        df4 = pd.DataFrame(phase4_rows)
        colors4 = {
            "CP-LSH (sift1m)": "#2ca02c",
            "CP-LSH+Rerank (sift1m)": "#9467bd",
            "CP-LSH (msmarco)": "#17becf",
            "CP-LSH+Rerank (msmarco)": "#e377c2",
        }
        for method, grp in df4.groupby("Method"):
            front = pareto_front(grp)
            ax.plot(
                front[:, 0], front[:, 1],
                marker="s", linewidth=2.2, linestyle="-",
                label=method, color=colors4.get(method, "#333"),
            )

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10", fontsize=13)
    ax.set_ylabel("QPS (log scale)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0.0, 1.02)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Pareto plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    enforce_single_thread()

    parser = argparse.ArgumentParser(description="Phase 4 LSH optimizations")
    parser.add_argument("--dataset", choices=["sift1m", "msmarco"], default="sift1m")
    parser.add_argument(
        "--step", choices=["4.1", "4.5", "all"], default="all",
        help="4.1=cross-polytope+multi-probe, 4.5=+reranking, all=both",
    )
    args = parser.parse_args()

    data_root = Path("data")
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)

    ds = args.dataset
    if ds == "sift1m":
        print("Loading SIFT1M...")
        base = np.ascontiguousarray(np.load(data_root / "sift1m" / "base.npy").astype(np.float32))
        queries = np.ascontiguousarray(np.load(data_root / "sift1m" / "query.npy").astype(np.float32))
        gt = np.load(data_root / "sift1m" / "groundtruth.npy")
        phase3_csv = str(results_root / "phase3_sift1m.csv")
        out_csv = str(results_root / "phase4_sift1m.csv")
        out_plot = str(results_root / "phase4_sift1m_pareto.png")
        title = "Phase 4 vs Phase 3 Pareto — SIFT1M"
        label = " (sift1m)"
    else:
        print("Loading MS MARCO...")
        base = np.ascontiguousarray(np.load(data_root / "msmarco" / "base.npy").astype(np.float32))
        queries = np.ascontiguousarray(np.load(data_root / "msmarco" / "query.npy").astype(np.float32))
        gt = np.load(data_root / "msmarco" / "groundtruth.npy")
        phase3_csv = str(results_root / "phase3_msmarco.csv")
        out_csv = str(results_root / "phase4_msmarco.csv")
        out_plot = str(results_root / "phase4_msmarco_pareto.png")
        title = "Phase 4 vs Phase 3 Pareto — MS MARCO"
        label = " (msmarco)"

    rows: list[dict] = []

    if args.step in ("4.1", "all"):
        sweep_step41_42(base, queries, gt, rows, label=label)

    if args.step in ("4.5", "all"):
        sweep_step45(base, queries, gt, rows, label=label)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    plot_cumulative_pareto(phase3_csv, rows, out_plot, title)
    print("Done.")


if __name__ == "__main__":
    main()
