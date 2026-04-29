#!/usr/bin/env python3
"""
Phase 2.2: Reproduce FALCONN baseline on SIFT1M.

Notes:
- Uses cross-polytope LSH via FALCONN.
- Parameter defaults here are practical starter values and should be swept
  for close reproduction to published ann-benchmarks points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_harness import benchmark, enforce_single_thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce FALCONN SIFT1M baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sift1m"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--num-hash-tables", type=int, default=20)
    parser.add_argument("--num-hash-bits", type=int, default=18)
    parser.add_argument("--num-probes", type=int, default=64)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase2/falconn_sift1m.json"),
    )
    return parser.parse_args()


def main() -> None:
    enforce_single_thread()

    try:
        import falconn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "falconn is required. Install it first (pip install falconn). "
            "If it fails to build on your platform, run the fallback script: "
            "python scripts/phase2/reproduce_faiss_lsh_sift1m.py"
        ) from exc

    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    base = np.load(args.data_dir / "base.npy", mmap_mode="r").astype(np.float32)
    query = np.load(args.data_dir / "query.npy", mmap_mode="r").astype(np.float32)
    gt = np.load(args.data_dir / "groundtruth.npy", mmap_mode="r")

    if args.max_queries > 0:
        query = query[: args.max_queries]
        gt = gt[: args.max_queries]

    params = falconn.get_default_parameters(base.shape[0], base.shape[1])
    params.lsh_family = falconn.LSHFamily.CrossPolytope
    params.distance_function = falconn.DistanceFunction.EuclideanSquared
    params.num_setup_threads = 0
    params.l = args.num_hash_tables
    params.k = args.num_hash_bits

    index = falconn.LSHIndex(params)
    index.setup(base)
    query_object = index.construct_query_object()
    query_object.set_num_probes(args.num_probes)

    def query_fn(q: np.ndarray, k: int) -> np.ndarray:
        # FALCONN may return fewer than k if search space is sparse.
        pred = query_object.find_k_nearest_neighbors(q, k)
        return np.asarray(pred, dtype=np.int32)

    result = benchmark(
        query_fn=query_fn,
        queries=query,
        groundtruth=np.asarray(gt),
        k=args.k,
        warmup=args.warmup,
    )

    payload = {
        "method": "falconn_cross_polytope",
        "dataset": "sift1m",
        "params": {
            "num_hash_tables": args.num_hash_tables,
            "num_hash_bits": args.num_hash_bits,
            "num_probes": args.num_probes,
            "k": args.k,
            "warmup": args.warmup,
            "max_queries": args.max_queries,
        },
        "metrics": result.as_dict(),
    }

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("FALCONN reproduction result")
    print(json.dumps(payload, indent=2))
    print("\nTune (tables, bits, probes) to hit published ballpark.")


if __name__ == "__main__":
    main()
