#!/usr/bin/env python3
"""
Phase 2.1: Reproduce HNSW baseline on SIFT1M.
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
    parser = argparse.ArgumentParser(description="Reproduce HNSW SIFT1M baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sift1m"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase2/hnsw_sift1m.json"),
    )
    return parser.parse_args()


def main() -> None:
    enforce_single_thread()

    try:
        import hnswlib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "hnswlib is required. Install it first (pip install hnswlib)."
        ) from exc

    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    base = np.load(args.data_dir / "base.npy", mmap_mode="r")
    query = np.load(args.data_dir / "query.npy", mmap_mode="r")
    gt = np.load(args.data_dir / "groundtruth.npy", mmap_mode="r")

    if args.max_queries > 0:
        query = query[: args.max_queries]
        gt = gt[: args.max_queries]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.init_index(
        max_elements=base.shape[0],
        M=args.M,
        ef_construction=args.ef_construction,
    )
    index.add_items(base, np.arange(base.shape[0]))
    index.set_ef(args.ef_search)

    def query_fn(q: np.ndarray, k: int) -> np.ndarray:
        labels, _ = index.knn_query(q.reshape(1, -1), k=k)
        return labels[0]

    result = benchmark(
        query_fn=query_fn,
        queries=np.asarray(query),
        groundtruth=np.asarray(gt),
        k=args.k,
        warmup=args.warmup,
    )

    payload = {
        "method": "hnswlib",
        "dataset": "sift1m",
        "params": {
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "k": args.k,
            "warmup": args.warmup,
            "max_queries": args.max_queries,
        },
        "metrics": result.as_dict(),
    }

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("HNSW reproduction result")
    print(json.dumps(payload, indent=2))
    print("\nTarget sanity: recall@10 should be ~0.94-0.97.")


if __name__ == "__main__":
    main()
