#!/usr/bin/env python3
"""
Phase 2 fallback: reproduce a classical LSH baseline on SIFT1M using FAISS IndexLSH.

Use this when FALCONN cannot be installed on the local machine.
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
    parser = argparse.ArgumentParser(description="Run FAISS IndexLSH baseline on SIFT1M.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sift1m"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--nbits", type=int, default=256)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase2/faiss_lsh_sift1m.json"),
    )
    return parser.parse_args()


def main() -> None:
    enforce_single_thread()

    try:
        import faiss
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss is required. Install it first (pip install faiss-cpu).") from exc

    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    base = np.load(args.data_dir / "base.npy", mmap_mode="r").astype(np.float32)
    query = np.load(args.data_dir / "query.npy", mmap_mode="r").astype(np.float32)
    gt = np.load(args.data_dir / "groundtruth.npy", mmap_mode="r")

    if args.max_queries > 0:
        query = query[: args.max_queries]
        gt = gt[: args.max_queries]

    dim = int(base.shape[1])
    index = faiss.IndexLSH(dim, args.nbits)
    index.add(base)

    def query_fn(q: np.ndarray, k: int) -> np.ndarray:
        _, labels = index.search(q.reshape(1, -1), k)
        return labels[0]

    result = benchmark(
        query_fn=query_fn,
        queries=query,
        groundtruth=np.asarray(gt),
        k=args.k,
        warmup=args.warmup,
    )

    payload = {
        "method": "faiss_index_lsh_fallback",
        "dataset": "sift1m",
        "params": {
            "nbits": args.nbits,
            "k": args.k,
            "warmup": args.warmup,
            "max_queries": args.max_queries,
        },
        "metrics": result.as_dict(),
        "note": "Fallback baseline used because FALCONN is unavailable in local environment.",
    }

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("FAISS IndexLSH fallback result")
    print(json.dumps(payload, indent=2))
    print("\nThis is a fallback baseline, not a substitute for FALCONN quality.")


if __name__ == "__main__":
    main()
