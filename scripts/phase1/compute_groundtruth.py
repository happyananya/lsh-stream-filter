#!/usr/bin/env python3
"""
Compute exact top-k nearest neighbors with FAISS for Phase 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


def load_vectors(base_path: Path, query_path: Path) -> tuple[np.ndarray, np.ndarray]:
    base = np.load(base_path).astype(np.float32)
    query = np.load(query_path).astype(np.float32)
    return base, query


def build_index(dim: int, metric: str, use_gpu: bool) -> faiss.Index:
    if metric == "ip":
        cpu_index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        cpu_index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if use_gpu:
        try:
            resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(resources, 0, cpu_index)
        except Exception as exc:  # pragma: no cover
            print(f"GPU unavailable, falling back to CPU: {exc}")
    return cpu_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute exact k-NN ground truth with FAISS.")
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--query-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--metric", choices=["ip", "l2"], required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    base, query = load_vectors(args.base_path, args.query_path)
    dim = int(base.shape[1])
    index = build_index(dim=dim, metric=args.metric, use_gpu=args.use_gpu)
    index.add(base)
    _, neighbors = index.search(query, args.k)
    neighbors = neighbors.astype(np.int32)
    np.save(args.output_path, neighbors)

    print("Saved ground truth:")
    print(f"- shape: {neighbors.shape}")
    print(f"- dtype: {neighbors.dtype}")
    print(f"- path:  {args.output_path}")


if __name__ == "__main__":
    main()
