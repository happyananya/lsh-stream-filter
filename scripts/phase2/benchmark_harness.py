#!/usr/bin/env python3
"""
Shared benchmark harness for Phase 2+ ANN experiments.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


QueryFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class BenchmarkResult:
    recall: float
    qps: float
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    n_queries: int
    warmup: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "recall": self.recall,
            "qps": self.qps,
            "p50_us": self.p50_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
            "mean_us": self.mean_us,
            "n_queries": self.n_queries,
            "warmup": self.warmup,
        }


def enforce_single_thread() -> None:
    # Must be set before importing heavy numerical libraries in real runs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def recall_at_k(pred: np.ndarray, gt_row: np.ndarray, k: int) -> float:
    pred_set = set(pred[:k].tolist())
    gt_set = set(gt_row[:k].tolist())
    return len(pred_set.intersection(gt_set)) / float(k)


def benchmark(
    query_fn: QueryFn,
    queries: np.ndarray,
    groundtruth: np.ndarray,
    k: int = 10,
    warmup: int = 1000,
) -> BenchmarkResult:
    if len(queries) != len(groundtruth):
        raise ValueError("queries and groundtruth must have same length")

    warmup_n = min(warmup, len(queries))
    for q in queries[:warmup_n]:
        _ = query_fn(q, k)

    latencies_ns: list[int] = []
    recalls: list[float] = []

    for q, gt_row in zip(queries, groundtruth):
        t0 = time.perf_counter_ns()
        pred = query_fn(q, k)
        latencies_ns.append(time.perf_counter_ns() - t0)
        recalls.append(recall_at_k(np.asarray(pred), gt_row, k=k))

    latency_arr = np.asarray(latencies_ns, dtype=np.float64)
    mean_ns = float(np.mean(latency_arr))
    qps = float(1e9 / mean_ns) if mean_ns > 0 else 0.0

    return BenchmarkResult(
        recall=float(np.mean(recalls)),
        qps=qps,
        p50_us=float(np.percentile(latency_arr, 50) / 1000.0),
        p95_us=float(np.percentile(latency_arr, 95) / 1000.0),
        p99_us=float(np.percentile(latency_arr, 99) / 1000.0),
        mean_us=float(mean_ns / 1000.0),
        n_queries=int(len(queries)),
        warmup=int(warmup_n),
    )

