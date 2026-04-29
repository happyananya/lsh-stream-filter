import time
import numpy as np

def benchmark(index_query_fn, queries, gt, k=10, warmup=1000):
    """
    Standardized benchmark harness for evaluating ANN algorithms.
    Runs warmup queries, measures per-query latency using high-resolution timers,
    and computes recall@k against the exact nearest neighbors ground truth.
    """
    # Warmup to settle CPU caches and JIT
    for q in queries[:warmup]:
        _ = index_query_fn(q, k)
        
    # Measure
    latencies = []
    recalls = []
    for q, g in zip(queries, gt):
        t0 = time.perf_counter_ns()
        pred = index_query_fn(q, k)
        latencies.append(time.perf_counter_ns() - t0)
        recalls.append(len(set(pred) & set(g[:k])) / k)
        
    return {
        'recall': np.mean(recalls),
        'qps': 1e9 / np.mean(latencies),
        'p50_us': np.percentile(latencies, 50) / 1000,
        'p95_us': np.percentile(latencies, 95) / 1000,
        'p99_us': np.percentile(latencies, 99) / 1000,
    }
