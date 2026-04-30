import time
import numpy as np

def benchmark(index_query_fn, queries, gt, base, metric='l2', k=10, warmup=1000):
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
    dist_ratios = []
    
    for q, g in zip(queries, gt):
        t0 = time.perf_counter_ns()
        pred = index_query_fn(q, k)
        latencies.append(time.perf_counter_ns() - t0)
        recalls.append(len(set(pred) & set(g[:k])) / k)
        
        if len(pred) > 0:
            pred_vec = base[pred[0]]
            gt_vec = base[g[0]]
            if metric == 'l2':
                d_pred = np.linalg.norm(q - pred_vec)
                d_gt = np.linalg.norm(q - gt_vec)
            else: # 'ip'
                # 1.0 - cosine similarity
                d_pred = 1.0 - np.dot(q, pred_vec)
                d_gt = 1.0 - np.dot(q, gt_vec)
            
            if d_gt < 1e-6:
                dist_ratios.append(1.0 if d_pred < 1e-6 else 10.0) # Penalty for missing exact match when dist is 0
            else:
                dist_ratios.append(d_pred / d_gt)
        
    return {
        'recall': np.mean(recalls),
        'qps': 1e9 / np.mean(latencies),
        'p50_us': np.percentile(latencies, 50) / 1000,
        'p95_us': np.percentile(latencies, 95) / 1000,
        'p99_us': np.percentile(latencies, 99) / 1000,
        'dist_err_ratio': np.mean(dist_ratios) if dist_ratios else float('nan'),
    }
