"""
Phase 2: Falconn++ Reproduction Benchmark (BATCH MODE)
=======================================================
Uses FalconnPP's native batch query API for realistic throughput measurement.

The official API: index.query(queries_t, k) takes ALL queries as a D×Q matrix
and returns a (Q, k) result array in a single C++ call. This is how the paper
benchmarks it — calling one query at a time through Python would be ~1000x slower.
"""
import os
import time
import numpy as np
import pandas as pd
import FalconnPP

def compute_recall(predictions, ground_truth, k=10):
    """Compute mean recall@k across all queries."""
    n_queries = len(predictions)
    recalls = []
    for i in range(n_queries):
        pred_set = set(predictions[i, :k])
        gt_set = set(ground_truth[i, :k])
        recalls.append(len(pred_set & gt_set) / k)
    return np.mean(recalls)

def run_falconnpp_sweep(dataset_name, base, queries, gt, csv_path):
    n_points, n_features = base.shape
    n_queries = len(queries)
    k = 10
    
    # Resume logic: load already-completed configs
    header_cols = ['Method','Parameters','Dataset','Build_Time_s','Index_Size_MB',
                   'n_tables','n_proj','iProbes','qProbes',
                   'recall','qps','p50_us','p95_us','p99_us','dist_err_ratio']
    completed = set()
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            for _, row in existing.iterrows():
                key = (int(row['n_tables']), int(row['n_proj']), int(row['iProbes']), int(row['qProbes']))
                completed.add(key)
            print(f"  Resuming: {len(completed)} configs already completed in {csv_path}")
        except Exception:
            completed = set()
    
    if not completed:
        with open(csv_path, 'w') as f:
            f.write(','.join(header_cols) + '\n')
    
    # FalconnPP requires D x N (transposed) float32
    base_t = np.ascontiguousarray(base.T, dtype=np.float32)
    queries_t = np.ascontiguousarray(queries.T, dtype=np.float32)
    
    # Parameter grid
    n_tables_list = [50, 100, 200]
    n_proj_list = [128, 256]
    iProbes_list = [10, 40]
    qProbes_list = [100, 500, 1000, 2000, 4000]
    
    # Fixed parameters
    bucketLimit = 50
    alpha = 0.01
    n_threads = 1  # Single-threaded for fair comparison
    
    # Sample size for per-query latency profiling (keeps it fast)
    LATENCY_SAMPLE = 200
    
    for n_tables in n_tables_list:
        for n_proj in n_proj_list:
            for iProbes in iProbes_list:
                # Check if ALL qProbes for this build config are done
                all_done = all((n_tables, n_proj, iProbes, qp) in completed for qp in qProbes_list)
                if all_done:
                    print(f"\n  SKIP L={n_tables}, D={n_proj}, iProbes={iProbes} (all qProbes done)")
                    continue
                
                print(f"\n  Building Falconn++ L={n_tables}, D={n_proj}, iProbes={iProbes}...")
                
                index = FalconnPP.FalconnPP(n_points, n_features)
                index.setIndexParam(n_tables, n_proj, bucketLimit, alpha, iProbes, n_threads)
                
                t0 = time.time()
                index.build(base_t)
                build_time = time.time() - t0
                print(f"    Built in {build_time:.2f}s")
                
                # Measure index size once per build config
                index_size_mb = np.nan
                try:
                    import tempfile, psutil
                    proc = psutil.Process(os.getpid())
                    index_size_mb = proc.memory_info().rss / (1024 * 1024)
                except ImportError:
                    pass
                
                for qProbes in qProbes_list:
                    if (n_tables, n_proj, iProbes, qProbes) in completed:
                        print(f"    SKIP qP={qProbes} (already done)")
                        continue
                    
                    index.set_qProbes(qProbes)
                    
                    # Warmup (single batch call)
                    _ = index.query(queries_t, k)
                    
                    # === 1. Batch query for QPS (fast, accurate throughput) ===
                    t0 = time.perf_counter()
                    results = index.query(queries_t, k)
                    elapsed = time.perf_counter() - t0
                    
                    qps = n_queries / elapsed
                    recall = compute_recall(results, gt, k)
                    
                    # === 2. Per-query sampling for latency percentiles ===
                    sample_idx = np.random.choice(n_queries, min(LATENCY_SAMPLE, n_queries), replace=False)
                    latencies = []
                    for si in sample_idx:
                        q_t = queries_t[:, si:si+1].copy()
                        t0 = time.perf_counter_ns()
                        _ = index.query(q_t, k)
                        latencies.append(time.perf_counter_ns() - t0)
                    
                    latencies = np.array(latencies)
                    p50_us = np.percentile(latencies, 50) / 1000
                    p95_us = np.percentile(latencies, 95) / 1000
                    p99_us = np.percentile(latencies, 99) / 1000
                    
                    # === 3. Distance error ratio from batch results ===
                    dist_ratios = []
                    for i in range(n_queries):
                        pred_id = results[i, 0]
                        gt_id = gt[i, 0]
                        d_pred = 1.0 - np.dot(queries[i], base[pred_id])
                        d_gt = 1.0 - np.dot(queries[i], base[gt_id])
                        if d_gt < 1e-6:
                            dist_ratios.append(1.0 if d_pred < 1e-6 else 10.0)
                        else:
                            dist_ratios.append(d_pred / d_gt)
                    dist_err_ratio = np.mean(dist_ratios)
                    
                    print(f"    qP={qProbes:5d} | Recall={recall:.4f} | QPS={qps:.0f} | p50={p50_us:.0f}µs | distErr={dist_err_ratio:.4f}")
                    
                    # Append row to CSV immediately (crash-safe)
                    row = {
                        'Method': 'Falconn++',
                        'Parameters': f'"L={n_tables}, D={n_proj}, iP={iProbes}, qP={qProbes}"',
                        'Dataset': dataset_name,
                        'Build_Time_s': build_time,
                        'Index_Size_MB': index_size_mb,
                        'n_tables': n_tables,
                        'n_proj': n_proj,
                        'iProbes': iProbes,
                        'qProbes': qProbes,
                        'recall': recall,
                        'qps': qps,
                        'p50_us': p50_us,
                        'p95_us': p95_us,
                        'p99_us': p99_us,
                        'dist_err_ratio': dist_err_ratio,
                    }
                    pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
                
                del index

def main():
    os.makedirs('results', exist_ok=True)
    
    # --- MS MARCO ---
    print("=" * 60)
    print("MS MARCO (Inner Product)")
    print("=" * 60)
    base = np.load('data/msmarco/base.npy')
    queries = np.load('data/msmarco/query.npy')
    gt = np.load('data/msmarco/groundtruth.npy')
    
    run_falconnpp_sweep('MSMARCO', base, queries, gt, 'results/phase2_falconnpp_msmarco.csv')
    print(f"\nDone MS MARCO.")
    
    # --- SIFT1M ---
    print("\n" + "=" * 60)
    print("SIFT1M (L2-normalized -> Inner Product)")
    print("=" * 60)
    base = np.load('data/sift1m/base.npy')
    queries = np.load('data/sift1m/query.npy')
    gt = np.load('data/sift1m/groundtruth.npy')
    
    # L2-normalize for angular distance
    base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
    queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
    
    run_falconnpp_sweep('SIFT1M', base, queries, gt, 'results/phase2_falconnpp_sift1m.csv')
    print(f"\nDone SIFT1M.")

if __name__ == "__main__":
    main()
