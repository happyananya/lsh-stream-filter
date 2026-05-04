"""
Phase 2: Characterization Experiments (Research Plan §3.6)
============================================================
Runs the five characterization experiments that define the contribution:

1. Steady-state memory bound — |M| over time for various (K, L, T)
2. Bucket-occupancy distribution — histograms at t=100K, 500K, 1M
3. Retention rate vs. duplicate rate (0%, 10%, 30%, 50%)
4. Retention rate vs. K — for fixed L=8, sweep K=6..14
5. Distribution drift response — retention rate vs. stream position

All outputs go to results/phase2_characterization/
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from retention_policies import BucketOccupancyRetention

RESULTS_DIR = 'results/phase2_characterization'
STREAMS_DIR = 'data/streams'


# ──────────────────────────────────────────────────────────────
# Experiment 1: Steady-state memory bound
# ──────────────────────────────────────────────────────────────
def exp1_steady_state(stream, configs, sample_points=None):
    """
    Stream N items, record |M| at regular intervals for each (K, L, T) config.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Steady-State Memory Bound")
    print("=" * 60)
    
    N = len(stream)
    if sample_points is None:
        sample_points = list(range(0, N, 10000)) + [N - 1]
    
    all_rows = []
    
    for cfg in configs:
        label = f"K={cfg['K']}, L={cfg['L']}, cap={cfg.get('capacity', 'inf')}"
        agg = cfg.get('aggregator', 'median')
        thresh = cfg.get('threshold', None)
        print(f"\n  Config: {label}, agg={agg}, T={thresh}")
        
        policy = BucketOccupancyRetention(
            dim=stream.shape[1],
            L=cfg['L'], K=cfg['K'],
            capacity=cfg.get('capacity', N),
            aggregator=agg,
            threshold=thresh,
            seed=42,
        )
        
        t0 = time.time()
        for i in range(N):
            policy.insert(stream[i], item_id=i)
            
            if i in sample_points or i == N - 1:
                all_rows.append({
                    'config': label,
                    'K': cfg['K'], 'L': cfg['L'],
                    'threshold': thresh,
                    'capacity': cfg.get('capacity', N),
                    'aggregator': agg,
                    't': i + 1,
                    'kept': policy.kept_count(),
                    'retention_rate': policy.kept_count() / (i + 1),
                })
        
        elapsed = time.time() - t0
        stats = policy.stats()
        print(f"    Done in {elapsed:.1f}s | Final kept: {stats['kept']:,} | "
              f"Active buckets: {stats['active_buckets']:,}")
    
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp1_steady_state.csv'), index=False)
    print(f"\n  Saved exp1_steady_state.csv ({len(df)} rows)")
    return df


# ──────────────────────────────────────────────────────────────
# Experiment 2: Bucket-occupancy distributions
# ──────────────────────────────────────────────────────────────
def exp2_bucket_distributions(stream, K=10, L=8, capacity=100000,
                               checkpoints=[100000, 500000, 1000000]):
    """
    Run the policy and snapshot bucket-occupancy histograms at given checkpoints.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Bucket-Occupancy Distributions")
    print("=" * 60)
    
    policy = BucketOccupancyRetention(
        dim=stream.shape[1], L=L, K=K,
        capacity=capacity, aggregator='median', seed=42,
    )
    
    N = min(len(stream), max(checkpoints))
    histograms = {}
    
    for i in range(N):
        policy.insert(stream[i], item_id=i)
        
        if (i + 1) in checkpoints:
            hist = policy.bucket_occupancy_histogram()
            histograms[i + 1] = hist
            print(f"  t={i+1:,}: {len(hist)} active buckets, "
                  f"mean={hist.mean():.1f}, max={hist.max()}, "
                  f"kept={policy.kept_count():,}")
    
    # Save histograms as separate files
    for t, hist in histograms.items():
        np.save(os.path.join(RESULTS_DIR, f'exp2_bucket_hist_t{t}.npy'), hist)
    
    print(f"  Saved {len(histograms)} histogram snapshots")
    return histograms


# ──────────────────────────────────────────────────────────────
# Experiment 3: Retention rate vs. duplicate rate
# ──────────────────────────────────────────────────────────────
def exp3_duplicate_sweep(K=10, L=8, capacity=100000):
    """
    Run the policy on streams with 0%, 10%, 30%, 50% duplicates.
    Measure what fraction of duplicates are correctly discarded.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Retention Rate vs. Duplicate Rate")
    print("=" * 60)
    
    streams = {
        '0%': ('stream_clean.npy', None),
        '10%': ('stream_dup10.npy', 'stream_dup10_mask.npy'),
        '30%': ('stream_dup30.npy', 'stream_dup30_mask.npy'),
        '50%': ('stream_dup50.npy', 'stream_dup50_mask.npy'),
    }
    
    rows = []
    
    for dup_label, (stream_file, mask_file) in streams.items():
        print(f"\n  Stream: {dup_label} duplicates")
        stream = np.load(os.path.join(STREAMS_DIR, stream_file))
        dup_mask = np.load(os.path.join(STREAMS_DIR, mask_file)) if mask_file else np.zeros(len(stream), dtype=bool)
        
        dim = stream.shape[1]
        
        policy = BucketOccupancyRetention(
            dim=dim, L=L, K=K,
            capacity=capacity, aggregator='median', seed=42,
        )
        
        n_dupes_kept = 0
        n_dupes_total = 0
        n_originals_kept = 0
        n_originals_total = 0
        
        t0 = time.time()
        for i in range(len(stream)):
            kept = policy.insert(stream[i], item_id=i)
            
            if dup_mask[i]:
                n_dupes_total += 1
                if kept:
                    n_dupes_kept += 1
            else:
                n_originals_total += 1
                if kept:
                    n_originals_kept += 1
        
        elapsed = time.time() - t0
        stats = policy.stats()
        
        if n_dupes_total == 0:
            dup_discard_rate = float('nan')
        else:
            dup_discard_rate = 1.0 - (n_dupes_kept / n_dupes_total)
            
        orig_keep_rate = n_originals_kept / max(n_originals_total, 1)
        
        print(f"    Time: {elapsed:.1f}s | Kept: {stats['kept']:,}")
        print(f"    Duplicates: {n_dupes_kept}/{n_dupes_total} kept (discard rate: {dup_discard_rate:.2%})")
        print(f"    Originals: {n_originals_kept}/{n_originals_total} kept (keep rate: {orig_keep_rate:.2%})")
        
        rows.append({
            'dup_rate': dup_label,
            'stream_size': len(stream),
            'n_dupes_total': n_dupes_total,
            'n_dupes_kept': n_dupes_kept,
            'dup_discard_rate': dup_discard_rate,
            'n_originals_total': n_originals_total,
            'n_originals_kept': n_originals_kept,
            'orig_keep_rate': orig_keep_rate,
            'final_kept': stats['kept'],
            'throughput': len(stream) / elapsed,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp3_duplicate_sweep.csv'), index=False)
    print(f"\n  Saved exp3_duplicate_sweep.csv")
    return df


# ──────────────────────────────────────────────────────────────
# Experiment 4: Retention rate vs. K
# ──────────────────────────────────────────────────────────────
def exp4_k_sweep(stream, queries, oracle_gt, L=8, capacity=100000):
    """
    For fixed L, sweep K from 6 to 14.
    Report (kept-set size, recall@10 on held-out queries).
    """
    from evaluation import evaluate_recall, run_experiment
    
    source_ids = np.load(os.path.join(STREAMS_DIR, 'stream_clean_source_ids.npy'))
    
    print("\n" + "=" * 60)
    print("Experiment 4: Recall vs Hash Granularity (K Sweep)")
    print("=" * 60)
    
    rows = []
    
    for K in [6, 8, 10, 12, 14]:
        print(f"\nEvaluating K={K} (L=8)...")
        policy = BucketOccupancyRetention(
            dim=stream.shape[1], L=L, K=K,
            capacity=capacity, aggregator='median', seed=42,
        )
        
        result, _ = run_experiment(
            policy=policy,
            stream=stream,
            source_ids=source_ids,
            queries=queries,
            oracle_gt=oracle_gt,
            policy_name=f"K={K}, L=8"
        )
        
        rows.append({
            'K': K,
            'L': L,
            'n_buckets_per_table': 2**K,
            'capacity': capacity,
            'kept': result['kept'],
            'retention_rate': result['retention_rate'],
            'recall_at_10': result.get('recall@10_mean', 0),
            'recall_p10': result.get('recall@10_p10', 0),
            'recall_p50': result.get('recall@10_p50', 0),
            'recall_p90': result.get('recall@10_p90', 0),
            'throughput': result['throughput_items_per_sec'],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp4_k_sweep.csv'), index=False)
    print(f"\n  Saved exp4_k_sweep.csv")
    return df


# ──────────────────────────────────────────────────────────────
# Experiment 5: Distribution drift response
# ──────────────────────────────────────────────────────────────
def exp5_drift_response(K=10, L=8, capacity=100000, window=50000):
    """
    Run the policy on the drift-injected stream.
    Plot retention rate as a windowed function of stream position.
    """
    print("\n" + "=" * 60)
    print("Experiment 5: Distribution Drift Response")
    print("=" * 60)
    
    stream = np.load(os.path.join(STREAMS_DIR, 'stream_drift.npy'))
    labels = np.load(os.path.join(STREAMS_DIR, 'stream_drift_labels.npy'))
    
    policy = BucketOccupancyRetention(
        dim=stream.shape[1], L=L, K=K,
        capacity=capacity, aggregator='median', seed=42,
    )
    
    # Track per-item keep/discard decisions
    decisions = np.zeros(len(stream), dtype=bool)
    
    t0 = time.time()
    for i in range(len(stream)):
        decisions[i] = policy.insert(stream[i], item_id=i)
    elapsed = time.time() - t0
    
    # Compute windowed retention rate
    rows = []
    for start in range(0, len(stream), window // 2):  # 50% overlap
        end = min(start + window, len(stream))
        win_decisions = decisions[start:end]
        win_labels = labels[start:end]
        
        rows.append({
            'window_start': start,
            'window_end': end,
            'window_retention_rate': win_decisions.mean(),
            'window_kept': int(win_decisions.sum()),
            'dominant_cluster': int(np.bincount(win_labels).argmax()),
            'n_unique_clusters': len(np.unique(win_labels)),
        })
    
    print(f"  Done in {elapsed:.1f}s | Final kept: {policy.kept_count():,}")
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp5_drift_response.csv'), index=False)
    print(f"  Saved exp5_drift_response.csv ({len(df)} rows)")
    return df


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load the clean stream + queries + oracle GT
    print("Loading datasets...")
    clean_stream = np.load(os.path.join(STREAMS_DIR, 'stream_clean.npy'))
    queries = np.load(os.path.join(STREAMS_DIR, 'queries.npy'))
    oracle_gt = np.load(os.path.join(STREAMS_DIR, 'oracle_groundtruth.npy'))
    print(f"  Clean stream: {clean_stream.shape}")
    print(f"  Queries: {queries.shape}")
    print(f"  Oracle GT: {oracle_gt.shape}")
    
    # --- Experiment 1: Steady-state memory bound ---
    configs = [
        {'K': 8,  'L': 8, 'capacity': 100000, 'aggregator': 'median'},
        {'K': 10, 'L': 8, 'capacity': 100000, 'aggregator': 'median'},
        {'K': 12, 'L': 8, 'capacity': 100000, 'aggregator': 'median'},
        {'K': 10, 'L': 4, 'capacity': 100000, 'aggregator': 'median'},
        {'K': 10, 'L': 16, 'capacity': 100000, 'aggregator': 'median'},
        {'K': 10, 'L': 8, 'capacity': 100000, 'aggregator': 'min'},
        {'K': 10, 'L': 8, 'capacity': 100000, 'aggregator': 'max'},
    ]
    exp1_steady_state(clean_stream, configs)
    
    # --- Experiment 2: Bucket-occupancy distributions ---
    exp2_bucket_distributions(clean_stream, K=10, L=8, capacity=100000)
    
    # --- Experiment 3: Retention rate vs. duplicate rate ---
    exp3_duplicate_sweep(K=10, L=8, capacity=100000)
    
    # --- Experiment 4: Retention rate vs. K ---
    exp4_k_sweep(clean_stream, queries, oracle_gt, L=8, capacity=100000)
    
    # --- Experiment 5: Drift response ---
    exp5_drift_response(K=10, L=8, capacity=100000)
    
    print("\n" + "=" * 60)
    print("All Phase 2 characterization experiments complete!")
    print(f"Results in: {RESULTS_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
