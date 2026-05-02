"""
Phase 1: Prepare streaming datasets for the LSH Memory Retention project.

Creates two controlled streams from the existing MS MARCO embeddings:
  1. Ordered stream with configurable duplicate injection
  2. Drift-injected stream (ordered by topic clusters)

Also generates held-out query sets and oracle ground truth.

Usage:
    python scripts/prepare_streams.py
"""
import os
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def inject_duplicates(base, rate, seed=42):
    """
    Inject duplicates into the dataset at a given rate.
    rate=0.1 means 10% of the output will be duplicates of existing items.
    Returns (augmented_array, duplicate_mask) where duplicate_mask[i]=True if item i is a duplicate.
    """
    rng = np.random.RandomState(seed)
    n = len(base)
    n_dupes = int(n * rate / (1.0 - rate))  # so final size has `rate` fraction of dupes
    
    dupe_source_idx = rng.choice(n, size=n_dupes, replace=True)
    dupes = base[dupe_source_idx]
    
    # Track original source indices
    original_ids = np.arange(n)
    combined_ids = np.concatenate([original_ids, dupe_source_idx])
    
    # Interleave originals and duplicates
    combined = np.concatenate([base, dupes], axis=0)
    is_duplicate = np.concatenate([np.zeros(n, dtype=bool), np.ones(n_dupes, dtype=bool)])
    
    # Shuffle to interleave
    perm = rng.permutation(len(combined))
    combined = combined[perm]
    combined_ids = combined_ids[perm]
    is_duplicate = is_duplicate[perm]
    
    return combined, is_duplicate, combined_ids

def create_drift_stream(base, n_clusters=20, seed=42):
    """
    Reorder the dataset by topic clusters to simulate distribution drift.
    Items from cluster 0 come first, then cluster 1, etc.
    Returns (reordered_array, cluster_labels, cluster_boundaries).
    """
    print(f"  Clustering {len(base)} vectors into {n_clusters} topic clusters...")
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=seed, n_init=3)
    labels = km.fit_predict(base)
    
    # Sort by cluster label
    order = np.argsort(labels, kind='stable')
    reordered = base[order]
    sorted_labels = labels[order]
    
    # Find cluster boundaries
    boundaries = []
    for c in range(n_clusters):
        idx = np.where(sorted_labels == c)[0]
        if len(idx) > 0:
            boundaries.append((c, idx[0], idx[-1]))
    
    return reordered, sorted_labels, boundaries, order

def create_query_set(base, n_queries=5000, seed=42):
    """
    Sample a held-out query set from the base vectors.
    Returns (queries, query_source_indices).
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n_queries, replace=False)
    return base[idx].copy(), idx

def compute_oracle_groundtruth_gpu(base, queries, metric='ip', k=100, batch_size=256):
    """
    Compute exact k-NN ground truth using PyTorch on GPU.
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Computing oracle GT on {device} ({len(queries)} queries vs {len(base)} base)...")
    
    base_t = torch.from_numpy(base).to(device)
    all_indices = []
    
    t0 = time.time()
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_t = torch.from_numpy(batch).to(device)
        
        if metric == 'ip':
            scores = torch.mm(batch_t, base_t.T)
            _, topk = torch.topk(scores, k, dim=1, largest=True)
        else:
            dists = torch.cdist(batch_t, base_t, p=2)
            _, topk = torch.topk(dists, k, dim=1, largest=False)
        
        all_indices.append(topk.cpu().numpy())
    
    gt = np.concatenate(all_indices, axis=0)
    print(f"  Oracle GT computed in {time.time() - t0:.1f}s, shape: {gt.shape}")
    return gt

def main():
    OUT = 'data/streams'
    os.makedirs(OUT, exist_ok=True)
    
    # Load existing MS MARCO embeddings (already L2-normalized, d=384)
    print("Loading MS MARCO base embeddings...")
    base = np.load('data/msmarco/base.npy')
    print(f"  Shape: {base.shape}, dtype: {base.dtype}")
    
    # Verify L2-normalization
    norms = np.linalg.norm(base[:100], axis=1)
    if not np.allclose(norms, 1.0, atol=0.01):
        print("  WARNING: base vectors not L2-normalized. Normalizing now...")
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
    else:
        print(f"  L2-normalized: ✓ (sample norms: {norms[:5].round(4)})")
    
    # === 1. Create held-out query set (from the full base, BEFORE any filtering) ===
    print("\n=== Creating held-out query set ===")
    queries, query_src_idx = create_query_set(base, n_queries=5000, seed=42)
    np.save(os.path.join(OUT, 'queries.npy'), queries)
    np.save(os.path.join(OUT, 'query_source_idx.npy'), query_src_idx)
    print(f"  Saved {len(queries)} queries")
    
    # === 2. Oracle ground truth (against the FULL unfiltered set) ===
    print("\n=== Computing oracle ground truth ===")
    oracle_gt = compute_oracle_groundtruth_gpu(base, queries, metric='ip', k=100)
    np.save(os.path.join(OUT, 'oracle_groundtruth.npy'), oracle_gt)
    
    # === 3. Clean stream (no duplicates, random order) ===
    print("\n=== Creating clean stream (random order, 0% duplicates) ===")
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(base))
    clean_stream = base[perm]
    np.save(os.path.join(OUT, 'stream_clean.npy'), clean_stream)
    np.save(os.path.join(OUT, 'stream_clean_source_ids.npy'), perm)
    print(f"  Saved stream_clean.npy: {clean_stream.shape}")
    
    # === 4. Duplicate-injected streams ===
    for dup_rate in [0.10, 0.30, 0.50]:
        print(f"\n=== Creating stream with {int(dup_rate*100)}% duplicates ===")
        stream, dup_mask, source_ids = inject_duplicates(base, rate=dup_rate, seed=42)
        tag = f"dup{int(dup_rate*100)}"
        np.save(os.path.join(OUT, f'stream_{tag}.npy'), stream)
        np.save(os.path.join(OUT, f'stream_{tag}_mask.npy'), dup_mask)
        np.save(os.path.join(OUT, f'stream_{tag}_source_ids.npy'), source_ids)
        print(f"  Saved stream_{tag}.npy: {stream.shape} ({dup_mask.sum()} dupes)")
    
    # === 5. Drift-injected stream ===
    print("\n=== Creating drift-injected stream ===")
    drift_stream, drift_labels, drift_bounds, drift_source_ids = create_drift_stream(base, n_clusters=20, seed=42)
    np.save(os.path.join(OUT, 'stream_drift.npy'), drift_stream)
    np.save(os.path.join(OUT, 'stream_drift_labels.npy'), drift_labels)
    np.save(os.path.join(OUT, 'stream_drift_source_ids.npy'), drift_source_ids)
    print(f"  Saved stream_drift.npy: {drift_stream.shape}")
    print(f"  Cluster boundaries:")
    for c, start, end in drift_bounds:
        print(f"    Cluster {c:2d}: [{start:7d}, {end:7d}] ({end-start+1:6d} items)")
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("Phase 1 dataset preparation complete!")
    print("=" * 60)
    print(f"Output directory: {OUT}/")
    print(f"Files created:")
    for f in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f:40s} {size/1024/1024:8.1f} MB")

if __name__ == "__main__":
    main()
