"""
Recompute exact k=100 nearest neighbors using PyTorch on GPU.
Supports both L2 and Inner Product (cosine on unit vectors).
"""
import os
import time
import numpy as np
import torch

def compute_groundtruth_gpu(base_path, query_path, output_path, metric='ip', k=100, batch_size=256):
    print(f"Loading base from {base_path}...")
    base = np.load(base_path)
    queries = np.load(query_path)
    
    print(f"  Base: {base.shape}, Queries: {queries.shape}, Metric: {metric}, k={k}")
    
    device = torch.device('cuda')
    base_t = torch.from_numpy(base).to(device)
    
    all_indices = []
    
    t0 = time.time()
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_t = torch.from_numpy(batch).to(device)
        
        if metric == 'ip':
            # Inner product: higher = closer
            scores = torch.mm(batch_t, base_t.T)
            _, topk = torch.topk(scores, k, dim=1, largest=True)
        else:
            # L2: lower = closer
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            dists = torch.cdist(batch_t, base_t, p=2)
            _, topk = torch.topk(dists, k, dim=1, largest=False)
        
        all_indices.append(topk.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+len(batch)}/{len(queries)} queries ({elapsed:.1f}s)")
    
    gt = np.concatenate(all_indices, axis=0)
    elapsed = time.time() - t0
    print(f"  Ground truth computed in {elapsed:.1f}s, shape: {gt.shape}")
    
    np.save(output_path, gt)
    print(f"  Saved to {output_path}")
    return gt

def spot_check(base_path, query_path, gt_path, metric='ip', n_checks=10):
    """Spot-check ground truth by brute-force on CPU for a few queries."""
    base = np.load(base_path)
    queries = np.load(query_path)
    gt = np.load(gt_path)
    
    print(f"\nSpot-checking {n_checks} queries...")
    for i in range(n_checks):
        q = queries[i]
        if metric == 'ip':
            scores = base @ q
            cpu_top = np.argsort(scores)[::-1][:10]
        else:
            dists = np.linalg.norm(base - q, axis=1)
            cpu_top = np.argsort(dists)[:10]
        
        gpu_top = gt[i, :10]
        overlap = len(set(cpu_top) & set(gpu_top))
        status = "✓" if overlap == 10 else f"MISMATCH ({overlap}/10)"
        print(f"  Query {i}: {status}")

def main():
    os.makedirs('data/sift1m', exist_ok=True)
    os.makedirs('data/msmarco', exist_ok=True)
    
    # SIFT1M: L2 metric
    print("=" * 60)
    print("SIFT1M (L2)")
    print("=" * 60)
    compute_groundtruth_gpu(
        'data/sift1m/base.npy', 'data/sift1m/query.npy',
        'data/sift1m/groundtruth.npy', metric='l2', k=100
    )
    spot_check('data/sift1m/base.npy', 'data/sift1m/query.npy',
               'data/sift1m/groundtruth.npy', metric='l2')
    
    # MS MARCO: Inner Product (cosine on unit vectors)
    print("\n" + "=" * 60)
    print("MS MARCO (Inner Product)")
    print("=" * 60)
    compute_groundtruth_gpu(
        'data/msmarco/base.npy', 'data/msmarco/query.npy',
        'data/msmarco/groundtruth.npy', metric='ip', k=100
    )
    spot_check('data/msmarco/base.npy', 'data/msmarco/query.npy',
               'data/msmarco/groundtruth.npy', metric='ip')

if __name__ == "__main__":
    main()
