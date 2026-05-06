"""
Phase 6: JL Projection Tradeoff
===============================
Tests whether Johnson-Lindenstrauss (JL) projection can compound the memory
advantage of BucketOccupancy by allowing us to store far more items for the
same fixed memory footprint (in bytes/floats), without fatally harming recall.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from baselines import FIFORetention
from retention_policies import BucketOccupancyRetention
from evaluation import run_experiment

RESULTS_DIR = 'results/phase6_jl'
STREAMS_DIR = 'data/streams'

def run_jl_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading queries, ground truth, and HeavyDuplication_50 stream...")
    queries = np.load(os.path.join(STREAMS_DIR, 'queries.npy'))
    oracle_gt = np.load(os.path.join(STREAMS_DIR, 'oracle_groundtruth.npy'))
    
    stream = np.load(os.path.join(STREAMS_DIR, 'stream_dup50.npy'))
    source_ids = np.load(os.path.join(STREAMS_DIR, 'stream_dup50_source_ids.npy'))
    
    N, orig_dim = stream.shape
    print(f"Original stream shape: {stream.shape}")
    
    # Pre-compute clusters for Tier 2 diversity metrics
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from added_metrics import precompute_clusters
    print(f"Precomputing clusters...")
    cluster_assignments = precompute_clusters(stream, n_clusters=20)
    
    # We want to compare across equivalent *memory footprints*.
    # Memory Footprint = Capacity (B) * Dimension (d)
    # So if we compress to d_new, we get to store (orig_dim / d_new) * B items!
    
    budget_fractions = [0.01, 0.05, 0.10, 0.25, 0.50]
    target_dims = [orig_dim, 128, 64]
    
    csv_path = os.path.join(RESULTS_DIR, 'phase6_jl_results.csv')
    all_results = []
    
    if os.path.exists(csv_path):
        print(f"Found existing results at {csv_path}. Loading to resume...")
        df_existing = pd.read_csv(csv_path)
        all_results = df_existing.to_dict('records')
        
    completed_runs = set()
    for row in all_results:
        completed_runs.add((row['policy'], row['budget_fraction'], row['d_new']))

    # Generate JL projections once to reuse
    projections = {orig_dim: (stream, queries)}
    for d_new in target_dims:
        if d_new == orig_dim:
            continue
        print(f"Generating Gaussian Random Projection for d={d_new}...")
        grp = GaussianRandomProjection(n_components=d_new, random_state=42)
        stream_proj = grp.fit_transform(stream).astype(np.float32)
        queries_proj = grp.transform(queries).astype(np.float32)
        
        # L2 normalize the projected vectors for cosine similarity/FAISS Inner Product
        from sklearn.preprocessing import normalize
        stream_proj = normalize(stream_proj, axis=1)
        queries_proj = normalize(queries_proj, axis=1)
        
        projections[d_new] = (stream_proj, queries_proj)

    # Pre-select recommended default parameters for BucketOccupancy
    K_opt = 10
    L_opt = 8

    for frac in budget_fractions:
        # Base capacity B for the uncompressed original dimension
        base_B = max(int(N * frac), 5)
        # Total float budget we are allowed to use
        float_budget = base_B * orig_dim
        
        print(f"\n{'='*60}")
        print(f"Target Memory Budget: {float_budget:,} floats ({frac*100:.0f}% of raw stream)")
        
        for d_new in target_dims:
            # How many items can we fit in this float budget if we compress to d_new?
            capacity_B = int(float_budget / d_new)
            
            stream_proj, queries_proj = projections[d_new]
            
            policies = {}
            if d_new == orig_dim:
                # Baseline comparison: no compression
                policies['FIFO (Baseline, d=384)'] = FIFORetention(dim=orig_dim, capacity=capacity_B)
                policies['BucketOccupancy (Baseline, d=384)'] = BucketOccupancyRetention(dim=orig_dim, L=L_opt, K=K_opt, capacity=capacity_B, aggregator='median')
            else:
                # JL comparison: compressed dimension, but more items
                name = f'BucketOccupancy + JL (d={d_new})'
                policies[name] = BucketOccupancyRetention(dim=d_new, L=L_opt, K=K_opt, capacity=capacity_B, aggregator='median')
            
            for name, policy in policies.items():
                if (name, frac, d_new) in completed_runs:
                    print(f"  Skipping {name} (already computed for frac {frac})")
                    continue
                    
                print(f"\nEvaluating {name} with capacity B={capacity_B:,} items...")
                result, _ = run_experiment(
                    policy=policy, 
                    stream=stream_proj, 
                    source_ids=source_ids,
                    queries=queries_proj, 
                    oracle_gt=oracle_gt, 
                    policy_name=name,
                    cluster_assignments=cluster_assignments,
                    n_clusters=20
                )
                
                # Tag with experiment metadata
                result['budget_fraction'] = frac
                result['d_new'] = d_new
                result['items_capacity'] = capacity_B
                result['memory_footprint_floats'] = float_budget
                
                all_results.append(result)
                
                # Incrementally save results
                df = pd.DataFrame(all_results)
                df.to_csv(csv_path, index=False)

    print(f"\nCompleted JL projection sweep. Results saved to {csv_path}.")

if __name__ == "__main__":
    run_jl_experiment()
