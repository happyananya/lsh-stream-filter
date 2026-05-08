"""
Run Phase 4 Evaluation with Cosine Metrics
==========================================
Re-runs the evaluation on HeavyDuplication_50 to capture the newly
added Tier 2b Cosine diversity metrics, saving to a new CSV.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from baselines import FIFORetention, ReservoirSamplingRetention
from retention_policies import BucketOccupancyRetention
from evaluation import run_experiment

RESULTS_DIR = 'results/phase4_cosine'
STREAMS_DIR = 'data/streams'

def run_cosine_sweep():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stream_name = "HeavyDuplication_50"
    print(f"\n{'#'*70}")
    print(f"Running Cosine Metrics Sweep on: {stream_name}")
    print(f"{'#'*70}")
    
    stream = np.load(os.path.join(STREAMS_DIR, 'stream_dup50.npy'))
    source_ids = np.load(os.path.join(STREAMS_DIR, 'stream_dup50_source_ids.npy'))
    queries = np.load(os.path.join(STREAMS_DIR, 'queries.npy'))
    oracle_gt = np.load(os.path.join(STREAMS_DIR, 'oracle_groundtruth.npy'))
    N, dim = stream.shape
    
    # Pre-compute clusters for Tier 2 diversity metrics
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from added_metrics import precompute_clusters
    print(f"  Precomputing clusters...")
    cluster_assignments = precompute_clusters(stream, n_clusters=20)
    
    csv_path = os.path.join(RESULTS_DIR, f'cosine_sweep_{stream_name}.csv')
    all_results = []
    
    if os.path.exists(csv_path):
        print(f"  Found existing results at {csv_path}. Loading to resume...")
        df_existing = pd.read_csv(csv_path)
        all_results = df_existing.to_dict('records')
    
    completed_runs = set()
    for row in all_results:
        completed_runs.add((row['policy'], row['budget_fraction']))
    
    budget_fractions = [0.01, 0.05, 0.10, 0.25, 0.50]
    K_opt = 10
    L_opt = 8
    
    for frac in budget_fractions:
        B = int(N * frac)
        print(f"\n>>> Evaluating Memory Budget: B = {B:,} ({frac*100:.0f}% of N)")
        
        policies = {
            'FIFO': FIFORetention(dim=dim, capacity=B),
            'Reservoir': ReservoirSamplingRetention(dim=dim, capacity=B),
            'BucketOccupancy': BucketOccupancyRetention(dim=dim, L=L_opt, K=K_opt, capacity=B, aggregator='median'),
        }
        
        for name, policy in policies.items():
            if (name, frac) in completed_runs:
                print(f"  Skipping {name} (already computed for frac {frac})")
                continue
                
            result, _ = run_experiment(
                policy=policy, 
                stream=stream, 
                source_ids=source_ids,
                queries=queries, 
                oracle_gt=oracle_gt, 
                policy_name=name,
                cluster_assignments=cluster_assignments,
                n_clusters=20
            )
            
            result['stream'] = stream_name
            result['budget_fraction'] = frac
            result['budget_B'] = B
            
            all_results.append(result)
            
            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)

    print(f"\nCompleted Cosine metrics sweep. Results saved to {csv_path}.")

if __name__ == "__main__":
    run_cosine_sweep()
