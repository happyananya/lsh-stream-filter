"""
Phase 4: Framing 1 Experiments (Recall under bounded memory)
============================================================
Headline experiments evaluating retrieval recall@10 for all baseline policies
across various memory budgets (B/N).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from baselines import FIFORetention, ReservoirSamplingRetention, SemanticDedupRetention, StreamLSHRetention
from retention_policies import BucketOccupancyRetention
from evaluation import run_experiment

RESULTS_DIR = 'results/phase4_framing1'
STREAMS_DIR = 'data/streams'


def run_budget_sweep(stream_name, stream_file, source_ids_file, queries, oracle_gt, 
                     budget_fractions=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]):
    """
    Run all policies on a given stream across different memory budgets.
    """
    print(f"\n{'#'*70}")
    print(f"Running Phase 4 Sweep on: {stream_name}")
    print(f"B/N fractions: {budget_fractions}")
    print(f"{'#'*70}")
    
    stream = np.load(os.path.join(STREAMS_DIR, stream_file))
    source_ids = np.load(os.path.join(STREAMS_DIR, source_ids_file))
    N, dim = stream.shape
    
    # Pre-select recommended default parameters for BucketOccupancy
    K_opt = 10
    L_opt = 8
    csv_path = os.path.join(RESULTS_DIR, f'phase4_sweep_{stream_name}.csv')
    all_results = []
    
    # Load existing results to resume if interrupted
    if os.path.exists(csv_path):
        print(f"  Found existing results at {csv_path}. Loading to resume...")
        df_existing = pd.read_csv(csv_path)
        all_results = df_existing.to_dict('records')
    
    # Track which (policy, fraction) combinations are already done
    completed_runs = set()
    for row in all_results:
        completed_runs.add((row['policy'], row['budget_fraction']))
    
    for frac in budget_fractions:
        B = int(N * frac)
        print(f"\n>>> Evaluating Memory Budget: B = {B:,} ({frac*100:.0f}% of N)")
        
        # Instantiate policies
        policies = {
            'FIFO': FIFORetention(dim=dim, capacity=B),
            'Reservoir': ReservoirSamplingRetention(dim=dim, capacity=B),
            'Stream-LSH': StreamLSHRetention(dim=dim, capacity=B),
            'BucketOccupancy': BucketOccupancyRetention(dim=dim, L=L_opt, K=K_opt, capacity=B, aggregator='median'),
        }
        
        # Semantic Dedup is fundamentally O(|M|) per insertion because it does an exact 
        # nearest-neighbor search for every incoming item. Even with FAISS, it scales linearly.
        # We restrict it to frac <= 0.25 (budget <= 500,000) to get a good curve for the paper,
        # but be aware it will take ~10-15 mins per stream at frac=0.25.
        if frac <= 0.25:
            policies['SemanticDedup (eps=0.1)'] = SemanticDedupRetention(dim=dim, capacity=B, epsilon=0.1)
        
        # At B=1.00, we don't need to run Reservoir/FIFO since they just keep everything
        if frac == 1.00:
            policies = {
                'BucketOccupancy': BucketOccupancyRetention(dim=dim, L=L_opt, K=K_opt, capacity=B, aggregator='median'),
                'FIFO (Oracle Upper Bound)': FIFORetention(dim=dim, capacity=B)
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
                k=10, 
                metric='ip'
            )
            
            # Tag with experiment metadata
            result['stream'] = stream_name
            result['budget_fraction'] = frac
            result['budget_B'] = B
            
            all_results.append(result)
            
            # Incrementally save results
            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)

    print(f"\nCompleted sweep for {stream_name}. Results saved.")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading queries and ground truth...")
    queries = np.load(os.path.join(STREAMS_DIR, 'queries.npy'))
    oracle_gt = np.load(os.path.join(STREAMS_DIR, 'oracle_groundtruth.npy'))
    
    # 1. Stress test: Heavy duplication stream (Where BucketOccupancy should dominate)
    run_budget_sweep(
        stream_name="HeavyDuplication_50", 
        stream_file="stream_dup50.npy", 
        source_ids_file="stream_dup50_source_ids.npy",
        queries=queries, 
        oracle_gt=oracle_gt,
        budget_fractions=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    )
    
    # 2. Baseline test: Clean stream (No duplication, worst-case for BucketOccupancy)
    run_budget_sweep(
        stream_name="CleanStream_0", 
        stream_file="stream_clean.npy", 
        source_ids_file="stream_clean_source_ids.npy",
        queries=queries, 
        oracle_gt=oracle_gt,
        budget_fractions=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    )
    
    # 3. Robustness test: Drift stream (Distribution drift)
    run_budget_sweep(
        stream_name="TopicDrift", 
        stream_file="stream_drift.npy", 
        source_ids_file="stream_drift_source_ids.npy",
        queries=queries, 
        oracle_gt=oracle_gt,
        budget_fractions=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    )
    
    print("\nPhase 4 bounded memory experiments finished!")


if __name__ == "__main__":
    main()
