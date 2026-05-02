import os
import time
import numpy as np
import pandas as pd
from benchmark_harness import benchmark
from layered_lsh import LayeredLSH

def run_layered_lsh_sweep(base, queries, gt, output_rows):
    print("Building Layered LSH (FALCONN + PQ) Index for MS MARCO...")
    
    # Base FALCONN configuration that had good memory footprint (L=50 tables)
    num_tables = 50
    num_hash_bits = 16
    pq_M = 32 # 384 / 32 = 12-dimensional subvectors
    pq_nbits = 8
    
    t0 = time.time()
    layered_index = LayeredLSH(
        base=base, 
        num_tables=num_tables, 
        num_hash_bits=num_hash_bits,
        pq_M=pq_M, 
        pq_nbits=pq_nbits, 
        metric='ip'
    )
    build_time = time.time() - t0
    print(f"Layered LSH built in {build_time:.2f}s")
    
    # We don't track dynamic memory here because we proved memory is fine in Phase 3.
    # The goal of Phase 4 is to show QPS / Recall improvement.
    
    probes_list = [50, 100, 500, 1000, 2000]
    rerank_k = 100 # Exact distance calculation on top 100
    
    for probes in probes_list:
        print(f"  Evaluating LayeredLSH probes={probes}")
        
        def query_fn(q, k):
            return layered_index.search(q, k=k, num_probes=probes, rerank_k=rerank_k)
            
        stats = benchmark(query_fn, queries, gt, base=base, metric='ip', k=10)
        
        output_rows.append({
            'Method': 'Layered LSH (ScaNN-inspired)',
            'Parameters': f"L={num_tables}, probes={probes}",
            'Build_Time_s': build_time,
            'probes': probes,
            **stats
        })

def main():
    print("Loading datasets...")
    base = np.load('data/msmarco/base.npy')
    queries = np.load('data/msmarco/query.npy')
    gt = np.load('data/msmarco/groundtruth.npy')
    
    base = np.ascontiguousarray(base)
    queries = np.ascontiguousarray(queries)
    
    results = []
    run_layered_lsh_sweep(base, queries, gt, results)
    
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/phase4_msmarco.csv', index=False)
    print("Finished benchmarking. Results saved to results/phase4_msmarco.csv")

if __name__ == "__main__":
    main()
