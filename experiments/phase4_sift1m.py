import os
import time
import numpy as np
import pandas as pd
from benchmark_harness import benchmark
from layered_lsh import LayeredLSH

def run_layered_lsh_sweep(base, queries, gt, output_rows):
    print("Building Layered LSH (FALCONN + PQ) Index for SIFT1M...")
    
    # SIFT1M relies on centering for FALCONN Cross-polytope Euclidean emulation
    print("  Centering datasets for FALCONN and PQ...")
    mean = np.mean(base, axis=0)
    base_centered = base - mean
    queries_centered = queries - mean
    
    num_tables = 50
    num_hash_bits = 16
    pq_M = 16 # 128 / 16 = 8-dimensional subvectors
    pq_nbits = 8
    
    t0 = time.time()
    layered_index = LayeredLSH(
        base=base_centered, 
        num_tables=num_tables, 
        num_hash_bits=num_hash_bits,
        pq_M=pq_M, 
        pq_nbits=pq_nbits, 
        metric='l2'
    )
    build_time = time.time() - t0
    print(f"Layered LSH built in {build_time:.2f}s")
    
    probes_list = [50, 100, 500, 1000]
    rerank_k = 100
    
    for probes in probes_list:
        print(f"  Evaluating LayeredLSH probes={probes}")
        
        def query_fn(q, k):
            return layered_index.search(q, k=k, num_probes=probes, rerank_k=rerank_k)
            
        stats = benchmark(query_fn, queries_centered, gt, base=base_centered, metric='l2', k=10)
        
        output_rows.append({
            'Method': 'Layered LSH (ScaNN-inspired)',
            'Parameters': f"L={num_tables}, probes={probes}",
            'Build_Time_s': build_time,
            'probes': probes,
            **stats
        })

def main():
    print("Loading SIFT1M datasets...")
    base = np.load('data/sift1m/base.npy')
    queries = np.load('data/sift1m/query.npy')
    gt = np.load('data/sift1m/groundtruth.npy')
    
    base = np.ascontiguousarray(base)
    queries = np.ascontiguousarray(queries)
    
    results = []
    run_layered_lsh_sweep(base, queries, gt, results)
    
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/phase4_sift1m.csv', index=False)
    print("Finished benchmarking. Results saved to results/phase4_sift1m.csv")

if __name__ == "__main__":
    main()
