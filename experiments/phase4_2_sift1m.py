import os
import time
import tempfile
import numpy as np
import pandas as pd
import faiss
from benchmark_harness import benchmark
from lsh_hnsw_hybrid import LSH_HNSW_Hybrid

def run_hybrid_sweep(base, queries, gt, output_rows):
    print("Building LSH-HNSW Hybrid Index for SIFT1M...")
    dim = base.shape[1]
    
    n_clusters = 1024 # ~976 vectors per bucket.
    M = 32 # local HNSW edges per node
    
    index = LSH_HNSW_Hybrid(dim, n_clusters=n_clusters, M=M, metric='l2')
    
    t0 = time.time()
    index.build(base)
    build_time = time.time() - t0
    print(f"Hybrid index built in {build_time:.2f}s")
    
    total_size_bytes = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        # size of coarse router
        router_path = os.path.join(tmpdir, "router.bin")
        faiss.write_index(index.router, router_path)
        total_size_bytes += os.path.getsize(router_path)
        
        # size of local graphs
        for b_id, g in index.local_graphs.items():
            p = os.path.join(tmpdir, f"{b_id}.bin")
            g.save_index(p)
            total_size_bytes += os.path.getsize(p)
            
    index_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Index Size: {index_size_mb:.2f} MB")
    
    nprobe_list = [8, 16, 32, 64, 128, 256]
    ef_search_list = [16, 32, 64, 128]
    
    for nprobe in nprobe_list:
        for ef in ef_search_list:
            print(f"  Evaluating Hybrid nprobe={nprobe}, efSearch={ef}")
            
            def query_fn(q, k):
                return index.search(q, k, nprobe=nprobe, efSearch=ef)
                
            stats = benchmark(query_fn, queries, gt, base=base, metric='l2', k=10)
            
            output_rows.append({
                'Method': 'LSH-HNSW Hybrid (FAISS)',
                'Parameters': f"clusters={n_clusters}, nprobe={nprobe}, ef={ef}",
                'Build_Time_s': build_time,
                'Index_Size_MB': index_size_mb,
                'nprobe': nprobe,
                'efSearch': ef,
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
    run_hybrid_sweep(base, queries, gt, results)
    
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/phase4_2_sift1m.csv', index=False)
    print("Finished benchmarking. Results saved to results/phase4_2_sift1m.csv")

if __name__ == "__main__":
    main()
