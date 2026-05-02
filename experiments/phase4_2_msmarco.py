import os
import time
import tempfile
import numpy as np
import pandas as pd
import faiss
from benchmark_harness import benchmark
from lsh_hnsw_hybrid import LSH_HNSW_Hybrid

def run_hybrid_sweep(base, queries, gt, output_rows):
    print("Building LSH-HNSW Hybrid Index for MS MARCO...")
    dim = base.shape[1]
    
    n_clusters = 1024 # ~976 vectors per bucket.
    M = 32 # local HNSW edges per node
    
    index = LSH_HNSW_Hybrid(dim, n_clusters=n_clusters, M=M, metric='ip')
    
    t0 = time.time()
    index.build(base)
    build_time = time.time() - t0
    print(f"Hybrid index built in {build_time:.2f}s")
    
    # Measure index size (Approximate by summing local graph sizes)
    # Since hnswlib doesn't have an exact size method without saving, we temporarily save one and multiply,
    # or just skip exact size measurement since we proved the memory benefit conceptually.
    # Actually, we can just save all local graphs to temp files to be mathematically exact.
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
                
            stats = benchmark(query_fn, queries, gt, base=base, metric='ip', k=10)
            
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
    print("Loading MS MARCO datasets...")
    base = np.load('data/msmarco/base.npy')
    queries = np.load('data/msmarco/query.npy')
    gt = np.load('data/msmarco/groundtruth.npy')
    
    base = np.ascontiguousarray(base)
    queries = np.ascontiguousarray(queries)
    
    results = []
    run_hybrid_sweep(base, queries, gt, results)
    
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/phase4_2_msmarco.csv', index=False)
    print("Finished benchmarking. Results saved to results/phase4_2_msmarco.csv")

if __name__ == "__main__":
    main()
