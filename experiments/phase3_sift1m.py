import os
import time
import numpy as np
import pandas as pd
import hnswlib
import faiss
import falconn
from benchmark_harness import benchmark

def run_hnsw_sweep(base, queries, gt, output_rows):
    print("Building HNSW Index (M=16, ef_construction=200)...")
    dim = base.shape[1]
    
    # SIFT1M relies on L2 Euclidean distance
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=len(base), ef_construction=200, M=16)
    
    # Utilize CPU threads for index building
    index.set_num_threads(os.cpu_count() or 1)
    
    t0 = time.time()
    index.add_items(base, np.arange(len(base)))
    build_time = time.time() - t0
    print(f"HNSW built in {build_time:.2f}s")
    
    # Single thread for sequential benchmarking per the research plan
    index.set_num_threads(1)
    
    def query_fn(q, k):
        labels, _ = index.knn_query(q, k=k)
        return labels[0]
        
    ef_search_values = [10, 20, 40, 80, 160, 320, 640]
    for ef in ef_search_values:
        print(f"  Evaluating HNSW efSearch={ef}")
        index.set_ef(ef)
        stats = benchmark(query_fn, queries, gt, k=10)
        
        output_rows.append({
            'Method': 'HNSW',
            'Parameters': f"ef={ef}",
            'Build_Time_s': build_time,
            **stats
        })

def run_faiss_lsh_sweep(base, queries, gt, output_rows):
    print("Running FAISS Random-Hyperplane LSH sweeps...")
    dim = base.shape[1]
    
    nbits_values = [128, 256, 512, 1024, 2048, 4096]
    for nbits in nbits_values:
        print(f"  Evaluating FAISS LSH nbits={nbits}")
        index = faiss.IndexLSH(dim, nbits)
        
        t0 = time.time()
        index.add(base)
        build_time = time.time() - t0
        
        def query_fn(q, k):
            _, I = index.search(q.reshape(1, -1), k)
            return I[0]
            
        stats = benchmark(query_fn, queries, gt, k=10)
        output_rows.append({
            'Method': 'Vanilla LSH (FAISS)',
            'Parameters': f"nbits={nbits}",
            'Build_Time_s': build_time,
            **stats
        })

def run_falconn_sweep(base, queries, gt, output_rows):
    print("Running FALCONN Cross-Polytope LSH sweeps...")
    dim = base.shape[1]
    
    # SIFT1M vectors are not naturally normalized.
    # FALCONN with EuclideanSquared relies on data centering to emulate angular distance well for CP.
    print("  Centering datasets for FALCONN Euclidean operations...")
    mean = np.mean(base, axis=0)
    base_centered = base - mean
    queries_centered = queries - mean
    
    num_tables_list = [10, 50, 100]
    num_hash_bits = 16 
    
    for num_tables in num_tables_list:
        print(f"  Building FALCONN L= {num_tables} tables...")
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dim
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = num_tables
        params_cp.num_rotations = 2 # Best for SIFT1M (128d)
        params_cp.seed = 42
        params_cp.num_setup_threads = os.cpu_count() or 1
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        
        falconn.compute_number_of_hash_functions(num_hash_bits, params_cp)
        
        t0 = time.time()
        table = falconn.LSHIndex(params_cp)
        table.setup(base_centered)
        build_time = time.time() - t0
        
        query_object = table.construct_query_object()
        
        def query_fn(q, k):
            return query_object.find_k_nearest_neighbors(q, k)
            
        probes_list = [10, 50, 100, 500, 1000]
        for probes in probes_list:
            if probes < num_tables:
                continue
            print(f"    Evaluating FALCONN tables={num_tables}, probes={probes}")
            query_object.set_num_probes(probes)
            stats = benchmark(query_fn, queries_centered, gt, k=10)
            
            output_rows.append({
                'Method': 'FALCONN',
                'Parameters': f"L={num_tables}, probes={probes}",
                'Build_Time_s': build_time,
                **stats
            })

def main():
    print("Loading SIFT1M datasets...")
    base = np.load('data/sift1m/base.npy')
    queries = np.load('data/sift1m/query.npy')
    gt = np.load('data/sift1m/groundtruth.npy')
    
    # Ensure memory mapping is C-contiguous to avoid C++ binding issues
    base = np.ascontiguousarray(base)
    queries = np.ascontiguousarray(queries)
    
    results = []
    
    run_faiss_lsh_sweep(base, queries, gt, results)
    run_falconn_sweep(base, queries, gt, results)
    run_hnsw_sweep(base, queries, gt, results)
    
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/phase3_sift1m.csv', index=False)
    print("Finished benchmarking. Results saved to results/phase3_sift1m.csv")

if __name__ == "__main__":
    main()
