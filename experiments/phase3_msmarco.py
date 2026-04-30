import os
import time
import tempfile
import numpy as np
import pandas as pd
import hnswlib
import faiss
import falconn
import psutil
import gc
from benchmark_harness import benchmark

def run_hnsw_sweep(base, queries, gt, output_rows):
    print("Building HNSW Index (M=16, ef_construction=200)...")
    dim = base.shape[1]
    
    # HNSW uses 'ip' (Inner Product) since vectors are L2 normalized
    index = hnswlib.Index(space='ip', dim=dim)
    index.init_index(max_elements=len(base), ef_construction=200, M=16)
    
    # Utilize all CPU cores for building
    index.set_num_threads(os.cpu_count() or 1)
    
    t0 = time.time()
    index.add_items(base, np.arange(len(base)))
    build_time = time.time() - t0
    print(f"HNSW built in {build_time:.2f}s")
    
    # Measure index size
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    index.save_index(temp_path)
    index_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    print(f"  Index size: {index_size_mb:.2f} MB")
    
    # Set thread count back to 1 for the query benchmark to measure raw sequential QPS
    # (The plan dictates single-threaded sequential querying to simulate latency)
    index.set_num_threads(1)
    
    def query_fn(q, k):
        labels, _ = index.knn_query(q, k=k)
        return labels[0]
        
    ef_search_values = [10, 20, 40, 80, 160, 320, 640]
    for ef in ef_search_values:
        print(f"  Evaluating HNSW efSearch={ef}")
        index.set_ef(ef)
        stats = benchmark(query_fn, queries, gt, base=base, metric='ip', k=10)
        
        output_rows.append({
            'Method': 'HNSW',
            'Parameters': f"ef={ef}",
            'Build_Time_s': build_time,
            'Index_Size_MB': index_size_mb,
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
        
        # Measure index size
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        faiss.write_index(index, temp_path)
        index_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        
        # FAISS search expects a batch of queries, so we wrap it
        def query_fn(q, k):
            # q is 1D (384,), faiss needs 2D (1, 384)
            _, I = index.search(q.reshape(1, -1), k)
            return I[0]
            
        stats = benchmark(query_fn, queries, gt, base=base, metric='ip', k=10)
        output_rows.append({
            'Method': 'Vanilla LSH (FAISS)',
            'Parameters': f"nbits={nbits}",
            'Build_Time_s': build_time,
            'Index_Size_MB': index_size_mb,
            **stats
        })

def run_falconn_sweep(base, queries, gt, output_rows):
    print("Running FALCONN Cross-Polytope LSH sweeps...")
    dim = base.shape[1]
    
    # Sweep over a set of parameters (Hash Tables and Probes)
    num_tables_list = [10, 50, 100]
    num_hash_bits = 16 # Fixed for memory efficiency on MS MARCO
    
    for num_tables in num_tables_list:
        print(f"  Building FALCONN L= {num_tables} tables...")
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dim
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
        params_cp.l = num_tables
        params_cp.num_rotations = 1
        params_cp.seed = 42
        # Setup cross polytope dimensions appropriately
        params_cp.num_setup_threads = os.cpu_count() or 1
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        
        falconn.compute_number_of_hash_functions(num_hash_bits, params_cp)
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        t0 = time.time()
        table = falconn.LSHIndex(params_cp)
        table.setup(base)
        build_time = time.time() - t0
        
        mem_after = process.memory_info().rss
        index_size_mb = max(0, (mem_after - mem_before) / (1024 * 1024))
        
        query_object = table.construct_query_object()
        
        def query_fn(q, k):
            return query_object.find_k_nearest_neighbors(q, k)
            
        probes_list = [10, 50, 100, 500, 1000]
        for probes in probes_list:
            if probes < num_tables:
                continue # num probes must be >= num tables
            print(f"    Evaluating FALCONN tables={num_tables}, probes={probes}")
            query_object.set_num_probes(probes)
            stats = benchmark(query_fn, queries, gt, base=base, metric='ip', k=10)
            
            output_rows.append({
                'Method': 'FALCONN',
                'Parameters': f"L={num_tables}, probes={probes}",
                'Build_Time_s': build_time,
                'Index_Size_MB': index_size_mb,
                **stats
            })
            
        # Free memory before next FALCONN table
        del query_object
        del table
        gc.collect()

def main():
    print("Loading datasets...")
    # These paths assume you run from the root directory of the project
    base = np.load('data/msmarco/base.npy')
    queries = np.load('data/msmarco/query.npy')
    gt = np.load('data/msmarco/groundtruth.npy')
    
    # Ensure C-contiguous for C++ libraries (hnsw, falconn, faiss)
    base = np.ascontiguousarray(base)
    queries = np.ascontiguousarray(queries)
    
    results = []
    
    # 1. FAISS LSH Baseline
    run_faiss_lsh_sweep(base, queries, gt, results)
    
    # 2. FALCONN Baseline
    run_falconn_sweep(base, queries, gt, results)
    
    # 3. HNSW Baseline
    run_hnsw_sweep(base, queries, gt, results)
    
    # Save Results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/phase3_msmarco.csv', index=False)
    print("Finished benchmarking. Results saved to results/phase3_msmarco.csv")

if __name__ == "__main__":
    main()
