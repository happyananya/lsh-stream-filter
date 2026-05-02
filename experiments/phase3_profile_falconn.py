import os
import time
import numpy as np
import pandas as pd
import falconn
from benchmark_harness import benchmark

def profile_dataset(dataset_name, base_path, query_path, gt_path, metric, num_tables=50):
    print(f"Loading {dataset_name}...")
    base = np.load(base_path)
    queries = np.load(query_path)
    gt = np.load(gt_path)
    
    # SIFT1M requires centering for FALCONN Cross-polytope
    if dataset_name == 'SIFT1M':
        mean = np.mean(base, axis=0)
        base_aligned = base - mean
        queries_aligned = queries - mean
    else:
        base_aligned = base
        queries_aligned = queries
        
    base_aligned = np.ascontiguousarray(base_aligned)
    queries_aligned = np.ascontiguousarray(queries_aligned)
    dim = base.shape[1]
    
    print(f"Building FALCONN Index for {dataset_name} (L={num_tables})...")
    params = falconn.LSHConstructionParameters()
    params.dimension = dim
    params.lsh_family = falconn.LSHFamily.CrossPolytope
    params.distance_function = falconn.DistanceFunction.EuclideanSquared if metric == 'l2' else falconn.DistanceFunction.NegativeInnerProduct
    params.l = num_tables
    params.num_rotations = 2 if metric == 'l2' else 1
    params.seed = 42
    params.num_setup_threads = 0
    params.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    
    falconn.compute_number_of_hash_functions(16, params)
    table = falconn.LSHIndex(params)
    table.setup(base_aligned)
    
    query_object = table.construct_query_object()
    
    probes_list = [50, 100, 500, 1000, 2000]
    output_rows = []
    
    for probes in probes_list:
        print(f"  Profiling {dataset_name} probes={probes}...")
        query_object.set_num_probes(probes)
        query_object.reset_query_statistics()
        
        def query_fn(q, k):
            return query_object.find_k_nearest_neighbors(q, k)
            
        # benchmark_harness will execute the queries, calculating recall and QPS
        stats = benchmark(query_fn, queries_aligned, gt, base=base_aligned, metric=metric, k=10)
        
        # After executing, extract the micro-stats accumulated inside FALCONN
        q_stats = query_object.get_query_statistics()
        
        output_rows.append({
            'Dataset': dataset_name,
            'probes': probes,
            'recall': stats['recall'],
            'qps': stats['qps'],
            'avg_total_query_time_s': q_stats.average_total_query_time,
            'avg_lsh_time_s': q_stats.average_lsh_time,
            'avg_hash_table_time_s': q_stats.average_hash_table_time,
            'avg_distance_time_s': q_stats.average_distance_time,
            'avg_num_candidates': q_stats.average_num_candidates,
            'avg_unique_candidates': q_stats.average_num_unique_candidates,
        })
        
    df = pd.DataFrame(output_rows)
    out_path = f"results/profiling_{dataset_name.lower()}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved profiling results to {out_path}")

def main():
    os.makedirs('results', exist_ok=True)
    profile_dataset('MSMARCO', 'data/msmarco/base.npy', 'data/msmarco/query.npy', 'data/msmarco/groundtruth.npy', 'ip')
    profile_dataset('SIFT1M', 'data/sift1m/base.npy', 'data/sift1m/query.npy', 'data/sift1m/groundtruth.npy', 'l2')

if __name__ == "__main__":
    main()
