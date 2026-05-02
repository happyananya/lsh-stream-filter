import faiss
import hnswlib
import numpy as np

class LSH_HNSW_Hybrid:
    """
    A true Space-Partitioned Graph Hybrid (SPANN-inspired).
    Stage 1: Coarse Routing via FAISS K-Means + Flat Index.
    Stage 2: Local Graph Traversal via a dictionary of small hnswlib graphs.
    """
    def __init__(self, dim, n_clusters=1024, M=16, ef_construction=200, metric='ip'):
        self.dim = dim
        self.n_clusters = n_clusters
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        
        # Coarse router
        faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == 'ip' else faiss.METRIC_L2
        self.router = faiss.IndexFlat(self.dim, faiss_metric)
        
        # Local graphs
        self.local_graphs = {}
        self.local_to_global = {}
        
    def build(self, base_vectors):
        print(f"Clustering {len(base_vectors)} vectors into {self.n_clusters} regions...")
        kmeans = faiss.Kmeans(self.dim, self.n_clusters, niter=10, verbose=False, spherical=(self.metric=='ip'))
        kmeans.train(base_vectors)
        
        print("Building coarse router...")
        self.router.add(kmeans.centroids)
        
        print("Assigning vectors to regions...")
        _, bucket_ids = self.router.search(base_vectors, 1)
        bucket_ids = bucket_ids.flatten()
        
        print("Distributing vectors to local HNSW graphs...")
        unique_buckets = np.unique(bucket_ids)
        hnsw_space = 'ip' if self.metric == 'ip' else 'l2'
        
        for bucket_id in unique_buckets:
            global_ids = np.where(bucket_ids == bucket_id)[0]
            if len(global_ids) == 0:
                continue
                
            local_index = hnswlib.Index(space=hnsw_space, dim=self.dim)
            local_index.init_index(max_elements=len(global_ids), 
                                   ef_construction=self.ef_construction, M=self.M)
            
            local_index.add_items(base_vectors[global_ids], np.arange(len(global_ids)))
            # Optimize memory: set threads to 1 for search phase
            local_index.set_num_threads(1)
            
            self.local_graphs[bucket_id] = local_index
            self.local_to_global[bucket_id] = global_ids
            
    def search(self, q, k=10, nprobe=1, efSearch=32):
        q_2d = q.reshape(1, -1)
        
        # 1. Coarse routing (microseconds)
        _, top_buckets = self.router.search(q_2d, nprobe)
        top_buckets = top_buckets.flatten()
        
        all_local_ids = []
        all_distances = []
        all_global_ids = []
        
        # 2. Local graph traversal
        for bucket_id in top_buckets:
            if bucket_id not in self.local_graphs:
                continue
            local_index = self.local_graphs[bucket_id]
            local_index.set_ef(efSearch)
            
            try:
                # Query the local graph
                labels, distances = local_index.knn_query(q_2d, k=k)
                all_local_ids.extend(labels[0])
                all_distances.extend(distances[0])
                # Map local ID back to global ID
                all_global_ids.extend(self.local_to_global[bucket_id][labels[0]])
            except RuntimeError:
                pass # Graph might have fewer than k elements
                
        if len(all_global_ids) == 0:
            return []
            
        # 3. Merge and sort
        all_distances = np.array(all_distances)
        all_global_ids = np.array(all_global_ids)
        
        # hnswlib returns distances directly. For L2, smaller is better. For IP, smaller is better (it returns 1 - IP).
        # So we always sort ascending.
        sort_idx = np.argsort(all_distances)[:k]
        
        return all_global_ids[sort_idx]
