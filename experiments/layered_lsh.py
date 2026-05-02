import faiss
import falconn
import numpy as np

class LayeredLSH:
    """
    A Two-Stage Quantized Filtering approach that acts as a wrapper around FALCONN.
    Phase 1: Coarse Routing. FALCONN probes hash tables to get a candidate pool.
    Phase 2: Asymmetric Distance Computation (ADC). Uses Product Quantization (PQ) 
             via SIMD-optimized NumPy to score candidates in micro-seconds without 
             full dimension distance evaluations.
    Phase 3: Exact Reranking. Full distance evaluation on the top `rerank_k` items.
    """
    def __init__(self, base, num_tables=50, num_hash_bits=16, 
                 pq_M=32, pq_nbits=8, metric='ip'):
        self.base = base
        self.dim = base.shape[1]
        self.metric = metric
        self.M = pq_M
        self.nbits = pq_nbits
        self.ksub = 1 << pq_nbits
        self.dsub = self.dim // self.M
        
        # 1. Build FALCONN Index
        falconn_metric = falconn.DistanceFunction.EuclideanSquared if metric == 'l2' else falconn.DistanceFunction.NegativeInnerProduct
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = self.dim
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn_metric
        params_cp.l = num_tables
        params_cp.num_rotations = 2 if metric == 'l2' else 1
        params_cp.seed = 42
        params_cp.num_setup_threads = 0 # Uses available cores
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        
        falconn.compute_number_of_hash_functions(num_hash_bits, params_cp)
        self.table = falconn.LSHIndex(params_cp)
        self.table.setup(base)
        self.query_object = self.table.construct_query_object()
        
        # 2. Build FAISS PQ Index
        # We use FAISS just to train the centroids and compute codes quickly
        self.pq = faiss.ProductQuantizer(self.dim, self.M, self.nbits)
        
        # Train on a random subset
        np.random.seed(42)
        idx = np.random.choice(len(base), min(len(base), 100000), replace=False)
        self.pq.train(base[idx])
        
        # Compute codes for the entire base database
        self.codes = self.pq.compute_codes(base)
        
        # Extract centroids for ultra-fast NumPy ADC
        centroids = faiss.vector_to_array(self.pq.centroids)
        self.centroids = centroids.reshape(self.M, self.ksub, self.dsub)
        
    def search(self, q, k=10, num_probes=100, max_candidates=5000, rerank_k=100):
        self.query_object.set_num_probes(num_probes)
        
        # FALCONN Coarse Routing
        # get_unique_candidates skips FALCONN's exact distance calculation!
        candidates = self.query_object.get_unique_candidates(q)
        
        if len(candidates) == 0:
            return []
            
        candidates = np.array(candidates)
        # We NO LONGER arbitrarily truncate candidates here because get_unique_candidates 
        # destroys the geometric ordering of the multi-probe sequence (it returns an unordered set).
        # Slicing it would result in a random sample, severely degrading recall at high probes.
        # Since PQ ADC filtering is heavily vectorized and takes < 1ms for even 50,000 vectors,
        # we can afford to rapidly score the *entire* candidate pool!
            
        # PQ Asymmetric Distance Computation (ADC)
        q_split = q.reshape(self.M, 1, self.dsub)
        if self.metric == 'l2':
            L = np.sum((q_split - self.centroids)**2, axis=2)
        else:
            L = np.sum(q_split * self.centroids, axis=2)
            
        cand_codes = self.codes[candidates]
        m_idx = np.arange(self.M)[np.newaxis, :]
        
        # ADC scores using lookup table L
        adc_scores = np.sum(L[m_idx, cand_codes], axis=1)
        
        # Exact Reranking
        top_k = min(len(candidates), rerank_k)
        if self.metric == 'l2':
            top_adc_idx = np.argpartition(adc_scores, top_k - 1)[:top_k]
        else:
            top_adc_idx = np.argpartition(adc_scores, -top_k)[-top_k:]
            
        rerank_candidates = candidates[top_adc_idx]
        
        # Final Exact distance calculation
        exact_base = self.base[rerank_candidates]
        if self.metric == 'l2':
            exact_scores = np.linalg.norm(exact_base - q, axis=1)
            final_idx = np.argsort(exact_scores)[:k]
        else:
            exact_scores = np.dot(exact_base, q)
            final_idx = np.argsort(exact_scores)[::-1][:k]
            
        return rerank_candidates[final_idx]
