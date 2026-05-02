"""
Retention Policy Framework
===========================
Abstract base class and the core Bucket-Occupancy Retention policy.

All policies conform to the RetentionPolicy interface specified in the research plan §4.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import heapq
import falconn


class RetentionPolicy(ABC):
    """
    Base class for all retention policies.
    
    Every policy:
      - Receives items one at a time via insert()
      - Maintains a kept set of bounded size
      - Can report its current retained items
    """
    
    @abstractmethod
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        """
        Process a new streaming item.
        Returns True if the item was kept, False if discarded.
        """
        pass
    
    @abstractmethod
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (embeddings, item_ids) for all currently retained items.
        embeddings: shape (|M|, d)
        item_ids: shape (|M|,)
        """
        pass
    
    @abstractmethod
    def kept_count(self) -> int:
        """Number of items currently retained."""
        pass
    
    def stats(self) -> dict:
        """Optional stats for diagnostics."""
        return {'kept': self.kept_count()}


class BucketOccupancyRetention(RetentionPolicy):
    """
    LSH Bucket-Occupancy Retention Policy.
    
    Uses LSH bucket occupancy as an intrinsic novelty signal:
    - Items landing in sparse buckets are novel → keep
    - Items landing in dense buckets are redundant → discard
    
    Parameters:
        dim: embedding dimensionality
        L: number of hash tables
        K: number of hash bits (controls bucket granularity)
        capacity: maximum number of retained items (memory budget B)
        aggregator: how to combine occupancy across L tables ('min', 'max', 'mean', 'median')
        threshold: hard occupancy threshold T (keep if O(x) < T). 
                   If None, uses capacity-bounded mode with most-redundant eviction.
        seed: random seed for hash functions
    """
    
    def __init__(self, dim: int, L: int = 8, K: int = 10, 
                 capacity: int = 100000,
                 aggregator: str = 'median',
                 threshold: Optional[int] = None,
                 seed: int = 42):
        self.dim = dim
        self.L = L
        self.K = K
        self.capacity = capacity
        self.aggregator = aggregator
        self.threshold = threshold
        self.seed = seed
        
        # Storage for retained items
        self._storage = {}  # item_id -> (embedding, occupancy_score, bucket_ids)
        self._redundancy_heap = []  # max-heap of (-occupancy, item_id)
        
        # Bucket counts: L tables, each a dict mapping bucket_id -> count
        self._bucket_counts = [{} for _ in range(L)]
        
        # Build the LSH hash family using FALCONN
        self._setup_hash(dim, L, K, seed)
        
        # Stats
        self._n_seen = 0
        self._n_kept = 0
        self._n_discarded = 0
    
    def _setup_hash(self, dim, L, K, seed):
        """Initialize FALCONN cross-polytope hash tables for hashing only."""
        params = falconn.LSHConstructionParameters()
        params.dimension = dim
        params.lsh_family = falconn.LSHFamily.CrossPolytope
        params.distance_function = falconn.DistanceFunction.NegativeInnerProduct
        params.l = L
        params.num_rotations = 1
        params.seed = seed
        params.num_setup_threads = 0
        params.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(K, params)
        
        # We need a tiny dummy dataset to initialize FALCONN
        # (FALCONN requires setup() with data before we can hash)
        dummy = np.random.RandomState(seed).randn(100, dim).astype(np.float32)
        dummy = dummy / np.linalg.norm(dummy, axis=1, keepdims=True)
        
        self._table = falconn.LSHIndex(params)
        self._table.setup(dummy)
        self._hasher = self._table.construct_query_object()
        
        # Store the params for reference
        self._lsh_params = params
    
    def _hash_vector(self, x: np.ndarray) -> list:
        """
        Get the L bucket IDs for a vector x.
        Uses FALCONN's internal hashing to get the bucket signature.
        
        Since FALCONN doesn't expose raw bucket IDs directly,
        we use get_unique_candidates with probes=L to get candidate
        bucket memberships. As a simpler alternative, we compute 
        our own cross-polytope-style hash using random projections.
        """
        # Fallback: use random hyperplane hashing (simpler, still LSH)
        # This is equivalent to SimHash / random hyperplane LSH
        return self._hash_vector_rp(x)
    
    def _hash_vector_rp(self, x: np.ndarray) -> list:
        """Random-projection based LSH hashing."""
        if not hasattr(self, '_projections'):
            rng = np.random.RandomState(self.seed)
            # Combine all projections into a single matrix (L * K, dim)
            self._projections = rng.randn(self.L * self.K, self.dim).astype(np.float32)
            self._powers = (1 << np.arange(self.K)[::-1]).astype(np.int32)
        
        # Vectorized projection and binarization
        proj = self._projections @ x
        bits = (proj > 0).astype(np.int32).reshape(self.L, self.K)
        
        # Convert bits to integers for each table
        bucket_ids = bits.dot(self._powers).tolist()
        return bucket_ids
    
    def _compute_occupancy(self, bucket_ids: list) -> float:
        """Aggregate bucket occupancy across L tables."""
        occupancies = []
        for i, bid in enumerate(bucket_ids):
            occupancies.append(self._bucket_counts[i].get(bid, 0))
        
        occupancies = np.array(occupancies, dtype=np.float64)
        
        if self.aggregator == 'min':
            return float(np.min(occupancies))
        elif self.aggregator == 'max':
            return float(np.max(occupancies))
        elif self.aggregator == 'mean':
            return float(np.mean(occupancies))
        elif self.aggregator == 'median':
            return float(np.median(occupancies))
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
    
    def _increment_buckets(self, bucket_ids: list):
        """Increment bucket counts for all L tables."""
        for i, bid in enumerate(bucket_ids):
            self._bucket_counts[i][bid] = self._bucket_counts[i].get(bid, 0) + 1
    
    def _decrement_buckets(self, bucket_ids: list):
        """Decrement bucket counts (used during eviction)."""
        for i, bid in enumerate(bucket_ids):
            if bid in self._bucket_counts[i]:
                self._bucket_counts[i][bid] -= 1
                if self._bucket_counts[i][bid] <= 0:
                    del self._bucket_counts[i][bid]
    
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        self._n_seen += 1
        
        bucket_ids = self._hash_vector(embedding)
        occupancy = self._compute_occupancy(bucket_ids)
        
        if self.threshold is not None:
            # Hard threshold mode
            if occupancy >= self.threshold:
                self._n_discarded += 1
                return False
            
            # Keep the item
            if len(self._storage) >= self.capacity:
                self._evict_most_redundant()
            
            self._storage[item_id] = (embedding.copy(), occupancy, bucket_ids)
            heapq.heappush(self._redundancy_heap, (-occupancy, item_id))
            self._increment_buckets(bucket_ids)
            self._n_kept += 1
            return True
        
        else:
            # Capacity-bounded mode (no threshold)
            if len(self._storage) < self.capacity:
                # Under capacity: always keep
                self._storage[item_id] = (embedding.copy(), occupancy, bucket_ids)
                heapq.heappush(self._redundancy_heap, (-occupancy, item_id))
                self._increment_buckets(bucket_ids)
                self._n_kept += 1
                return True
            else:
                # At capacity: only keep if more novel than most redundant
                # max_redundant is at the top of the max-heap (negative occupancy)
                max_neg_occ, max_redundant_id = self._redundancy_heap[0]
                max_redundant_occ = -max_neg_occ
                
                if occupancy < max_redundant_occ:
                    # Evict the most redundant item
                    self._evict_most_redundant()
                    
                    self._storage[item_id] = (embedding.copy(), occupancy, bucket_ids)
                    heapq.heappush(self._redundancy_heap, (-occupancy, item_id))
                    self._increment_buckets(bucket_ids)
                    self._n_kept += 1
                    return True
                else:
                    self._n_discarded += 1
                    return False
    
    def _evict_most_redundant(self):
        """Evict the item with the highest occupancy score from the heap."""
        if not self._redundancy_heap:
            return
        
        # Find a valid item in the heap (lazy deletion handling)
        while self._redundancy_heap:
            neg_occ, item_id = heapq.heappop(self._redundancy_heap)
            if item_id in self._storage:
                _, _, evicted_bids = self._storage.pop(item_id)
                self._decrement_buckets(evicted_bids)
                return
    
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._storage:
            return np.array([]).reshape(0, self.dim), np.array([], dtype=np.int64)
        
        embeddings = []
        item_ids = []
        for item_id, (emb, _, _) in self._storage.items():
            embeddings.append(emb)
            item_ids.append(item_id)
            
        return np.array(embeddings), np.array(item_ids)
    
    def kept_count(self) -> int:
        return len(self._storage)
    
    def stats(self) -> dict:
        total_buckets = sum(len(bc) for bc in self._bucket_counts)
        all_counts = []
        for bc in self._bucket_counts:
            all_counts.extend(bc.values())
        
        return {
            'kept': self.kept_count(),
            'seen': self._n_seen,
            'discarded': self._n_discarded,
            'retention_rate': self._n_kept / max(self._n_seen, 1),
            'active_buckets': total_buckets,
            'mean_bucket_occupancy': np.mean(all_counts) if all_counts else 0,
            'max_bucket_occupancy': max(all_counts) if all_counts else 0,
        }
    
    def bucket_occupancy_histogram(self) -> np.ndarray:
        """Returns all bucket counts for histogram plotting."""
        all_counts = []
        for bc in self._bucket_counts:
            all_counts.extend(bc.values())
        return np.array(all_counts) if all_counts else np.array([0])
