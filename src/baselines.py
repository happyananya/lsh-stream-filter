"""
Baseline Retention Policies (Phase 3 of the research plan)
============================================================
All baselines conform to the same RetentionPolicy interface.
"""
import numpy as np
from typing import Tuple
from retention_policies import RetentionPolicy


class OracleRetention(RetentionPolicy):
    """Keep everything. Recall ceiling / upper bound."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self._embeddings = []
        self._item_ids = []
    
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        self._embeddings.append(embedding.copy())
        self._item_ids.append(item_id)
        return True
    
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._embeddings:
            return np.array([]).reshape(0, self.dim), np.array([], dtype=np.int64)
        return np.array(self._embeddings), np.array(self._item_ids)
    
    def kept_count(self) -> int:
        return len(self._embeddings)


class FIFORetention(RetentionPolicy):
    """Keep most recent B items. Evict oldest on overflow."""
    
    def __init__(self, dim: int, capacity: int):
        self.dim = dim
        self.capacity = capacity
        self._embeddings = []
        self._item_ids = []
    
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        if len(self._embeddings) >= self.capacity:
            self._embeddings.pop(0)
            self._item_ids.pop(0)
        self._embeddings.append(embedding.copy())
        self._item_ids.append(item_id)
        return True
    
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._embeddings:
            return np.array([]).reshape(0, self.dim), np.array([], dtype=np.int64)
        return np.array(self._embeddings), np.array(self._item_ids)
    
    def kept_count(self) -> int:
        return len(self._embeddings)


class ReservoirSamplingRetention(RetentionPolicy):
    """
    Vitter's Algorithm R — uniform random sample of size B.
    Standard streaming baseline.
    """
    
    def __init__(self, dim: int, capacity: int, seed: int = 42):
        self.dim = dim
        self.capacity = capacity
        self._rng = np.random.RandomState(seed)
        self._embeddings = []
        self._item_ids = []
        self._n_seen = 0
    
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        self._n_seen += 1
        
        if len(self._embeddings) < self.capacity:
            self._embeddings.append(embedding.copy())
            self._item_ids.append(item_id)
            return True
        else:
            # Replace with probability capacity / n_seen
            j = self._rng.randint(0, self._n_seen)
            if j < self.capacity:
                self._embeddings[j] = embedding.copy()
                self._item_ids[j] = item_id
                return True
            return False
    
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._embeddings:
            return np.array([]).reshape(0, self.dim), np.array([], dtype=np.int64)
        return np.array(self._embeddings), np.array(self._item_ids)
    
    def kept_count(self) -> int:
        return len(self._embeddings)


class RandomSamplingRetention(RetentionPolicy):
    """Keep each item with fixed probability p = B/N (requires knowing N)."""
    
    def __init__(self, dim: int, capacity: int, stream_size: int, seed: int = 42):
        self.dim = dim
        self.capacity = capacity
        self._p = capacity / stream_size
        self._rng = np.random.RandomState(seed)
        self._embeddings = []
        self._item_ids = []
    
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        if self._rng.random() < self._p:
            self._embeddings.append(embedding.copy())
            self._item_ids.append(item_id)
            return True
        return False
    
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._embeddings:
            return np.array([]).reshape(0, self.dim), np.array([], dtype=np.int64)
        return np.array(self._embeddings), np.array(self._item_ids)
    
    def kept_count(self) -> int:
        return len(self._embeddings)


class SemanticDedupRetention(RetentionPolicy):
    """
    Semantic deduplication with ε-threshold.
    For each new item, if max cosine similarity to any kept item > 1 - ε, discard.
    Otherwise keep until capacity B is reached.
    """
    
    def __init__(self, dim: int, capacity: int, epsilon: float = 0.1):
        self.dim = dim
        self.capacity = capacity
        self.epsilon = epsilon
        self._embeddings = []
        self._item_ids = []
        
        import faiss
        # We assume L2-normalized embeddings, so Inner Product = Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
    
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        if len(self._embeddings) >= self.capacity:
            return False  # At capacity, don't accept more
        
        if self.index.ntotal > 0:
            # Query the FAISS index (fast)
            q = embedding.reshape(1, -1).astype(np.float32)
            D, _ = self.index.search(q, 1)
            max_sim = D[0][0]
            
            if max_sim > (1.0 - self.epsilon):
                # Too similar to an existing item — discard
                return False
        
        # Keep it
        self._embeddings.append(embedding.copy())
        self._item_ids.append(item_id)
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        return True
    
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._embeddings:
            return np.array([]).reshape(0, self.dim), np.array([], dtype=np.int64)
        return np.array(self._embeddings), np.array(self._item_ids)
    
    def kept_count(self) -> int:
        return len(self._embeddings)


class StreamLSHRetention(RetentionPolicy):
    """
    Stream-LSH (Kraus, Carmel, Keidar — IEEE BigData 2017).
    
    In the general case, this maintains:
    score(x) = freshness(x) * quality(x) + popularity(x)
    
    For our ingestion-only experiments (as per Research Plan):
    - quality = 1 (no application-supplied quality)
    - popularity = 0 (no query stream)
    - freshness = exp(-decay * (t - t_insert))
    
    Under these specific settings, the item with the lowest score is ALWAYS
    the oldest item, making this functionally equivalent to a FIFO cache.
    We implement it explicitly to match the literature and to allow
    future extension with query streams.
    """
    
    def __init__(self, dim: int, capacity: int, decay_rate: float = 0.001):
        self.dim = dim
        self.capacity = capacity
        self.decay_rate = decay_rate
        
        self._embeddings = []
        self._item_ids = []
        # In a full implementation with dynamic popularity, we would need 
        # a min-heap or priority queue. Since we only use freshness here,
        # popping the oldest item (index 0) is sufficient and exactly correct.
    
    def insert(self, embedding: np.ndarray, item_id: int) -> bool:
        if len(self._embeddings) >= self.capacity:
            # Evict the item with the lowest score (the oldest item)
            self._embeddings.pop(0)
            self._item_ids.pop(0)
            
        self._embeddings.append(embedding.copy())
        self._item_ids.append(item_id)
        return True
    
    def kept_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._embeddings:
            return np.array([]).reshape(0, self.dim), np.array([], dtype=np.int64)
        return np.array(self._embeddings), np.array(self._item_ids)
    
    def kept_count(self) -> int:
        return len(self._embeddings)
