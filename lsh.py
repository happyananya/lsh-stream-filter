import numpy as np


class LSHAdmissionFilter:
    """
    SimHash-based admission filter for streaming novelty detection.

    A vector is admitted only if no bucket-mate in any table has cosine
    similarity >= similarity_threshold. Collisions alone are not enough to
    reject — we verify actual similarity to avoid false rejections from
    unrelated vectors landing in the same bucket by chance.
    """

    def __init__(self, dim: int, k: int, L: int, similarity_threshold: float = 0.9, seed: int = 42):
        """
        dim                 : embedding dimension (384 for all-MiniLM-L6-v2)
        k                   : number of hyperplanes per table (hash bits)
        L                   : number of independent hash tables
        similarity_threshold: cosine similarity above which a vector is redundant
        seed                : RNG seed for reproducibility
        """
        self.dim = dim
        self.k = k
        self.L = L
        self.similarity_threshold = similarity_threshold

        rng = np.random.default_rng(seed)
        # planes[l, i, :] is the i-th hyperplane normal in table l
        self.planes = rng.standard_normal((L, k, dim))  # (L, k, dim)

        self.tables: list[dict[tuple, list[np.ndarray]]] = [{} for _ in range(L)]
        self.stored: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def _hash(self, vec: np.ndarray, table_idx: int) -> tuple:
        """Project normalized vec onto k hyperplanes; return bit-tuple as bucket key."""
        projections = self.planes[table_idx] @ vec  # (k,)
        return tuple((projections >= 0).astype(np.int8).tolist())

    def _is_redundant(self, vec: np.ndarray) -> bool:
        """Return True if any bucket-mate in any table is too similar to vec."""
        for l in range(self.L):
            key = self._hash(vec, l)
            bucket = self.tables[l].get(key)
            if bucket is None:
                continue
            for stored_vec in bucket:
                # Both are already normalized; dot product == cosine similarity
                if float(np.dot(vec, stored_vec)) >= self.similarity_threshold:
                    return True
        return False

    def _store(self, vec: np.ndarray) -> None:
        """Insert normalized vec into all tables and the linear store."""
        for l in range(self.L):
            key = self._hash(vec, l)
            if key not in self.tables[l]:
                self.tables[l][key] = []
            self.tables[l][key].append(vec)
        self.stored.append(vec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, vec: np.ndarray) -> bool:
        """
        Attempt to admit vec into the filter.

        Returns True  — vector is novel, admitted and stored.
        Returns False — vector is redundant, rejected.
        """
        vec = self._normalize(vec.astype(np.float64))

        if self._is_redundant(vec):
            return False

        self._store(vec)
        return True

    def __len__(self) -> int:
        return len(self.stored)

    def reset(self) -> None:
        """Clear all stored state (tables + linear store)."""
        self.tables = [{} for _ in range(self.L)]
        self.stored = []


# ------------------------------------------------------------------
# Smoke tests
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    DIM = 16

    # --- test 1: identical vector is rejected on second insert ---
    f = LSHAdmissionFilter(dim=DIM, k=4, L=4, similarity_threshold=0.9)
    v = rng.standard_normal(DIM)
    assert f.insert(v) is True,  "first insert of unique vector should be admitted"
    assert f.insert(v) is False, "exact duplicate should be rejected"
    assert len(f) == 1,          "exactly one vector should be stored"

    # --- test 2: near-duplicate (tiny noise) is rejected ---
    f.reset()
    v = rng.standard_normal(DIM)
    near_dup = v + rng.standard_normal(DIM) * 0.01
    assert f.insert(v) is True
    assert f.insert(near_dup) is False, "near-duplicate should be rejected"

    # --- test 3: orthogonal vector is admitted ---
    f.reset()
    v1 = np.zeros(DIM); v1[0] = 1.0
    v2 = np.zeros(DIM); v2[1] = 1.0  # perfectly orthogonal
    assert f.insert(v1) is True
    assert f.insert(v2) is True, "orthogonal vector should be admitted"
    assert len(f) == 2

    # --- test 4: many distinct random vectors are admitted ---
    f2 = LSHAdmissionFilter(dim=DIM, k=8, L=8, similarity_threshold=0.9)
    distinct = [rng.standard_normal(DIM) * (i + 1) for i in range(20)]
    # Each cluster vector is in a very different direction; most should pass
    admits = [f2.insert(v) for v in distinct]
    assert sum(admits) >= 10, f"expected most distinct vectors admitted, got {sum(admits)}"

    # --- test 5: cluster of near-duplicates: only ~1 admitted per cluster ---
    f3 = LSHAdmissionFilter(dim=DIM, k=8, L=8, similarity_threshold=0.9)
    center = rng.standard_normal(DIM)
    cluster = [center + rng.standard_normal(DIM) * 0.02 for _ in range(10)]
    cluster_admits = sum(f3.insert(v) for v in cluster)
    assert cluster_admits <= 3, f"tight cluster should admit very few, got {cluster_admits}"

    print("All LSHAdmissionFilter smoke tests passed.")
    print(f"  test4 admitted {sum(admits)}/20 distinct vectors")
    print(f"  test5 admitted {cluster_admits}/10 near-duplicate vectors")
