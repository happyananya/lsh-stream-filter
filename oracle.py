import numpy as np


class BruteForceOracle:
    """
    O(n)-per-insert ground-truth novelty filter.

    Rejects a vector if any already-stored vector has cosine similarity
    >= similarity_threshold. Gives perfect novelty decisions with no
    approximation error; used as the evaluation baseline.
    """

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.stored: list[np.ndarray] = []

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def insert(self, vec: np.ndarray) -> bool:
        """
        Returns True  — vector is novel, admitted and stored.
        Returns False — vector is redundant, rejected.
        """
        vec = self._normalize(vec.astype(np.float64))

        if self.stored:
            # Matrix of stored vectors: (n, dim); dot with vec gives cosine sims
            sims = np.array(self.stored) @ vec
            if np.any(sims >= self.similarity_threshold):
                return False

        self.stored.append(vec)
        return True

    def __len__(self) -> int:
        return len(self.stored)

    def reset(self) -> None:
        self.stored = []


# ------------------------------------------------------------------
# Smoke tests
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    DIM = 16

    # --- test 1: exact duplicate rejected ---
    o = BruteForceOracle(similarity_threshold=0.9)
    v = rng.standard_normal(DIM)
    assert o.insert(v) is True
    assert o.insert(v) is False, "exact duplicate must be rejected"
    assert len(o) == 1

    # --- test 2: near-duplicate rejected ---
    o.reset()
    v = rng.standard_normal(DIM)
    near_dup = v + rng.standard_normal(DIM) * 0.01
    assert o.insert(v) is True
    assert o.insert(near_dup) is False, "near-duplicate must be rejected"

    # --- test 3: orthogonal vectors both admitted ---
    o.reset()
    v1 = np.zeros(DIM); v1[0] = 1.0
    v2 = np.zeros(DIM); v2[1] = 1.0
    assert o.insert(v1) is True
    assert o.insert(v2) is True
    assert len(o) == 2

    # --- test 4: cluster of 10 near-dups — oracle stores exactly 1 ---
    o2 = BruteForceOracle(similarity_threshold=0.9)
    center = rng.standard_normal(DIM)
    cluster = [center + rng.standard_normal(DIM) * 0.02 for _ in range(10)]
    admits = [o2.insert(v) for v in cluster]
    assert sum(admits) == 1, f"oracle should admit exactly 1 from tight cluster, got {sum(admits)}"

    # --- test 5: threshold boundary — just above vs just below ---
    o3 = BruteForceOracle(similarity_threshold=0.9)
    base = np.zeros(DIM); base[0] = 1.0
    o3.insert(base)

    # construct vector with known cosine sim ~0.95 (above threshold)
    above = np.zeros(DIM); above[0] = 0.95; above[1] = np.sqrt(1 - 0.95**2)
    assert o3.insert(above) is False, "vector above threshold should be rejected"

    # construct vector with cosine sim ~0.5 (below threshold)
    below = np.zeros(DIM); below[0] = 0.5; below[1] = np.sqrt(1 - 0.5**2)
    assert o3.insert(below) is True, "vector below threshold should be admitted"

    print("All BruteForceOracle smoke tests passed.")
