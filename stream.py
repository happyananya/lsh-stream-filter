"""
Stream generation for the LSH admission filter experiments.

Two modes
---------
synthetic_stream  — controlled clusters of near-duplicates in low-dim space
real_stream       — MS MARCO passages encoded with all-MiniLM-L6-v2,
                    with injected redundancy at a specified fraction
"""

from __future__ import annotations

import os
import numpy as np


# ------------------------------------------------------------------
# Synthetic stream
# ------------------------------------------------------------------

def synthetic_stream(
    n_clusters: int = 10,
    cluster_size: int = 20,
    dim: int = 50,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[bool]]:
    """
    Build a shuffled stream of near-duplicate clusters.

    Each cluster has one "center" vector (novel) plus (cluster_size - 1)
    noisy copies (redundant).  After shuffling, cluster members are
    interleaved so the filter sees them in realistic order.

    Returns
    -------
    vectors : list of (dim,) float64 arrays, already L2-normalised
    labels  : True = novel (first occurrence of cluster center),
              False = redundant (near-duplicate copy)
    """
    rng = np.random.default_rng(seed)

    vectors: list[np.ndarray] = []
    labels:  list[bool]       = []

    for _ in range(n_clusters):
        center = rng.standard_normal(dim)
        center /= np.linalg.norm(center)

        # First element of the cluster is the novel prototype
        vectors.append(center.copy())
        labels.append(True)

        # Remaining elements are near-duplicates
        for _ in range(cluster_size - 1):
            noisy = center + rng.standard_normal(dim) * noise_sigma
            noisy /= np.linalg.norm(noisy)
            vectors.append(noisy)
            labels.append(False)

    # Shuffle together so clusters are interleaved
    indices = rng.permutation(len(vectors))
    vectors = [vectors[i] for i in indices]
    labels  = [labels[i]  for i in indices]

    return vectors, labels


# ------------------------------------------------------------------
# Real stream (MS MARCO)
# ------------------------------------------------------------------

_EMBED_CACHE = os.path.join(os.path.dirname(__file__), "embeddings.npy")
_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
_N_PASSAGES  = 5000


def _load_or_encode(cache_path: str = _EMBED_CACHE, n: int = _N_PASSAGES) -> np.ndarray:
    """Return (n, 384) float32 embeddings, encoding once and caching thereafter."""
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        print(f"Loaded cached embeddings: {embeddings.shape} from {cache_path}")
        return embeddings

    print(f"Encoding {n} MS MARCO passages with {_MODEL_NAME} …")
    # Lazy imports — not needed if cache already exists
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    passages: list[str] = []
    for row in tqdm(ds, total=n, desc="loading passages"):
        for p in row["passages"]["passage_text"]:
            passages.append(p)
            if len(passages) >= n:
                break
        if len(passages) >= n:
            break
    passages = passages[:n]

    model = SentenceTransformer(_MODEL_NAME)
    embeddings = model.encode(passages, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}: {embeddings.shape}")
    return embeddings


def real_stream(
    redundancy: float = 0.5,
    noise_sigma: float = 0.01,
    seed: int = 42,
    cache_path: str = _EMBED_CACHE,
    n_passages: int = _N_PASSAGES,
) -> tuple[list[np.ndarray], list[bool]]:
    """
    Build a real-embedding stream with controlled redundancy.

    redundancy   : fraction of stream that is duplicate (0.0 – 1.0)
    noise_sigma  : Gaussian noise added when duplicating a vector
    seed         : RNG seed

    Returns
    -------
    vectors : list of (384,) float64 arrays, L2-normalised
    labels  : True = novel base embedding, False = injected near-duplicate
    """
    rng = np.random.default_rng(seed)
    embeddings = _load_or_encode(cache_path, n_passages).astype(np.float64)

    # Normalise (model may already return unit vectors, but be safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1.0, norms)

    n_base = len(embeddings)
    # How many duplicates to inject so that:  n_dup / (n_base + n_dup) = redundancy
    if redundancy >= 1.0:
        raise ValueError("redundancy must be < 1.0")
    n_dup = int(round(n_base * redundancy / (1.0 - redundancy)))

    # Choose which base vectors to duplicate (with replacement if n_dup > n_base)
    dup_indices = rng.choice(n_base, size=n_dup, replace=True)

    novel_vecs = list(embeddings)
    novel_labels: list[bool] = [True] * n_base

    dup_vecs: list[np.ndarray] = []
    dup_labels: list[bool] = []
    for idx in dup_indices:
        noisy = embeddings[idx] + rng.standard_normal(embeddings.shape[1]) * noise_sigma
        noisy /= np.linalg.norm(noisy)
        dup_vecs.append(noisy)
        dup_labels.append(False)

    combined_vecs   = novel_vecs + dup_vecs
    combined_labels = novel_labels + dup_labels

    # Shuffle
    perm = rng.permutation(len(combined_vecs))
    vectors = [combined_vecs[i]   for i in perm]
    labels  = [combined_labels[i] for i in perm]

    return vectors, labels


# ------------------------------------------------------------------
# Embedding analysis utilities
# ------------------------------------------------------------------

def empirical_near_dup_angle(
    embeddings: np.ndarray,
    sim_threshold: float = 0.8,
    n_sample: int = 500,
    seed: int = 42,
) -> float:
    """
    Sample up to n_sample embeddings, find near-duplicate pairs
    (cosine similarity > sim_threshold), and return the mean angle in degrees.

    Prints a summary; returns mean angle (or NaN if no pairs found).
    """
    rng = np.random.default_rng(seed)
    n = min(n_sample, len(embeddings))
    idx = rng.choice(len(embeddings), size=n, replace=False)
    sample = embeddings[idx].astype(np.float64)

    # Normalise sample
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    sample = sample / np.where(norms == 0, 1.0, norms)

    # Full pairwise cosine similarity matrix (n x n)
    sim_matrix = sample @ sample.T

    # Extract upper triangle (exclude diagonal)
    upper_i, upper_j = np.triu_indices(n, k=1)
    sims = sim_matrix[upper_i, upper_j]

    near_dup_sims = sims[sims > sim_threshold]
    if len(near_dup_sims) == 0:
        print(f"Empirical near-dup angle: no pairs found with sim > {sim_threshold}")
        return float("nan")

    # Clamp to [-1, 1] before arccos to guard against float rounding
    angles_deg = np.degrees(np.arccos(np.clip(near_dup_sims, -1.0, 1.0)))
    mean_angle = float(np.mean(angles_deg))

    print(f"Empirical near-dup angle (sim > {sim_threshold}):")
    print(f"  pairs found : {len(near_dup_sims)} / {len(sims)} total upper-tri pairs")
    print(f"  angle range : [{angles_deg.min():.2f}°, {angles_deg.max():.2f}°]")
    print(f"  mean angle  : {mean_angle:.2f}°  ← use this in theory formulas")
    return mean_angle


def participation_ratio(embeddings: np.ndarray) -> float:
    """
    Compute the participation ratio (effective dimensionality) of the embedding matrix.

    PR = (sum S_i)^2 / sum(S_i^2)

    A low PR relative to the ambient dimension reveals that the embeddings
    live on a low-dimensional manifold.
    """
    E = embeddings.astype(np.float64)
    _, S, _ = np.linalg.svd(E, full_matrices=False)
    pr = float(S.sum() ** 2 / (S ** 2).sum())
    print(f"Participation ratio: {pr:.1f}  (ambient dim = {E.shape[1]})")
    return pr


# ------------------------------------------------------------------
# Smoke tests
# ------------------------------------------------------------------

if __name__ == "__main__":
    # --- test 1: synthetic stream shape and label counts ---
    vecs, labels = synthetic_stream(n_clusters=5, cluster_size=4, dim=16)
    assert len(vecs) == 20,   f"expected 20 vectors, got {len(vecs)}"
    assert len(labels) == 20, f"expected 20 labels, got {len(labels)}"
    assert sum(labels) == 5,  f"expected 5 novel labels, got {sum(labels)}"
    # All vectors should be unit norm
    norms = [abs(np.linalg.norm(v) - 1.0) for v in vecs]
    assert max(norms) < 1e-9, "all vectors should be unit norm"

    # --- test 2: within-cluster cosine similarity should be high ---
    vecs2, labels2 = synthetic_stream(n_clusters=3, cluster_size=10, dim=32,
                                       noise_sigma=0.02, seed=1)
    # Find the first novel vector and check its near-dups are very similar
    # (before shuffling they'd be adjacent, but after shuffle we just check
    #  that there exist high-similarity pairs)
    mat = np.array(vecs2)
    sims = mat @ mat.T
    np.fill_diagonal(sims, 0.0)
    assert sims.max() > 0.9, "expected at least one near-duplicate pair"

    # --- test 3: label fraction matches cluster structure ---
    vecs3, labels3 = synthetic_stream(n_clusters=10, cluster_size=20, dim=50)
    assert sum(labels3) == 10, "exactly 10 novel vectors expected"
    assert len(labels3) == 200

    # --- test 4: empirical_near_dup_angle on synthetic embeddings ---
    rng = np.random.default_rng(7)
    # build tight cluster to guarantee near-dup pairs
    center = rng.standard_normal(32); center /= np.linalg.norm(center)
    tight = center + rng.standard_normal((50, 32)) * 0.02
    norms_ = np.linalg.norm(tight, axis=1, keepdims=True)
    tight /= norms_
    angle = empirical_near_dup_angle(tight, sim_threshold=0.8, n_sample=50)
    assert not np.isnan(angle), "expected near-dup pairs in tight cluster"
    assert 0 <= angle <= 30,    f"expected small angle, got {angle:.2f}°"

    # --- test 5: participation_ratio ---
    low_rank = np.zeros((100, 32))
    low_rank[:, :3] = rng.standard_normal((100, 3))  # only 3 active dims
    pr = participation_ratio(low_rank)
    assert pr <= 5, f"low-rank matrix should have small PR, got {pr:.2f}"

    print("\nAll stream.py smoke tests passed.")
