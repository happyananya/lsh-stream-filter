"""
metrics_addons.py

Drop-in metrics for the BucketOccupancy retention experiments.
Designed to add to an existing pipeline without re-running the stream.

Required inputs (you should already have these):
    - kept_set: np.ndarray of shape (M, d), float32, the policy's retained embeddings
    - kept_ids: np.ndarray of shape (M,), int — original stream positions of kept items
    - queries: np.ndarray of shape (Q, d), float32
    - oracle_topk: np.ndarray of shape (Q, K_max), int — original stream IDs of oracle top-k
                   (computed once against the full unfiltered stream, K_max >= 100)
    - per_item_latencies_ns: np.ndarray of shape (N,), int64 — nanoseconds per insert call
    - full_stream: np.ndarray of shape (N, d), float32 — the original stream (for diversity metrics)

All metrics are pure functions. Run them after each (policy, budget, dataset) experiment.
"""

import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from typing import Dict


# =============================================================================
# TIER 1 — Recall and latency improvements
# =============================================================================

def _get_unique_topk(index, queries: np.ndarray, kept_ids: np.ndarray, k: int) -> np.ndarray:
    """
    Helper to fetch top-k unique global IDs.
    FAISS might return duplicate items (if the stream had duplicates).
    We search deeper and filter to unique global IDs.
    """
    search_k = min(k * 10, index.ntotal)
    if search_k == 0:
        return np.zeros((len(queries), 0), dtype=np.int64)
        
    _, I = index.search(queries.astype(np.float32), search_k)
    
    unique_global = []
    for q_idx in range(len(queries)):
        q_unique = []
        for local_idx in I[q_idx]:
            if local_idx >= 0 and local_idx < len(kept_ids):
                g_id = kept_ids[local_idx]
                if g_id not in q_unique:
                    q_unique.append(g_id)
                if len(q_unique) == k:
                    break
        unique_global.append(q_unique)
    
    # Pad to shape (Q, k) just in case
    padded = np.zeros((len(queries), k), dtype=np.int64) - 1
    for i, row in enumerate(unique_global):
        padded[i, :len(row)] = row
    return padded


def recall_at_k_distribution(
    kept_set: np.ndarray,
    kept_ids: np.ndarray,
    queries: np.ndarray,
    oracle_topk: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """
    Per-query recall@k, returning the full distribution rather than just the mean.

    Reveals tail behavior: a method with mean recall 0.7 might have most queries at
    0.9 with a tail of 0.0 failures, OR everyone uniformly at 0.7. These are very
    different operationally. Reviewers will ask.

    Returns mean, percentiles, and the fraction of queries with recall == 0
    (catastrophic failures).
    """
    if len(kept_set) == 0:
        return {f'recall@{k}_mean': 0.0, f'recall@{k}_p10': 0.0,
                f'recall@{k}_p50': 0.0, f'recall@{k}_p90': 0.0,
                f'recall@{k}_zero_frac': 1.0}

    # Build flat index over kept set
    index = faiss.IndexFlatIP(kept_set.shape[1])
    index.add(kept_set.astype(np.float32))
    
    kept_topk_global = _get_unique_topk(index, queries, kept_ids, k)

    # Per-query recall: |kept_topk ∩ oracle_topk[:k]| / k
    oracle_truncated = oracle_topk[:, :k]
    per_query = np.array([
        len(set(kept_topk_global[i]) & set(oracle_truncated[i])) / k
        for i in range(len(queries))
    ])

    return {
        f'recall@{k}_mean': float(per_query.mean()),
        f'recall@{k}_p10': float(np.percentile(per_query, 10)),
        f'recall@{k}_p50': float(np.percentile(per_query, 50)),
        f'recall@{k}_p90': float(np.percentile(per_query, 90)),
        f'recall@{k}_zero_frac': float((per_query == 0).mean()),
        'per_query_recall': per_query,  # keep for plotting CDFs
    }


def recall_at_multiple_k(
    kept_set: np.ndarray,
    kept_ids: np.ndarray,
    queries: np.ndarray,
    oracle_topk: np.ndarray,
    ks=(1, 10, 100),
) -> Dict[int, Dict[str, float]]:
    """
    Recall at multiple k values in a single pass. Different k values measure
    different things:
      k=1   : did we keep the single most relevant memory?
      k=10  : balanced view, the standard headline
      k=100 : did the kept set broadly cover query-relevant regions?

    BucketOccupancy should excel at k=100 (broad coverage) more than k=1
    (specific items). If k=1 looks weak but k=100 looks strong, that's a
    coherent story: the method preserves *diversity*, not *exact items*.
    """
    if len(kept_set) == 0:
        return {k: {'mean': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0} for k in ks}

    max_k = max(ks)
    index = faiss.IndexFlatIP(kept_set.shape[1])
    index.add(kept_set.astype(np.float32))
    
    kept_topk_global = _get_unique_topk(index, queries, kept_ids, max_k)

    out = {}
    for k in ks:
        per_query = np.array([
            len(set(kept_topk_global[i, :k]) & set(oracle_topk[i, :k])) / k
            for i in range(len(queries))
        ])
        out[k] = {
            'mean': float(per_query.mean()),
            'p10': float(np.percentile(per_query, 10)),
            'p50': float(np.percentile(per_query, 50)),
            'p90': float(np.percentile(per_query, 90)),
            'zero_frac': float((per_query == 0).mean()),
        }
    return out


def query_relevant_coverage(
    kept_ids: np.ndarray,
    oracle_topk: np.ndarray,
    k: int = 10,
) -> float:
    """
    Fraction of the union of all queries' oracle top-k that the kept set retains.

    Complement to per-query recall. Per-query recall asks "for query q,
    did we keep its top-k?" Coverage asks "across all queries, what fraction
    of the regions queries actually probe did we cover?"

    Reservoir sampling does well on this (uniform). BucketOccupancy should
    do better on heavy-duplication streams because it stops wasting space
    on dense regions and starts covering sparse ones.

    A value of 1.0 means every item that any query wanted is in the kept set.
    """
    relevant_union = set()
    for i in range(oracle_topk.shape[0]):
        relevant_union.update(oracle_topk[i, :k].tolist())
    if not relevant_union:
        return 0.0
    kept_set = set(kept_ids.tolist())
    return len(relevant_union & kept_set) / len(relevant_union)


def latency_distribution(per_item_latencies_ns: np.ndarray) -> Dict[str, float]:
    """
    Latency distribution per insertion. Mean throughput hides eviction tail
    behavior — heap operations are O(log M) but cache-miss-heavy at large M,
    and a sudden cluster of redundant items can trigger many evictions back-to-back.

    p99 is the production-relevant number. p999 catches pathological behavior
    that mean and p99 both hide.
    """
    arr = np.asarray(per_item_latencies_ns)
    return {
        'mean_us': float(arr.mean() / 1000),
        'p50_us': float(np.percentile(arr, 50) / 1000),
        'p95_us': float(np.percentile(arr, 95) / 1000),
        'p99_us': float(np.percentile(arr, 99) / 1000),
        'p999_us': float(np.percentile(arr, 99.9) / 1000),
        'max_us': float(arr.max() / 1000),
        # Ratio reveals tail-heaviness; healthy methods sit near 1.0
        'tail_ratio': float(np.percentile(arr, 99) / np.percentile(arr, 50)),
    }


# =============================================================================
# TIER 2 — Diversity metrics (the direct measurement of the method's claim)
# =============================================================================

def k_center_radius(
    kept_set: np.ndarray,
    full_stream: np.ndarray,
    sample_size: int = 50_000,
    seed: int = 0,
) -> float:
    """
    Worst-case distance from any original-stream item to its nearest representative
    in the kept set. Lower is better. Directly measures coverage of the embedding space.

    For 1M-item streams, sample 50K random items rather than computing exhaustively.
    Sampling is unbiased and 20x faster.
    """
    if len(kept_set) == 0:
        return float('inf')

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(full_stream), size=min(sample_size, len(full_stream)),
                            replace=False)
    sample = full_stream[sample_idx].astype(np.float32)

    index = faiss.IndexFlatL2(kept_set.shape[1])
    index.add(kept_set.astype(np.float32))
    distances, _ = index.search(sample, 1)
    # FAISS returns squared L2; take sqrt for interpretability
    return float(np.sqrt(distances.max()))


def cluster_coverage(
    kept_ids: np.ndarray,
    cluster_assignments: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    """
    Of the clusters in the original stream, how many have at least one
    representative in the kept set?

    cluster_assignments: shape (N,) with cluster ID for each stream item.
    Compute ONCE per dataset (k-means on the original stream), reuse across
    all policy/budget combinations.

    BucketOccupancy's advantage on heavy-duplication streams should be most
    visible here: reservoir wastes space on duplicates of dense clusters and
    misses sparse ones, whereas bucket occupancy preserves sparse clusters.
    """
    kept_clusters = cluster_assignments[kept_ids]
    unique_kept = np.unique(kept_clusters)
    return {
        'clusters_covered': int(len(unique_kept)),
        'coverage_fraction': float(len(unique_kept) / n_clusters),
        'cluster_entropy': float(_entropy(kept_clusters, n_clusters)),
    }


def _entropy(labels: np.ndarray, n_clusters: int) -> float:
    """
    Entropy of the kept set's cluster distribution. Higher = more even coverage.
    Maximum entropy is log(n_clusters) when items are uniformly spread.
    """
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    nz = probs[probs > 0]
    return float(-(nz * np.log(nz)).sum())


def precompute_clusters(
    full_stream: np.ndarray,
    n_clusters: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute once per dataset, save to disk, reuse across all experiments.
    MiniBatchKMeans is much faster than KMeans on 1M items and the cluster
    quality difference is negligible for this purpose.
    """
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         batch_size=10_000, n_init=3)
    return km.fit_predict(full_stream)


def mean_intra_set_distance(
    kept_set: np.ndarray,
    sample_size: int = 5_000,
    seed: int = 0,
) -> float:
    """
    Mean nearest-neighbor distance within the kept set. Higher means the kept
    items are more spread out — i.e., more diverse.

    Combine with k_center_radius: k-center is worst-case coverage of the
    *original* stream; intra-set distance is average spread *within* the
    kept set. They measure complementary aspects of diversity.
    """
    if len(kept_set) < 2:
        return 0.0

    rng = np.random.RandomState(seed)
    if len(kept_set) > sample_size:
        idx = rng.choice(len(kept_set), size=sample_size, replace=False)
        sample = kept_set[idx].astype(np.float32)
    else:
        sample = kept_set.astype(np.float32)

    index = faiss.IndexFlatL2(sample.shape[1])
    index.add(sample)
    # k=2 because the nearest neighbor of each point is itself; we want the second
    distances, _ = index.search(sample, 2)
    return float(np.sqrt(distances[:, 1]).mean())


# =============================================================================
# TIER 2b — Cosine-based diversity metrics
# =============================================================================
# Complements the L2-based metrics above. On L2-normalized vectors,
# cosine_sim(a,b) = a · b and ||a-b||² = 2(1 - cos(a,b)), so rankings
# are identical — but cosine is bounded [0,1] and more interpretable
# for reviewers. The aggregation statistics (mean, percentiles) differ
# from the L2 versions due to the non-linear relationship.


def mean_pairwise_cosine_similarity(
    kept_set: np.ndarray,
    sample_size: int = 5_000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Nearest-neighbor cosine similarity distribution within the kept set.

    Lower mean = more diverse (items spread out in angular space).
    Higher mean = more redundant (items cluster together).

    On L2-normalized vectors, cosine_sim(a,b) = a · b  (inner product).

    Complementary to mean_intra_set_distance (L2):
      - L2 distance answers "how far apart are kept items?"
      - Cosine similarity answers "how aligned are kept items?"
    The distinction matters for aggregate statistics because
    mean(sqrt(2 - 2*cos)) ≠ sqrt(2 - 2*mean(cos)).
    """
    if len(kept_set) < 2:
        return {
            'mean_nn_cosine': 0.0,
            'max_nn_cosine': 0.0,
            'p90_nn_cosine': 0.0,
            'cosine_diversity_score': 1.0,
        }

    rng = np.random.RandomState(seed)
    if len(kept_set) > sample_size:
        idx = rng.choice(len(kept_set), size=sample_size, replace=False)
        sample = kept_set[idx].astype(np.float32)
    else:
        sample = kept_set.astype(np.float32)

    # Inner product on L2-normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(sample.shape[1])
    index.add(sample)
    # k=2: first result is self (similarity ≈ 1.0); second is true nearest neighbor
    D, _ = index.search(sample, 2)
    nn_cosines = D[:, 1]

    return {
        'mean_nn_cosine': float(nn_cosines.mean()),
        'max_nn_cosine': float(nn_cosines.max()),
        'p90_nn_cosine': float(np.percentile(nn_cosines, 90)),
        # 1 - mean_nn_cosine: higher = more diverse
        'cosine_diversity_score': float(1.0 - nn_cosines.mean()),
    }


def cosine_coverage_radius(
    kept_set: np.ndarray,
    full_stream: np.ndarray,
    sample_size: int = 50_000,
    seed: int = 0,
) -> float:
    """
    Worst-case angular gap: for sampled stream items, find the nearest
    kept-set representative by cosine similarity, then report
    1 - min_similarity (the largest gap).

    Lower is better (tighter angular coverage).

    Complementary to k_center_radius (L2 worst-case distance):
    both measure coverage, but cosine_coverage_radius is bounded [0, 2]
    and doesn't depend on embedding magnitude — more stable across
    datasets with different scale characteristics.
    """
    if len(kept_set) == 0:
        return 2.0  # max possible gap: 1 - (-1)

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(
        len(full_stream),
        size=min(sample_size, len(full_stream)),
        replace=False,
    )
    sample = full_stream[sample_idx].astype(np.float32)

    index = faiss.IndexFlatIP(kept_set.shape[1])
    index.add(kept_set.astype(np.float32))
    D, _ = index.search(sample, 1)

    # D contains cosine similarities; coverage gap = 1 - similarity
    return float(1.0 - D.min())


def cosine_redundancy_score(
    full_stream: np.ndarray,
    sample_size: int = 50_000,
    seed: int = 0,
) -> float:
    """
    Stream redundancy measured in cosine space.

    Defined as mean nearest-neighbor cosine similarity (excluding self).
    Range [0, 1] for non-negative embeddings, [-1, 1] in general.
    Higher = more redundant.

    Advantage over the L2-based stream_redundancy_score:
    cosine similarity is inherently bounded, so this metric doesn't
    need a fragile sampled-diameter estimate. More stable across
    datasets with different dimensionality and scale.
    """
    if len(full_stream) < 2:
        return 0.0

    rng = np.random.RandomState(seed)
    idx = rng.choice(
        len(full_stream),
        size=min(sample_size, len(full_stream)),
        replace=False,
    )
    sample = full_stream[idx].astype(np.float32)

    index = faiss.IndexFlatIP(sample.shape[1])
    index.add(sample)
    D, _ = index.search(sample, 2)  # k=2; first is self

    return float(D[:, 1].mean())


# =============================================================================
# TIER 3 — Practical metrics for the LLM-memory framing
# =============================================================================

def memory_footprint_bytes(
    kept_set_size: int,
    embedding_dim: int,
    n_hash_tables: int = 8,
    bits_per_signature: int = 10,
    embedding_dtype_bytes: int = 4,  # float32
) -> Dict[str, float]:
    """
    Total memory cost in bytes, not just item count.

    Reviewers and production engineers care about MB, not |M|. With d=384 and fp32,
    each embedding is 1.5 KB. The LSH overhead is negligible by comparison
    (~10 bytes per item with these defaults), but report it for transparency.
    """
    embedding_bytes = kept_set_size * embedding_dim * embedding_dtype_bytes
    # Each item stores its bucket assignment in each of L tables (bits_per_signature bits each)
    hash_metadata_bytes = kept_set_size * n_hash_tables * (bits_per_signature / 8)
    # Bucket count tables (sparse, but estimate dense upper bound)
    bucket_count_bytes = n_hash_tables * (2 ** bits_per_signature) * 4  # int32 counts

    total = embedding_bytes + hash_metadata_bytes + bucket_count_bytes
    return {
        'embeddings_mb': embedding_bytes / 1e6,
        'hash_metadata_mb': hash_metadata_bytes / 1e6,
        'bucket_counts_mb': bucket_count_bytes / 1e6,
        'total_mb': total / 1e6,
        'overhead_fraction': (hash_metadata_bytes + bucket_count_bytes) / total
                             if total > 0 else 0.0,
    }


# =============================================================================
# TIER 4 — Stream-property diagnostics (compute once per dataset)
# =============================================================================

def stream_redundancy_score(
    full_stream: np.ndarray,
    sample_size: int = 50_000,
    seed: int = 0,
) -> float:
    """
    A scalar characterization of how redundant the stream is. Defined as
    1 - (mean nearest-neighbor distance / dataset diameter). Higher means
    more redundancy.

    Useful for the regime map: "BucketOccupancy beats Reservoir when this
    score exceeds X." Compute once per dataset; report alongside results.
    """
    if len(full_stream) < 2:
        return 0.0

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(full_stream), size=min(sample_size, len(full_stream)),
                     replace=False)
    sample = full_stream[idx].astype(np.float32)

    index = faiss.IndexFlatL2(sample.shape[1])
    index.add(sample)
    distances, _ = index.search(sample, 2)  # k=2; first is self
    mean_nn_dist = np.sqrt(distances[:, 1]).mean()

    # Dataset diameter: sample-based estimate. Pick random pairs.
    pair_idx_a = rng.choice(len(sample), size=1000, replace=False)
    pair_idx_b = rng.choice(len(sample), size=1000, replace=False)
    pair_distances = np.linalg.norm(sample[pair_idx_a] - sample[pair_idx_b], axis=1)
    diameter_estimate = pair_distances.max()

    return float(1.0 - mean_nn_dist / diameter_estimate)


# =============================================================================
# Reporting helpers
# =============================================================================

def evaluate_policy_complete(
    kept_set: np.ndarray,
    kept_ids: np.ndarray,
    full_stream: np.ndarray,
    queries: np.ndarray,
    oracle_topk: np.ndarray,
    cluster_assignments: np.ndarray,
    n_clusters: int,
    per_item_latencies_ns: np.ndarray,
    embedding_dim: int,
    n_hash_tables: int = 8,
    bits_per_signature: int = 10,
) -> Dict:
    """
    One call, all metrics. Use after every (policy, budget, dataset) experiment.

    Returns a flat dict suitable for a pandas DataFrame row.
    """
    out = {}

    # Tier 1
    rec_dist = recall_at_k_distribution(kept_set, kept_ids, queries, oracle_topk, k=10)
    rec_dist.pop('per_query_recall', None)  # don't store the array in the row
    out.update(rec_dist)

    multi_k = recall_at_multiple_k(kept_set, kept_ids, queries, oracle_topk, ks=(1, 10, 100))
    for k, m in multi_k.items():
        out[f'recall@{k}_mean'] = m['mean']
        out[f'recall@{k}_p10'] = m['p10']
        out[f'recall@{k}_zero_frac'] = m['zero_frac']

    out['coverage@10'] = query_relevant_coverage(kept_ids, oracle_topk, k=10)
    out['coverage@100'] = query_relevant_coverage(kept_ids, oracle_topk, k=100)

    out.update(latency_distribution(per_item_latencies_ns))

    # Tier 2 — L2 diversity
    out['k_center_radius'] = k_center_radius(kept_set, full_stream)
    out.update(cluster_coverage(kept_ids, cluster_assignments, n_clusters))
    out['mean_intra_set_distance'] = mean_intra_set_distance(kept_set)

    # Tier 2b — Cosine diversity
    out.update(mean_pairwise_cosine_similarity(kept_set))
    out['cosine_coverage_radius'] = cosine_coverage_radius(kept_set, full_stream)

    # Tier 3
    out.update(memory_footprint_bytes(
        len(kept_set), embedding_dim, n_hash_tables, bits_per_signature
    ))

    # |M| for reference
    out['kept_set_size'] = int(len(kept_set))

    return out