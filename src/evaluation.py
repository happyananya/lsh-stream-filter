"""
Evaluation Harness for Retention Policies
==========================================
Streams a dataset through a policy, then evaluates recall@k 
of the retained set against the oracle ground truth.
"""
import time
import numpy as np
import faiss


def stream_through_policy(policy, stream: np.ndarray, source_ids: np.ndarray, report_interval: int = 100000):
    """
    Stream embeddings through a retention policy one at a time.
    Returns timing stats.
    """
    n = len(stream)
    t0 = time.time()
    kept_count = 0
    
    for i in range(n):
        result = policy.insert(stream[i], item_id=source_ids[i])
        if result:
            kept_count += 1
        
        if (i + 1) % report_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Streamed {i+1:,}/{n:,} | Kept: {policy.kept_count():,} | "
                  f"Rate: {rate:,.0f} items/s")
    
    elapsed = time.time() - t0
    return {
        'stream_size': n,
        'kept': policy.kept_count(),
        'retention_rate': policy.kept_count() / n,
        'throughput_items_per_sec': n / elapsed,
        'total_time_s': elapsed,
    }


def evaluate_recall(policy, queries: np.ndarray, oracle_gt: np.ndarray, 
                    k: int = 10, metric: str = 'ip'):
    """
    Evaluate recall@k of the policy's retained set against oracle ground truth.
    
    Uses FAISS exact search on the retained set.
    Returns per-query recalls and mean recall.
    """
    kept_emb, kept_ids = policy.kept_set()
    
    if len(kept_emb) == 0:
        return {'recall': 0.0, 'per_query_recall': np.zeros(len(queries))}
    
    # Build FAISS index on the retained set
    d = kept_emb.shape[1]
    if metric == 'ip':
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    
    index.add(kept_emb.astype(np.float32))
    
    # Search
    _, I = index.search(queries.astype(np.float32), k)
    
    # Map local indices back to global item IDs
    per_query_recalls = []
    for q_idx in range(len(queries)):
        # Retrieved IDs (mapped from local to global)
        retrieved_global = set()
        for local_idx in I[q_idx]:
            if local_idx >= 0 and local_idx < len(kept_ids):
                retrieved_global.add(kept_ids[local_idx])
        
        # Oracle IDs
        oracle_ids = set(oracle_gt[q_idx, :k])
        
        # Recall
        overlap = len(retrieved_global & oracle_ids)
        per_query_recalls.append(overlap / k)
    
    per_query_recalls = np.array(per_query_recalls)
    
    return {
        'recall': float(np.mean(per_query_recalls)),
        'recall_std': float(np.std(per_query_recalls)),
        'recall_p10': float(np.percentile(per_query_recalls, 10)),
        'recall_p50': float(np.percentile(per_query_recalls, 50)),
        'recall_p90': float(np.percentile(per_query_recalls, 90)),
        'per_query_recall': per_query_recalls,
    }


def run_experiment(policy, stream: np.ndarray, source_ids: np.ndarray, queries: np.ndarray,
                   oracle_gt: np.ndarray, policy_name: str,
                   k: int = 10, metric: str = 'ip'):
    """
    Full experiment: stream → retain → evaluate.
    Returns a results dict suitable for CSV logging.
    """
    print(f"\n{'='*60}")
    print(f"Policy: {policy_name}")
    print(f"{'='*60}")
    
    # Stream
    stream_stats = stream_through_policy(policy, stream, source_ids)
    print(f"  Final: Kept {stream_stats['kept']:,} / {stream_stats['stream_size']:,} "
          f"({stream_stats['retention_rate']:.2%})")
    print(f"  Throughput: {stream_stats['throughput_items_per_sec']:,.0f} items/s")
    
    # Evaluate
    print(f"  Evaluating recall@{k}...")
    recall_stats = evaluate_recall(policy, queries, oracle_gt, k=k, metric=metric)
    print(f"  Recall@{k}: {recall_stats['recall']:.4f} "
          f"(p10={recall_stats['recall_p10']:.4f}, p50={recall_stats['recall_p50']:.4f}, "
          f"p90={recall_stats['recall_p90']:.4f})")
    
    # Combine
    result = {
        'policy': policy_name,
        **stream_stats,
        'recall_at_k': recall_stats['recall'],
        'recall_std': recall_stats['recall_std'],
        'recall_p10': recall_stats['recall_p10'],
        'recall_p50': recall_stats['recall_p50'],
        'recall_p90': recall_stats['recall_p90'],
    }
    
    # Add policy-specific stats if available
    extra = policy.stats()
    for key, val in extra.items():
        if key not in result:
            result[f'policy_{key}'] = val
    
    return result, recall_stats['per_query_recall']
