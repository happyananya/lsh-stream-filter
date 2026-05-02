"""
Unit tests confirming each policy respects the stated memory bound.
Fulfills Phase 3 deliverable: "Unit tests confirming each policy respects the stated memory bound"
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from baselines import FIFORetention, ReservoirSamplingRetention, RandomSamplingRetention, SemanticDedupRetention, StreamLSHRetention
from retention_policies import BucketOccupancyRetention


def test_memory_bounds():
    N = 1000  # Stream size
    B = 100   # Memory budget (capacity)
    dim = 64
    
    # Create random dummy stream
    rng = np.random.RandomState(42)
    stream = rng.randn(N, dim).astype(np.float32)
    stream = stream / np.linalg.norm(stream, axis=1, keepdims=True)
    
    policies = [
        ('FIFO', FIFORetention(dim=dim, capacity=B)),
        ('Reservoir', ReservoirSamplingRetention(dim=dim, capacity=B)),
        ('Random', RandomSamplingRetention(dim=dim, capacity=B, stream_size=N)),
        ('SemanticDedup', SemanticDedupRetention(dim=dim, capacity=B, epsilon=0.1)),
        ('Stream-LSH', StreamLSHRetention(dim=dim, capacity=B)),
        ('BucketOccupancy', BucketOccupancyRetention(dim=dim, L=4, K=8, capacity=B, aggregator='median'))
    ]
    
    print(f"Testing memory bounds for {len(policies)} policies (Stream N={N}, Budget B={B})...")
    
    all_passed = True
    for name, policy in policies:
        for i in range(N):
            policy.insert(stream[i], item_id=i)
        
        kept = policy.kept_count()
        
        # Semantic dedup and Random can retain LESS than B, but no policy should retain MORE than B.
        # Random expects ~B, but variance is high.
        passed = (kept <= B)
        
        if not passed:
            all_passed = False
            print(f"  [FAIL] {name}: retained {kept} items (Budget={B})")
        else:
            print(f"  [PASS] {name}: retained {kept} items (<= {B})")
            
    if all_passed:
        print("\nAll policies strictly respect the memory bound!")
    else:
        print("\nWARNING: Some policies exceeded the memory bound.")


if __name__ == "__main__":
    test_memory_bounds()
