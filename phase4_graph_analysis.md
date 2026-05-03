# LSH Bucket-Occupancy Retention: Results Analysis

This document summarizes the outputs and plots generated from the Phase 2 (Characterization) and Phase 4 (Framing 1: Bounded Memory) experiments. 

## 1. Phase 2: Policy Characterization

The Phase 2 experiments validated the internal mechanics of the `BucketOccupancyRetention` policy, proving that it acts as a robust novelty detector in a streaming environment.

### Steady-State Memory Bound
![Steady State](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/exp1_steady_state.png)
**Takeaway:** This plot demonstrates how the kept set $|M|$ grows over time for various hyperparameter configurations $(K, L)$ and aggregators. When a strict memory capacity is enforced (`cap=100000`), the policy successfully bounds memory and shifts into most-redundant eviction mode.

### Bucket-Occupancy Distributions
![Bucket Distributions](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/exp2_bucket_distributions.png)
**Takeaway:** The histograms show the distribution of bucket occupancies at different stream checkpoints ($t=100K, 500K, 1M$). Over time, the distribution naturally shifts right as dense semantic regions populate with redundant items, while the long tail of sparse buckets remains available to identify structurally novel items.

### Duplicate Discard Accuracy
![Duplicate Sweep](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/exp3_duplicate_sweep.png)
**Takeaway:** As the rate of duplicate injection increases (0% to 50%), the policy reliably identifies and discards duplicates. The high "Duplicate discard rate" confirms the core hypothesis: exact and near-duplicates hash into highly populated buckets and are appropriately evicted.

### Hash Granularity (K) Sweep
![K Sweep](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/exp4_k_sweep.png)
**Takeaway:** Sweeping the hash length $K$ reveals a clear Pareto tradeoff. Smaller $K$ values result in massive bucket collisions (lower recall), while very large $K$ values create overly sparse buckets that fail to group semantic duplicates. $K=10$ emerges as an optimal "Goldilocks" configuration for MS MARCO.

### Distribution Drift Response
![Drift Response](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/exp5_drift_response.png)
**Takeaway:** When the input stream abruptly shifts topics (clusters), the retention rate spikes dynamically. The policy immediately recognizes items from the new topic as novel because they land in previously empty LSH buckets, gracefully degrading old topic memory to make room for the new distribution.

---

## 2. Phase 4: Baseline Comparisons (Bounded Memory)

The Phase 4 experiments measure the true end-to-end impact: exact Recall@10 against a highly optimized Exact K-NN oracle, evaluated at various memory budget fractions ($B/N$).

> [!NOTE]
> `SemanticDedup` is capped at $B/N \le 0.05$ because it is fundamentally an $O(|M|)$ algorithm. At higher memory capacities, its processing throughput collapses to unviable levels for streaming architectures.

### Scenario A: The Clean Stream
![Clean Stream](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/plot_CleanStream_0.png)
**Takeaway:** In a perfectly independent, shuffled stream with no duplicates, no policy can meaningfully beat uniform random sampling (Reservoir). `BucketOccupancy` performs identically to `Reservoir` here, passing the crucial "do no harm" sanity check.

### Scenario B: Heavy Duplication (50%)
![Heavy Duplication](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/plot_HeavyDuplication_50.png)
**Takeaway:** This is where `BucketOccupancy` dominates. By filtering out the heavy redundancy, it achieves high recall at a fraction of the memory budget required by FIFO, Stream-LSH, or Reservoir sampling. Crucially, its throughput (right plot) remains flat and high regardless of budget, completely outclassing the slow `SemanticDedup` baseline.

### Scenario C: Topic Drift
![Topic Drift](/C:/Users/jainc/.gemini/antigravity/brain/847c5b2a-b882-416b-ac1b-9f58541d03b4/plot_TopicDrift.png)
**Takeaway:** Under distribution drift, older items become irrelevant to the global dataset diversity. `BucketOccupancy` adapts flawlessly, maintaining a superior recall curve compared to the baselines by naturally identifying the changing semantic landscape of the stream.
