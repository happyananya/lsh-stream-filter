## Directly useful

### Lecture 6 — LSH and Indyk-Motwani analysis (must engage)

You're already using LSH. The Indyk-Motwani analysis in particular gives you the formal language to describe what your bucket-occupancy policy is doing in expectation. The classical analysis shows Pr[h(x) = h(y)] = f(d(x,y)) for various hash families. You can extend this to:

"Given a stream with redundancy structure characterized by [some distribution], the expected occupancy of a bucket containing item x at time t is..."

This gives you a theoretical memory bound to complement your empirical one. Your Phase 2 plot shows |M| flatlines at 100K. The theory tells you why it flatlines at exactly that value as a function of K, L, T, and the input distribution. That's a real strengthening.
This is the most directly useful lecture for the paper.

### Lecture 1 — Concentration, Markov, Chebyshev

These give you the tools to put confidence intervals on bucket occupancy. Your method makes a decision based on O(x) = median bucket count. Concentration inequalities tell you how much that decision can vary across runs/seeds.
Practical use: a one-paragraph theory section saying "By Chebyshev, the deviation of bucket occupancy from its expectation is bounded by..., which means the retention decision is correct with probability at least 1-δ when..." Reviewers love this kind of thing because it converts a heuristic into a principled method.

### Lecture 5 — Johnson-Lindenstrauss

This is almost directly relevant. JL says you can project d-dimensional vectors to O(log n / ε²) dimensions while preserving distances. Practical implication for you: you might be able to use much smaller embeddings (d=384 → d=64 or so) without harming retention quality, dramatically reducing memory.
Worth one experiment: apply JL projection before hashing, measure the recall/memory tradeoff. If it works, your memory advantage over baselines compounds (smaller per-item footprint × better retention). If not, you've established that 384-d is necessary, which is itself a finding.
Useful for evaluation, not method

### Lecture 4 — High-dimensional geometry

The "curse of dimensionality" content here is critical for your failure analysis. In high dimensions, distances between random points concentrate around the mean. This is exactly why your bucket histograms look peaked rather than long-tailed — distance concentration means most items end up in similar-occupancy buckets.
You could write a paragraph framing your "negative" findings (no advantage on TopicDrift, peaked rather than long-tail occupancy) in terms of distance concentration. That converts unexplained results into theoretically-anticipated ones, which is much stronger.

## Concrete recommendations

If you want to actually use this material, here's what I'd prioritize:

### Easy wins (1–2 paragraphs of theory each):

Use concentration inequalities (Lec 1, 3) to bound the variance of your retention decision. Add a short "theoretical analysis" subsection to your paper. Even a Chebyshev-level bound elevates the contribution from "heuristic that works empirically" to "principled method with guarantees."

### Medium experiment (2–3 days):

Apply JL projection (Lec 5) before hashing. Test whether you can compress embeddings 4–6× without harming retention. If yes, this is a free memory win on top of everything else. If no, document the floor.

### Larger experiment (2 weeks, only if pursuing the future-work direction):

Build adaptive hyperplanes using online gradient methods (Lec 9). Compare static-LSH BucketOccupancy vs. adaptive-LSH BucketOccupancy on TopicDrift. This would directly address the static-projection limitation in your conclusion. Big payoff if it works.

## Jaccard usage in this project

We use Jaccard as an alternative `aggregator` mode in `BucketOccupancyRetention` (see `src/jaccard_tests.py`) to score redundancy by bucket-signature overlap rather than raw bucket counts.

- **Signature definition**: each item gets an `L`-length bucket signature (one bucket id per table).
- **Candidate generation**: for a new item, we use an inverted index keyed by `(table_idx, bucket_id)` to collect retained items that share at least one bucket.
- **Similarity score**: for each candidate, we count matching table positions and compute:
  - `J = matches / (2L - matches)`
  - this is Jaccard on the set-like representation of the two `L`-table signatures.
- **Occupancy replacement**: in `aggregator='jaccard'` mode, the policy's "occupancy" value is `max_jaccard` (maximum similarity to any candidate).
- **Retention behavior**: high `max_jaccard` means the incoming item is highly redundant; low `max_jaccard` means more novel, so it is more likely to be retained under threshold/capacity logic.
- **Efficiency detail**: the inverted index is only maintained in Jaccard mode to avoid extra overhead for non-Jaccard aggregators.