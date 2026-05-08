# Research Paper Plan: LSH Stream Filter

This planning document is scoped to two immediate report sections:
1. `Literature Review`
2. `Theoretical Methodology Complementing Results`

It is written to help you move from repo artifacts to paper-ready narrative, with explicit references to which files to read and what conclusions to extract.

---

## Literature Review

### 1) Problem Framing in Prior Work

**Goal in paper:** Position bounded streaming memory as a constrained optimization problem (`|M| <= B`) where the objective is semantic coverage and recall under latency constraints.

**What to read in this repo first**
- `Comprehensive_Research_Report.md`
- `LSH_Stream_Filter_Research_Report.md`
- `experiments/benchmark_harness.py`
- `src/evaluation.py`

**What to infer and write**
- Prior approaches implicitly optimize one of three axes: recency (FIFO), randomness (Reservoir), or exact semantic filtering (Semantic Dedup).
- The key literature gap your project addresses: no method combines semantic-awareness with sublinear (non-`O(|M|)`) insertion cost suitable for real-time streams.

---

### 2) Baseline Families and Their Assumptions

**Goal in paper:** Build a principled baseline taxonomy before presenting your method.

**Primary implementation files**
- `src/baselines.py`
- `experiments/phase4_lsh.py`
- `experiments/phase4_bounded_memory.py`

**Result files to cite**
- `results/phase4_framing1/phase4_sweep_CleanStream_0.csv`
- `results/phase4_framing1/phase4_sweep_HeavyDuplication_50.csv`
- `results/phase4_framing1/phase4_sweep_TopicDrift.csv`

**What to infer and write**
- **FIFO** encodes a recency prior; good when query relevance drifts with time.
- **Reservoir** provides unbiased sample coverage but no intentional semantic anti-redundancy.
- **Semantic Dedup** is an accuracy-oriented oracle-like baseline but computationally expensive.
- **Stream-LSH-like methods** can collapse to recency behavior in ingestion-only settings without external quality signals.

---

### 3) LSH for Novelty and Diversity Retention

**Goal in paper:** Explain why LSH is appropriate for fast novelty estimation in embedding streams.

**Primary method files**
- `src/retention_policies.py`
- `src/jaccard_tests.py`
- `class_reference.md`

**Support files**
- `experiments/phase2_characterization.py`
- `results/phase2_characterization/exp1_steady_state.csv`
- `results/phase2_characterization/exp3_duplicate_sweep.csv`
- `results/phase2_characterization/exp4_k_sweep.csv`

**What to infer and write**
- LSH bucket occupancy serves as a proxy for local density (redundancy signal).
- Novelty decisions can be made in `O(L)` hashing/aggregation time plus heap update, avoiding per-item full-memory scans.
- Jaccard-over-signatures extends occupancy into an overlap-aware redundancy metric when bucket-count-only aggregation is insufficient.

---

### 4) Known Limits from Literature (for Honest Positioning)

**Goal in paper:** Preempt reviewer criticism with explicit known failure regimes.

**Files/results to anchor this subsection**
- `results/phase4_framing1/phase4_sweep_TopicDrift.csv`
- `results/phase4_framing1/plots/plot_TopicDrift.png`
- `Comprehensive_Research_Report.md`

**What to infer and write**
- In primarily drift-driven streams with weak duplication, recency can outperform diversity-oriented retention.
- This aligns with known tradeoff: maximizing global semantic diversity is not always equivalent to maximizing query-time relevance under temporal drift.
- State this as boundary condition, not failure of method design.

---

## Theoretical Methodology Complementing Results

### 1) Formal Problem Statement and Objective

**Goal in paper:** Define the mathematical objective before empirical sections.

**Use directly**
- Problem statement from your prompt and existing intro in `Comprehensive_Research_Report.md`

**What to formalize**
- Stream: `x_1, x_2, ...` in embedding space.
- Memory: `M_t`, constrained by `|M_t| <= B`.
- Policy objective: maximize retained semantic coverage / recall surrogate under bounded insertion time.
- Define novelty score `s(x_t)` from LSH signatures and describe keep/evict decision rule.

---

### 2) Algorithmic Complexity and Systems Argument

**Goal in paper:** Convert implementation choices into asymptotic and practical efficiency claims.

**Files to inspect**
- `src/retention_policies.py`
- `src/jaccard_tests.py`
- `experiments/plot_throughput_comparison.py`
- `results/phase4_framing1/plot_throughput_comparison.png`

**What to infer and write**
- Per-item operations are dominated by hashing (`O(L*K)` bit-projection style), occupancy aggregation (`O(L)`), and heap maintenance (`O(log B)` when full).
- Contrast with exact semantic dedup requiring nearest-neighbor search against retained memory (effectively `O(|M|)` behavior in this setup).
- Tie theory to observed throughput gap to support practical scalability claim.

---

### 3) Collision-Probability Intuition (Indyk-Motwani Link)

**Goal in paper:** Add theoretical credibility for using bucket collisions as semantic similarity evidence.

**Files to use**
- `class_reference.md`
- `results/phase2_characterization/plots/exp2_bucket_distributions.png`
- `results/phase2_characterization/plots/exp3_duplicate_sweep.png`

**What to infer and write**
- For LSH families, collision probability increases with similarity; occupancy therefore estimates local density in semantic space.
- Show empirical consistency:
  - duplicates map to denser signatures and are filtered at high rates,
  - retained set occupancy is flattened under bounded memory due to redundancy-aware eviction.
- Phrase this as "theory-guided heuristic with empirical validation."

---

### 4) Concentration and Stability of Retention Decisions

**Goal in paper:** Add statistical framing for robustness across runs and seeds.

**Files to use**
- `results/phase2_characterization/exp4_k_sweep.csv`
- `results/phase2_characterization/exp5_drift_response.csv`
- any multi-seed outputs you have in `results/` (if available)

**What to infer and write**
- Bound variance of occupancy-based novelty estimate using concentration-style arguments (Chebyshev-level is acceptable if lightweight).
- Explain that stable trends across `K` and scenarios indicate decision rule robustness, even if exact constants are not derived.
- Include this as a "practical guarantee" subsection rather than a full theorem if proofs are incomplete.

---

### 5) Theory-Result Alignment by Scenario

**Goal in paper:** Explicitly connect each major empirical result to a theoretical expectation.

**Files to use**
- `results/phase4_framing1/phase4_sweep_CleanStream_0.csv`
- `results/phase4_framing1/phase4_sweep_HeavyDuplication_50.csv`
- `results/phase4_framing1/phase4_sweep_TopicDrift.csv`
- `results/phase5_locomo/phase5_locomo_results.csv`
- `results/phase6_jl/phase6_jl_results.csv`

**What to infer and write**
- **Clean stream:** natural cluster imbalance still creates redundancy; occupancy balancing improves representative coverage.
- **Heavy duplication:** method should dominate because redundancy signal is strongest; observed near-perfect recall at moderate budgets supports this.
- **Topic drift:** recency prior can beat density prior; matches theory about objective mismatch under temporal non-stationarity.
- **LoCoMo:** semantic milestone retention improves downstream QA at constrained budgets.
- **JL projection:** low-dimensional compression increases capacity under fixed bytes, improving recall in tight-budget regime until distortion ceiling appears.

---

### 6) Threats to Validity and Theoretical Limits

**Goal in paper:** Strengthen credibility via explicit limitations.

**Artifacts to reference**
- `Comprehensive_Research_Report.md`
- `results/phase4_framing1/*.csv`
- `results/phase6_jl/phase6_jl_results.csv`

**What to infer and write**
- Dependence on embedding quality and stationarity assumptions.
- Sensitivity to hash hyperparameters (`L`, `K`) and projection distortion (`d` in JL).
- Recall metric dependence on oracle/query construction.
- Jaccard aggregator compute overhead vs median occupancy variant.

---

## Immediate Writing Workflow (Practical Execution Plan)

### Step 1: Lock citations and evidence map
- Create a two-column scratch table: `Claim` -> `Evidence file/figure`.
- Ensure every non-trivial claim in these two sections points to at least one CSV or plot in `results/`.

### Step 2: Draft Literature Review (first pass, 1-2 pages)
- Follow subsection order above.
- End with a short "Gap and Contribution" paragraph that sets up your method.

### Step 3: Draft Theoretical Methodology (first pass, 2-3 pages)
- Write formalism + complexity first.
- Then add scenario-wise theory-result alignment.
- Keep proofs lightweight but explicit about assumptions.

### Step 4: Add quantitative anchors
- Pull exact numbers from:
  - `results/phase4_framing1/*.csv`
  - `results/phase5_locomo/phase5_locomo_results.csv`
  - `results/phase6_jl/phase6_jl_results.csv`
- Avoid qualitative claims without a numeric anchor.

### Step 5: Final consistency check
- Verify notation consistency (`M`, `B`, `L`, `K`, `d`, recall metrics) against `Comprehensive_Research_Report.md`.
- Ensure all negative results (Topic Drift) are retained; do not overstate generality.

---

## Suggested Next File to Edit

Use this plan to populate new sections directly in:
- `LSH_Stream_Filter_Research_Report.md` (if this is your submission draft), or
- `Comprehensive_Research_Report.md` (if you are consolidating into one canonical report).
