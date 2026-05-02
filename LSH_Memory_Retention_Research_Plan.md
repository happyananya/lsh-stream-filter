# LSH Bucket-Occupancy Retention: A Novelty-Based Memory Policy for LLM Agents

> **Research question.** Can LSH bucket occupancy be used as an *intrinsic novelty signal* for bounded LLM memory? Specifically: if a new item lands in an empty (or sparse) bucket it is novel and we keep it; if it lands in a full bucket it is redundant and we discard it. Does this policy preserve more useful information than alternative bounded-memory policies — including the closest prior work, Stream-LSH (Kraus et al., 2017) — under matched memory budgets?

---

## 0. Scope and framing

### 0.1 What this project is

A study of a **retention policy** for streaming embedding ingestion under bounded memory. The policy uses LSH bucket occupancy as an intrinsic novelty signal — no application-supplied importance, popularity, or freshness scores required. The output of the policy is a kept set of embeddings (and their source items) that approximates the original stream's information content within a memory budget B.

### 0.2 What this project is NOT

- **Not a retrieval-method comparison** (no head-to-head Falconn++ vs. HNSW on standard ANN benchmarks). LSH here is a hashing primitive, not the retrieval backend being optimized.
- **Not a full LLM agent memory system.** We test the *retention component* in isolation. A complete agent memory system would also include retrieval, ranking, and consolidation — out of scope.
- **Not a sketch / approximate-counting structure** (RACE, count-min). We retain raw embeddings, not aggregates, because the goal is to feed the kept items back to an LLM.

### 0.3 Primary contribution

To our knowledge, no prior work uses LSH bucket occupancy itself as the retention signal. Closest prior work:

- **Stream-LSH (Kraus, Carmel, Keidar, IEEE BigData 2017)** — uses LSH for bounded streaming retrieval, but retention is driven by *freshness, application-supplied quality, and dynamic popularity*, not by bucket occupancy.
- **RACE (Coleman, Baraniuk, Shrivastava, ICML 2020)** — sublinear LSH-based sketches for streaming near-neighbor estimation; doesn't retain raw items.
- **Mem0, A-Mem, AMV-L, CraniMem (2024–2025)** — LLM-memory systems that bound memory via *importance scoring + temporal decay + tiering*, not LSH structure.

The contribution is to propose, characterize, and evaluate *bucket-occupancy retention* as a distinct mechanism, and to position it against these adjacent lines.

### 0.4 Win conditions (in increasing ambition)

1. **Characterization.** Specify the policy precisely; show that hash parameters (K, L, T) determine a predictable steady-state memory bound for given input distributions.
2. **Quality preservation under bounded memory (Framing 1).** On held-out queries, bucket-occupancy retention preserves recall@10 better than reservoir sampling, FIFO, and random sampling at matched memory budgets.
3. **Match or beat Stream-LSH** at retention quality without requiring application-supplied scores.
4. **Ecological validity (Framing 3, stretch).** On a long-context QA benchmark (LoCoMo or similar), an LLM equipped with bucket-occupancy retention answers more questions correctly under bounded memory than baseline policies.

---

## 1. Pipeline overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Environment, datasets, ground truth                       │
│  Streaming-format datasets; embedding pipeline; oracle k-NN          │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 2 — Policy design and characterization                        │
│  Define the policy formally; ablate K, L, T;                         │
│  bucket-occupancy distributions over time; steady-state behavior     │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 3 — Baseline implementation                                   │
│  Reservoir, FIFO, LRU, semantic dedup, Stream-LSH, SieveStreaming    │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 4 — Framing 1 experiments (recall under bounded memory)       │
│  Stream → policy → retention set → evaluate retrieval recall@k       │
│  Sweep memory budgets, distributions, drift conditions               │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 5 — Framing 3 experiments (LLM QA, stretch)                   │
│  Long-context QA; bounded memory; downstream answer accuracy         │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 6 — Analysis and writeup                                      │
│  Pareto curves, regime analysis, failure cases, paper                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1 — Environment, datasets, ground truth

**Duration:** 4–6 days.

### 2.1 Hardware and environment

Single fixed machine for all timing. ≥32 GB RAM, modern CPU with AVX2, one CUDA GPU for embedding and ground-truth computation. Pinned conda or Docker environment; `requirements.txt` checked in.

```
python==3.11
numpy, scipy, scikit-learn
sentence-transformers, torch       # embeddings
faiss-gpu                           # ground-truth k-NN
falconn                             # hash family (used as primitive only)
datasets                            # HuggingFace streaming corpora
matplotlib, seaborn, pandas
memray, py-spy                      # profiling
```

### 2.2 Datasets

We need stream-style datasets where a meaningful *retention* question exists. Static SIFT1M is not a stream. Pick at least two from below — one synthetic-controllable, one realistic.

#### Tier A — controlled streams (for characterization in Phase 2)

- **MS MARCO ordered stream**: 1M passages embedded with `all-MiniLM-L6-v2` (d=384), presented in a fixed permutation. Inject duplicates at controlled rates (10%, 30%, 50%) by random resampling. This lets you measure dedup quality with known answers.
- **Drift-injected stream**: same passages, but ordered by topic so the input distribution shifts halfway through. Tests robustness to non-stationarity.

#### Tier B — realistic streams (for Framing 1 in Phase 4)

- **Wikipedia recent-changes stream**: pull a temporal slice of Wikipedia edits, embed each edit's text. Natural redundancy from repeated topics.
- **arXiv abstracts by date**: ~2M abstracts with timestamps. Strong topical drift over years; mild duplication.
- **Reddit comment stream** (subset of Pushshift archives): high redundancy, conversational, closer to LLM-memory ergonomics.

#### Tier C — LLM-memory benchmark (for Framing 3 in Phase 5)

- **LoCoMo** (Long Conversation Memory benchmark): multi-session dialogues with ground-truth questions about earlier sessions. The standard benchmark in the LLM-memory literature.
- **LongMemEval**: similar, focused specifically on memory-retention skills.

### 2.3 Embedding pipeline

- Model: `sentence-transformers/all-MiniLM-L6-v2` for d=384 baseline; optionally `BAAI/bge-base-en-v1.5` for d=768 robustness check.
- L2-normalize all embeddings.
- Cache embeddings to memory-mapped `.npy` files. The streaming experiment then iterates over the file in order; embedding cost is paid once.

### 2.4 Ground truth and oracle queries

For each stream, designate the *full unfiltered set* as the oracle. For Framing 1:

1. After all N items have streamed in (oracle), pick a held-out query set of ~5,000 queries.
2. Compute oracle top-100 for each query against the full set using FAISS Flat on GPU.
3. *Also* compute top-100 against each policy's retained subset.
4. Recall is the intersection.

The oracle is the only fair upper bound — comparing recall against a policy's *own* output would be circular.

### 2.5 Phase 1 deliverables

- [ ] Frozen environment + Docker image
- [ ] Streamable datasets persisted with explicit ordering
- [ ] Oracle ground truth cached
- [ ] Synthetic duplicate injection script (parameterized)
- [ ] Drift-injection script (parameterized)

---

## 3. Phase 2 — Policy design and characterization

**Duration:** 3 weeks. **This phase defines the contribution.**

### 3.1 The policy, formally

Let `H = {h_1, ..., h_L}` be L hash tables, each producing K-bit signatures (e.g., from cross-polytope LSH). For each table `i`, maintain bucket counts `C_i[b]` for each bucket `b`. Maintain the kept set `M`.

For each new embedding `x` arriving in the stream:

1. Compute hash signatures `s_i = h_i(x)` for `i = 1..L`.
2. Compute the *occupancy score* `O(x) = aggregate({C_i[s_i]})` — definition is a research design choice (see 3.2).
3. Apply a *retention rule* `keep(x, O(x))` — also a design choice (see 3.3).
4. If kept: add `x` to `M`; increment `C_i[s_i]` for all `i`; optionally enforce a global memory cap.
5. If discarded: do nothing.

This is the policy. The whole research project is characterizing the design choices in steps 2–4.

### 3.2 Design dimension 1 — occupancy aggregation across L tables

Each item lives in L buckets, one per table. Aggregation options:

| Aggregator | Behavior | Intuition |
|---|---|---|
| `min` | Item is novel if *any* bucket is empty | Permissive — keep if novelty in any view |
| `max` | Item is novel only if *all* buckets are empty | Strict — discard if any view says redundant |
| `mean` | Average occupancy | Smooth |
| `median` | Robust to outlier tables | Compromise |

This is your first ablation. Hypothesis: `min` retains too much (low memory savings); `max` is too aggressive (drops novel items if any one table happens to be full); `median` should win.

### 3.3 Design dimension 2 — retention rule

Given an occupancy score `O(x)`, decide keep/discard:

- **Hard threshold T**: keep iff `O(x) < T`. Simple, but T is arbitrary.
- **Adaptive threshold**: T tracks the running median bucket size. Self-tuning.
- **Probabilistic**: keep with probability `p(O) = exp(-O/τ)` for temperature τ. Smoother; injects randomness for distribution coverage.
- **Capacity-bounded**: keep deterministically if `|M| < B`; else only keep if `O(x) < O_min` for the most-redundant existing item (and evict that item).

The capacity-bounded rule is what gives you the strict "memory bound" guarantee. It's also the most reviewer-friendly because it admits clean theoretical analysis.

### 3.4 Design dimension 3 — eviction (for capacity-bounded variant)

If a new item arrives and memory is full, who gets evicted?

- **No eviction**: discard the new item (FIFO-like).
- **Most-redundant eviction**: evict the item from the fullest bucket. Maintains diversity.
- **Hybrid**: evict only if new item's occupancy is much lower than evicted item's. Avoids thrashing.

### 3.5 Hash family considerations

Use cross-polytope LSH (FALCONN/Falconn++ implementation) as the hash primitive. Don't reinvent — Falconn++ is mature and well-tuned, and LSH-quality is not the contribution here. **Do not use Falconn++'s LSF filtering layer** — that's a query-time filter, irrelevant to insertion-time retention.

For each experiment, sweep:
- `L ∈ {1, 4, 8, 16}`
- `K ∈ {6, 8, 10, 12, 14}` (controls bucket granularity)

For d=384 (MiniLM), expect K=10 to 12 to be the sweet spot.

### 3.6 Characterization experiments

These are the "what is the policy doing?" experiments — *before* any baseline comparison.

1. **Steady-state memory bound.** Stream N=1M items. Plot kept-set size `|M|` over time for various (K, L, T) configurations. Confirm `|M|` saturates at a predictable bound proportional to `2^K · L / T` (or analogous).
2. **Bucket-occupancy distribution.** At t = 100K, 500K, 1M, plot histograms of bucket counts. Healthy: most buckets sparse, few hot. Pathological: one giant bucket.
3. **Retention rate vs. duplicate rate.** With injected duplicate rates {0%, 10%, 30%, 50%}, measure what fraction of duplicates the policy correctly discards vs. keeps. Ground truth = known by injection.
4. **Retention rate vs. K.** For fixed L=8 and stream of length 1M, sweep K from 6 to 14. Report (kept-set size, recall on held-out queries). Identify the goldilocks zone.
5. **Distribution drift response.** With drift-injected stream, plot retention rate as a function of stream position. Does the policy adapt or get stuck?

### 3.7 Phase 2 deliverables

- [ ] Formal policy specification (one page, with pseudocode)
- [ ] Ablation table for occupancy aggregation (3.2)
- [ ] Ablation table for retention rule (3.3)
- [ ] Ablation table for eviction (3.4)
- [ ] Five characterization plots (3.6)
- [ ] A "recommended default" configuration for use in Phase 4

---

## 4. Phase 3 — Baseline implementations

**Duration:** 2 weeks.

All baselines must operate under the *same memory budget B* and the *same input stream* as the bucket-occupancy policy. Implement each as an interface conforming to:

```python
class RetentionPolicy:
    def insert(self, embedding: np.ndarray, item: Any) -> bool:
        """Returns True if kept, False if discarded."""
    def kept_set(self) -> List[Tuple[np.ndarray, Any]]:
        """Returns the current retained items."""
```

### 4.1 Trivial baselines

- **Oracle (no filtering).** Keep everything. Recall ceiling.
- **Random sampling.** Keep with probability B/N if N is known; else use [Vitter's reservoir](https://en.wikipedia.org/wiki/Reservoir_sampling).
- **FIFO.** Keep most recent B items. Evict oldest on overflow.
- **LRU on retrieval.** Eviction informed by query history. Requires a query-stream alongside ingestion. Skip if only ingestion is streamed.

### 4.2 Stronger baselines

- **Reservoir sampling (Algorithm R).** Maintain uniform random sample of size B. Standard streaming baseline.
- **Semantic deduplication (exact, ε-threshold).** For each new item, if min distance to any kept item < ε, discard; else keep until B is reached. The "obvious" alternative — your method should match it on quality at much higher throughput.
- **SieveStreaming (submodular streaming).** Badanidiyuru, Mirzasoleiman, Karbasi, Krause, KDD 2014. Streaming submodular maximization for diversity-preserving subsampling. Strong theoretical baseline. Reference implementation in `apricot-select`.

### 4.3 Stream-LSH (the primary comparison baseline)

**Stream-LSH (Kraus, Carmel, Keidar — IEEE BigData 2017)** is the closest prior art. You must implement and compare against it.

The algorithm (paraphrased from the paper):

- Index incoming items in standard LSH tables.
- Each item carries three retention scores: freshness (decays with time), quality (per-item, application-supplied), and dynamic popularity (incremented on query hits).
- On overflow, evict items with lowest combined score.

For your experiments:

- Set quality = 1 for all items (since you don't have application-supplied quality — and the comparison should highlight that your method *also* doesn't need it).
- Set freshness with the paper's recommended decay.
- Popularity requires a query stream alongside ingestion. For experiments with a query stream, measure popularity; for ingestion-only experiments, set popularity = 0.

This isolates the comparison to: bucket-occupancy novelty (yours) vs. freshness/popularity heuristics (Stream-LSH), with quality held constant.

If you can't access the original Stream-LSH code, the paper's algorithm fits in ~300 lines of Python. Reach out to the authors first — academic code is often unpublished but available on request.

### 4.4 Optional: LLM-memory baselines

If targeting Framing 3 (Phase 5), include at least one:

- **Mem0-style importance scoring** (run an LLM over each item to score it 0–1, keep top B).
- **AMV-L-style utility tracking** with simulated query patterns.
- **Temporal decay + threshold** (CraniMem-style).

These are heavier to implement and may be out of scope. If included, frame them as "ecological validity comparisons," not direct baselines, since they use entirely different signals.

### 4.5 Phase 3 deliverables

- [ ] Working implementations of all baselines under a common interface
- [ ] Stream-LSH reproduction validated against the paper's published numbers on at least one of their datasets
- [ ] Unit tests confirming each policy respects the stated memory bound

---

## 5. Phase 4 — Framing 1 experiments (recall under bounded memory)

**Duration:** 4–6 weeks. **Headline experiments live here.**

### 5.1 Core experimental design

For each (dataset, policy, memory budget B):

1. Stream the dataset through the policy.
2. After streaming, the policy has retained some set `M` with `|M| ≤ B`.
3. For each query in the held-out query set, retrieve top-10 from `M` using exact search (FAISS Flat — search quality is *not* the variable being studied).
4. Compute recall@10 against the *oracle top-10* from the full unfiltered set.
5. Average over queries.

### 5.2 Primary metrics

- **Recall@10 vs. memory fraction (B/N).** Sweep B/N ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 1.00}. The headline plot.
- **Recall@10 vs. wall-clock streaming throughput.** Items/sec each policy can ingest. LSH bucket lookup should be fast (O(L)); semantic dedup is O(|M|).
- **Memory-efficiency ratio.** Memory needed by each policy to reach recall@10 = 0.9. Lower is better.

### 5.3 Secondary metrics

- **Recall@10 under input drift.** Same plot as 5.2, but on the drift-injected stream. Robust policies should degrade gracefully.
- **Bucket-occupancy diagnostic plots.** Hot-bucket fraction, occupancy entropy. Useful for explaining failures.
- **Tail behavior.** Recall on the *bottom 10%* of queries (worst-case). Average recall hides catastrophic single-query failures.

### 5.4 Stress tests

- **Heavy duplication (50% duplicates injected).** Bucket-occupancy should excel; reservoir should waste memory on duplicates.
- **No duplication (independent samples).** Bucket-occupancy should match reservoir sampling — if it does much worse, the policy isn't viable.
- **Adversarial ordering.** Items ordered to fill specific buckets early. Tests robustness of adaptive thresholds.

### 5.5 Statistical rigor

- ≥3 runs per (policy, B, dataset) with different seeds (for stream order, hash seeds, sampling).
- Bootstrap CIs on recall (10K resamples of queries).
- Paired bootstrap test for "policy A beats policy B at fixed B" claims.
- Recall differences below 0.005 are noise.

### 5.6 Failure analysis

Identify queries where bucket-occupancy retention performs poorly. Common patterns to look for:

- Queries that depend on items in dense semantic regions (where many similar items existed and most were dropped).
- Queries near distribution-drift boundaries.
- Queries with no near-duplicates in the original stream (the policy treats everything as novel — fine for these, easy case).

A 1–2 page failure analysis strengthens credibility.

### 5.7 Phase 4 deliverables

- [ ] Headline plot: recall@10 vs. B/N for all policies, both datasets
- [ ] Throughput-vs-recall scatter plot
- [ ] Drift-robustness plot
- [ ] Stress-test results table
- [ ] Failure analysis writeup

---

## 6. Phase 5 — Framing 3 experiments (LLM QA, stretch)

**Duration:** 3–4 weeks if pursued. Skip if Phase 4 alone is enough for the paper.

### 6.1 Setup

Use **LoCoMo** or **LongMemEval** as the benchmark. Both provide multi-session conversations with ground-truth questions referencing earlier sessions.

For each policy:

1. Stream the conversation history through the policy with memory budget B.
2. At question-asking time, the LLM has access only to `M` (the retained set) plus the current turn.
3. Retrieve top-K relevant items from `M` using standard cosine similarity.
4. Prompt LLM with retrieved items + question; record answer.
5. Score against ground truth (exact match for closed questions, LLM-judge for open questions).

### 6.2 Why this is harder

- LLM accuracy depends on retrieval quality, prompt construction, and model capability — many confounds. Use a small panel of models (e.g., GPT-4o-mini, Claude Haiku, Llama-3-8B) and report mean.
- Ground truth in long-context QA is itself noisy. Use the benchmark's official scorer, not your own.
- Cost: every QA evaluation runs an LLM. Budget API spend or use small open-weight models.

### 6.3 What success looks like

If bucket-occupancy retention preserves more answer-supporting context than reservoir/FIFO at matched memory budget, downstream answer accuracy should be measurably higher. Expected effect: smaller than recall differences in Phase 4 (LLMs can sometimes recover from imperfect context), but directionally consistent.

### 6.4 Phase 5 deliverables

- [ ] LoCoMo (or chosen benchmark) evaluation pipeline
- [ ] Answer-accuracy plot vs. memory budget
- [ ] Subjective failure-case analysis with example wrong answers

---

## 7. Phase 6 — Analysis and writeup

**Duration:** 2 weeks.

### 7.1 Suggested paper outline (8–10 pages)

1. Introduction — bounded LLM memory; why intrinsic novelty signals are interesting; gap left by Stream-LSH and the LLM-memory literature.
2. Related work — Stream-LSH, RACE, submodular streaming, LLM memory systems (Mem0, AMV-L, CraniMem).
3. Method — the bucket-occupancy policy, formally; design dimensions; theoretical memory bound.
4. Characterization — Phase 2 results.
5. Experiments — Framing 1 (Phase 4); Framing 3 (Phase 5) if included.
6. Discussion — when bucket-occupancy wins, when it loses, why.
7. Limitations — frequently-repeated-but-important items; distribution drift; static hash family.
8. Conclusion.

### 7.2 Repo hygiene

One-command reproduction. Pinned dependencies. Dataset download scripts with seeds. Result CSVs and pre-rendered plots checkpointed. Each experiment is a numbered notebook matching a paper section.

### 7.3 Phase 6 deliverables

- [ ] Submitted paper or thesis chapter
- [ ] Public repo with one-command reproduction
- [ ] Hardware/environment documentation

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Bucket-occupancy retention loses to Stream-LSH at parity memory | Medium | Reframe around "no application scores needed" + throughput advantage |
| Hash parameters too sensitive to dataset | High | This is itself a contribution — characterize the failure mode |
| Stream-LSH reproduction fails | Medium | Contact authors; reimplement from paper |
| LoCoMo (Phase 5) too expensive in API costs | Medium | Drop to a smaller QA benchmark or open-weight LLM |
| Distribution drift kills the static policy | Likely | Acknowledge as limitation; frame future work as "online recalibration" (per USR-LSH 2023) |
| Reviewers say "this is just Stream-LSH with a different score" | High | Defend on three axes: no app scores, faster, different failure modes |
| Frequent-but-important items get dropped | Certain | Acknowledge upfront; propose hybrid (occupancy + lightweight importance) as future work |

---

## 9. What success honestly looks like

The most likely outcome — based on the structure of the prior work — is:

- Bucket-occupancy retention **clearly beats reservoir / FIFO / random** at memory-efficiency on duplication-heavy streams. (High confidence.)
- It **roughly matches semantic dedup** on quality at much higher throughput. (Medium confidence.)
- It is **competitive with Stream-LSH** in some regimes (no-quality-score regime, drift-free streams) but loses in others (Stream-LSH's home turf with rich quality signals). (Medium confidence.)
- It **degrades meaningfully under distribution drift** — and that's a publishable limitation, not a failure.

A paper titled *"Bucket-Occupancy Retention: An Intrinsic Novelty Signal for Bounded LLM Memory"* with rigorous Phase-4 experiments and an honest limitations section is publishable. Don't claim a universal win. Claim a specific, measured contribution: a new retention mechanism, characterized end-to-end, positioned correctly against Stream-LSH and the LLM-memory literature.

---

## 10. Reading list

**Closest prior work (must read):**
- Kraus, Carmel, Keidar, *"Fishing in the Stream: Similarity Search over Endless Data"* (Stream-LSH), IEEE BigData 2017 ([arXiv:1708.02062](https://arxiv.org/abs/1708.02062))
- Coleman, Baraniuk, Shrivastava, *"Sub-linear Memory Sketches for Near Neighbor Search on Streaming Data"* (RACE), ICML 2020 ([arXiv:1902.06687](https://arxiv.org/abs/1902.06687))

**LSH foundations:**
- Andoni et al., *"Practical and Optimal LSH for Angular Distance"* (FALCONN), NeurIPS 2015
- Pham & Liu, *"Falconn++"*, NeurIPS 2022 — for the hash primitive

**Streaming subset selection (baselines):**
- Vitter, *"Random Sampling with a Reservoir"*, ACM TOMS 1985
- Badanidiyuru, Mirzasoleiman, Karbasi, Krause, *"Streaming Submodular Maximization: Massive Data Summarization on the Fly"* (SieveStreaming), KDD 2014

**LLM agent memory (positioning):**
- Chhikara et al., *"Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"*, 2025
- Xu et al., *"A-Mem: Agentic Memory for LLM Agents"*, 2025
- *"AMV-L: Lifecycle-Managed Agent Memory"*, 2024
- *"CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems"*, 2025

**Long-context QA benchmarks (Phase 5):**
- *"LoCoMo: Long Conversation Memory benchmark"*
- *"LongMemEval"*

**Streaming and online LSH (related context):**
- Sundaram et al., *"Streaming Similarity Search over One Billion Tweets using PLSH"*, VLDB 2013
- Singh et al., *"FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search"*, 2021
- *"USR-LSH: Unfolded Self-Reconstruction LSH for Machine Unlearning"*, 2023 — relevant for online recalibration as future work
