# Closing the Recall Gap: An Implementation Plan for Optimized LSH vs. HNSW

> **Research goal.** Determine whether a carefully optimized Locality-Sensitive Hashing (LSH) variant can match Hierarchical Navigable Small World (HNSW) graphs — the dominant production method — on the recall-vs-QPS frontier, evaluated on both a classical benchmark (SIFT1M) and a modern neural embedding workload (MS MARCO + MiniLM).

---

## 0. Scope and framing

This project deliberately limits its scope to keep results clean and defensible:

- **Methods compared:** Vanilla random-hyperplane LSH, FALCONN (strong classical LSH), HNSW (the target to match), and a sequence of optimized LSH variants (the contribution).
- **Datasets:** One classical (SIFT1M, d=128, L2) and one modern neural (MS MARCO passages embedded with `all-MiniLM-L6-v2`, d=384, cosine).
- **Win conditions (in order of ambition):**
  1. Show the recall-QPS gap to HNSW shrinks measurably with each optimization.
  2. Match HNSW recall@10 ≥ 0.95 within 2× QPS at competitive memory.
  3. Beat HNSW on at least one dimension (memory, build time, insertion throughput, or GPU throughput) at parity recall.

Hitting (1) is virtually guaranteed if the optimizations are implemented correctly. Hitting (2) on the neural dataset is the headline result. Hitting (3) is the stretch goal that turns the work into a publishable contribution.

---

## 1. Pipeline overview

The full pipeline is six phases, executed in order. Each phase produces concrete artifacts that feed the next.

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Environment & ground truth                                │
│  Build reproducible env, download datasets, compute exact k-NN       │
│  Output: frozen environment, ground-truth k-NN files                 │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 2 — Reproducibility baselines                                 │
│  Reproduce published HNSW and FALCONN numbers on SIFT1M              │
│  Output: validated reference numbers; calibration of harness         │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 3 — Honest baselines on both datasets                         │
│  Run vanilla LSH, FALCONN, HNSW on SIFT1M and MS MARCO               │
│  Output: the "before" Pareto curves — your starting gap              │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 4 — Layered LSH optimizations (the core contribution)         │
│  Add optimizations one at a time, ablating each                      │
│  Output: a sequence of Pareto curves showing progressive improvement │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 5 — Final comparison and analysis                             │
│  Where does optimized LSH match HNSW? Where does it lose? Why?       │
│  Output: regime analysis, confidence intervals, failure cases        │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 6 — Writeup                                                   │
│  Output: paper, code repo, reproducibility artifacts                 │
└──────────────────────────────────────────────────────────────────────┘
```

The rest of this document is a phase-by-phase implementation guide.

---

## 2. Phase 1 — Environment and ground truth

**Duration:** 3–5 days. **Goal:** A reproducible environment, both datasets loaded, exact k-NN computed and cached.

### 2.1 Hardware

Pin the entire project to one machine. All timing numbers in the paper come from this machine. Document the spec.

- **Recommended minimum:** 32 GB RAM, modern x86 CPU with AVX2 (AVX-512 ideal), one CUDA GPU (any consumer card with ≥8 GB) for ground-truth computation and optional GPU experiments.
- **Disable Turbo Boost** during timing runs, or pin frequency. If you can't, document the variance.
- **Single-threaded numbers are the headline.** Multi-threaded numbers are secondary and reported separately.

### 2.2 Software environment

Lock everything. Use a `conda` env with pinned versions, or a Docker image, or both.

```
# Core
python==3.11
numpy
scipy
scikit-learn

# ANN libraries
hnswlib                 # HNSW reference implementation
faiss-gpu               # for ground-truth k-NN (and optional baselines)
falconn                 # cross-polytope LSH reference

# Embeddings
sentence-transformers
torch                   # CUDA build matching your GPU

# Benchmarking
ann-benchmarks          # cloned, pinned to a specific commit

# Analysis
matplotlib, seaborn, pandas
```

Commit a `requirements.txt` with exact versions. Reviewers will replicate on different hardware; the version pin is the difference between "reproducible" and "doesn't run."

### 2.3 Datasets

#### SIFT1M

- **Source:** ann-benchmarks distributes a normalized HDF5 file. Download from `http://ann-benchmarks.com/sift-128-euclidean.hdf5`.
- **Contents:** 1,000,000 base vectors, 10,000 query vectors, 128 dimensions, L2 distance. Includes precomputed ground-truth k-NN with k=100.
- **Why this dataset:** It's the canonical ANN benchmark. Every paper in this space reports SIFT1M numbers. Skipping it invites "did you cherry-pick?"

#### MS MARCO + MiniLM

- **Corpus:** MS MARCO passages v1 (~8.8M passages). Download from the official MS MARCO site or the HuggingFace `ms_marco` dataset.
- **Subsample:** Take 1,000,000 passages with a fixed random seed (`numpy.random.RandomState(42).choice(...)`). Document the seed.
- **Embeddings:** Embed with `sentence-transformers/all-MiniLM-L6-v2`. Output is 384-d, normalize to unit L2.
- **Queries:** Use 6,980 dev set queries from MS MARCO, embedded with the same model.
- **Distance:** Cosine (which on unit-normalized vectors is equivalent to inner product, equivalent to `(2 - L2²)/2`). Pick one representation and stick with it.
- **Why this dataset:** It's the closest standardized analog to a real RAG workload. d=384 is the most common production embedding size. Results here transfer to readers' actual use cases.

#### Storage

Persist embeddings as `numpy.float32` memory-mappable files (`.npy` with `mmap_mode='r'`). Do *not* store as Python pickles or CSVs — the I/O cost will distort timing.

```
data/
  sift1m/
    base.npy       # (1_000_000, 128) float32
    query.npy      # (10_000, 128) float32
    groundtruth.npy  # (10_000, 100) int32
  msmarco/
    base.npy       # (1_000_000, 384) float32, L2-normalized
    query.npy      # (6_980, 384) float32, L2-normalized
    groundtruth.npy  # (6_980, 100) int32
    passage_ids.npy  # original MS MARCO IDs for reproducibility
```

### 2.4 Ground truth

For SIFT1M, the ann-benchmarks file already includes ground truth. For MS MARCO, compute it yourself with FAISS on GPU:

```python
import faiss
index = faiss.IndexFlatIP(384)            # inner product on unit vectors == cosine
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(base)                        # base is (N, 384) float32
D, I = gpu_index.search(query, 100)        # (Q, 100) neighbor IDs
np.save('data/msmarco/groundtruth.npy', I)
```

This takes ~2–3 minutes on a single consumer GPU. Verify by spot-checking 10 queries manually.

### 2.5 Phase 1 deliverables

- [ ] Frozen environment file
- [ ] Both datasets persisted to disk in standard format
- [ ] Ground-truth k-NN cached for both datasets
- [ ] A 1-page README documenting hardware, software versions, data sources, and seeds

---

## 3. Phase 2 — Reproducibility baselines

**Duration:** 1 week. **Goal:** Reproduce known published numbers within 10%. **Why it matters:** if you can't reproduce them, your environment or harness has a bug, and every later result is suspect.

### 3.1 Reproduce HNSW on SIFT1M

Target: `recall@10 ≥ 0.95` at `~10,000 QPS` single-threaded on a modern CPU, using `hnswlib` with `M=16, efConstruction=200, efSearch=100`.

```python
import hnswlib
index = hnswlib.Index(space='l2', dim=128)
index.init_index(max_elements=1_000_000, M=16, ef_construction=200)
index.add_items(base, np.arange(len(base)))
index.set_ef(100)
labels, distances = index.knn_query(query, k=10)
```

Compute recall@10 against the ground truth. If it's below 0.94 or above 0.97, something is wrong (hyperparameters, data normalization, or distance metric).

### 3.2 Reproduce FALCONN on SIFT1M

Use FALCONN's cross-polytope LSH with their recommended parameters for SIFT (their docs and the 2015 NeurIPS paper publish numbers). Verify recall and QPS land in the same ballpark.

### 3.3 Calibrate the timing harness

Build the harness once, here, and use it for every later experiment. Critical features:

- **Warm-up:** Run 1,000 throwaway queries before measuring. CPU caches and Python JIT (if any) need to settle.
- **Per-query timing:** Use `time.perf_counter_ns` per query, not wall-clock for the whole batch. You want latency *distributions*, not just averages.
- **Single-threaded enforcement:** Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` for single-threaded runs. Verify with `htop` that only one core is busy.
- **Recall computation:** `len(set(predicted) & set(ground_truth[:k])) / k`, averaged over queries.

```python
def benchmark(index_query_fn, queries, gt, k=10, warmup=1000):
    # Warmup
    for q in queries[:warmup]:
        _ = index_query_fn(q, k)
    # Measure
    latencies = []
    recalls = []
    for q, g in zip(queries, gt):
        t0 = time.perf_counter_ns()
        pred = index_query_fn(q, k)
        latencies.append(time.perf_counter_ns() - t0)
        recalls.append(len(set(pred) & set(g[:k])) / k)
    return {
        'recall': np.mean(recalls),
        'qps': 1e9 / np.mean(latencies),
        'p50_us': np.percentile(latencies, 50) / 1000,
        'p95_us': np.percentile(latencies, 95) / 1000,
        'p99_us': np.percentile(latencies, 99) / 1000,
    }
```

### 3.4 Phase 2 deliverables

- [ ] HNSW recall@10 on SIFT1M within 10% of published `ann-benchmarks` numbers
- [ ] FALCONN recall@10 on SIFT1M within 10% of published numbers
- [ ] Frozen benchmark harness module used unchanged for all later phases
- [ ] Brief notebook documenting the reproduction with plots

---

## 4. Phase 3 — Honest baselines on both datasets

**Duration:** 2 weeks. **Goal:** A complete "before" picture — full Pareto curves for vanilla LSH, FALCONN, and HNSW on both SIFT1M and MS MARCO.

### 4.1 Methods to run

| Method | Library | Role |
|---|---|---|
| Random-hyperplane LSH | Custom or FAISS `IndexLSH` | "Classical LSH" baseline — the punching bag |
| FALCONN cross-polytope LSH | `falconn` | Strong classical LSH baseline |
| HNSW | `hnswlib` | Target to match |

### 4.2 Hyperparameter sweeps

For each method on each dataset, sweep parameters to trace the full recall-QPS Pareto frontier:

- **Random-hyperplane LSH:** `K ∈ {8, 16, 24, 32}` (bits per signature), `L ∈ {1, 4, 16, 64, 128}` (tables). 20 points.
- **FALCONN:** Sweep `num_hash_tables`, `num_hash_bits`, `num_probes` per their docs. 20 points.
- **HNSW:** Build with `M=16, efConstruction=200`, then sweep `efSearch ∈ {10, 20, 40, 80, 160, 320, 640}`. 7 points (cheap because rebuild not required).

For each (method, dataset, hyperparameter) combination, record: recall@10, QPS, p95 latency, index memory, build time.

### 4.3 The "before" plots

Generate two plots — one per dataset:

- **X-axis:** recall@10 (linear, 0.0–1.0)
- **Y-axis:** QPS (log scale)
- **Lines:** one per method, connecting Pareto-optimal points only (drop dominated points)

These are the plots you will spend the rest of the project trying to change. Pin them somewhere visible.

### 4.4 Expected starting gap

Going in, your hypothesis should be:

- **SIFT1M (d=128):** FALCONN gets reasonably close to HNSW. Random hyperplane is far behind. Gap to close: small-to-moderate.
- **MS MARCO (d=384):** Both LSH methods are noticeably behind HNSW at recall@10 ≥ 0.9. Random hyperplane is dramatically behind. Gap to close: large.

The MS MARCO gap is the harder, more interesting research question. It's the one your paper will live or die on.

### 4.5 Phase 3 deliverables

- [ ] Raw results table (CSV) with one row per (method, dataset, hyperparameters, run)
- [ ] Two "before" Pareto plots
- [ ] A 1–2 paragraph written summary of the gap and what success looks like

---

## 5. Phase 4 — Layered LSH optimizations (the core contribution)

**Duration:** 6–8 weeks. **Goal:** Apply optimizations sequentially, ablating each, and watch the LSH curve climb toward HNSW.

This is the heart of the project. The discipline here matters: change *one thing at a time*, regenerate the Pareto curve, document the delta. Resist the urge to combine optimizations before each is independently validated.

### 5.1 Optimization sequence

Each step starts from the previous step's result.

| Step | Optimization | Expected impact | Why |
|---|---|---|---|
| 4.1 | Cross-polytope hashing (replace random hyperplane) | Large | Near-optimal `ρ` for angular distance; FALCONN-grade quality |
| 4.2 | Multi-probe LSH | Large memory reduction | Fewer tables (8–16 vs 100+) at similar recall |
| 4.3 | PCA preprocessing | Moderate | Capture variance in fewer effective dimensions |
| 4.4 | ITQ rotation | Moderate | Minimize quantization error; cheap free win |
| 4.5 | Reranking with exact distance | Large at high recall | Mirrors HNSW's `efSearch` mechanic |
| 4.6 | Bit-packing + SIMD popcount | Large QPS gain | 10–50× faster Hamming distance |
| 4.7 | Learned projections (stretch) | Largest, but risky | Data-adaptive partitioning beats random |

### 5.2 Step 4.1 — Cross-polytope hashing

Replace `h(x) = sign(w·x)` with cross-polytope hashing: rotate `x` with a pseudo-random rotation (Hadamard + diagonal sign matrices, Andoni et al. 2015), then output `argmax_i |coordinate_i|` along with its sign.

**Implementation tips:**

- Use FALCONN's pseudo-random rotations as a reference.
- For d=384 (MS MARCO), pad to the next power of 2 (512) to make the Hadamard transform clean.
- The rotation is `O(d log d)` instead of `O(d²)` — keep it that way; naive matrix-vector kills throughput.

**What to measure:** recall@10 at matched memory and matched QPS vs random-hyperplane baseline. Expect a substantial recall lift on MS MARCO especially.

### 5.3 Step 4.2 — Multi-probe LSH

Instead of `L = 100` independent tables, use `L = 8–16` tables and probe `T` perturbed buckets per table during query. Score perturbations by their expected probability of containing the true neighbors (Lv et al., 2007 — equation 11 in that paper).

**Implementation tips:**

- Precompute the probe sequence offline; it depends only on hash function statistics, not on the query.
- The probe sequence is the same for all queries within a table. Generate once, reuse.
- Sweep `T ∈ {1, 4, 16, 64}` to trace a new sub-frontier.

**What to measure:** memory drops by ~10× at matched recall. QPS may drop slightly (more probes per query), but the memory win is the headline.

### 5.4 Step 4.3 — PCA preprocessing

Fit PCA on the base set (or a 100k random sample to save time), keep top `d' < d` components, then run cross-polytope LSH on the PCA-projected vectors.

**Implementation tips:**

- For d=384 → d'=128 or d'=256. For d=128 → d'=64.
- Fit PCA on base vectors, *not* queries. Apply the same transform at query time.
- This costs build-time but pays off in distance-computation cost and bucket quality.

**What to measure:** does recall hold when `d'` is half of `d`? If yes, you've cut hash compute by 2× for free.

### 5.5 Step 4.4 — ITQ rotation

After PCA, apply Iterative Quantization (Gong & Lazebnik 2011): learn a rotation matrix that minimizes the quantization error `||sign(R·X) - R·X||²`. ~50 iterations of alternating optimization, no labels needed.

**Implementation tips:**

- Initialize `R` as a random rotation.
- Alternate: (a) `B = sign(R·X)`, (b) `R = U·V^T` from SVD of `B^T·X`.
- Converges in <50 iterations. Save `R` to disk; reuse at query time.

**What to measure:** ITQ typically yields a 1–3 point recall lift at fixed bit budget. Cheap to implement; almost always worth including.

### 5.6 Step 4.5 — Reranking

After LSH retrieves a candidate set of size `m` (e.g., 500), compute exact distances to the query for those `m` candidates and re-sort. Return top `k`.

**Implementation tips:**

- Sweep `m ∈ {100, 200, 500, 1000, 2000}`.
- This is HNSW's `efSearch` analog. **Without it, you're not giving LSH a fair fight.**
- Vectorize the reranking with NumPy or BLAS — never a Python loop over candidates.

**What to measure:** recall@10 typically climbs by 0.05–0.20 at minimal QPS cost. This step alone is often the difference between "LSH is hopeless" and "LSH is competitive."

### 5.7 Step 4.6 — Bit-packing and SIMD

Pack hash signatures into `uint64` words. Compute Hamming distance with `__builtin_popcountll` (in C) or `numpy.bitwise_xor` followed by population count via `numpy.unpackbits` (Python — slower but works for prototyping). On AVX-512 hardware, the `vpopcntq` instruction is the gold standard.

**Implementation tips:**

- Drop to C/C++ via Cython, ctypes, or pybind11 for the hot loop. Pure NumPy will leave 5–10× on the table.
- For prototyping, `numpy.bitwise_count` (NumPy ≥ 2.0) works.
- Profile before and after with `py-spy` to confirm the bottleneck moved.

**What to measure:** QPS lifts substantially without changing recall. This is the "make it fast" step.

### 5.8 Step 4.7 — Learned projections (stretch goal)

Replace random/cross-polytope projections with projections learned from data. Two options:

- **Lightweight:** A linear projection trained with a contrastive loss (preserve known neighbors, separate non-neighbors). Train on a subset of the base set with self-supervised positive pairs (e.g., a vector and its top-1 exact neighbor).
- **Heavier:** A small MLP outputting hash bits, trained end-to-end with a relaxed (sigmoid) hash and a quantization regularizer. See Neural LSH (Dong et al., ICLR 2020) for the template.

**Implementation tips:**

- This is the highest-variance bet. Budget 2–3 weeks.
- If you go this route, *also* keep the cross-polytope variant as a baseline. The learned variant is the contribution; the classical one is the floor.

**What to measure:** does the learned variant beat cross-polytope at matched memory and QPS? If yes, this is your headline result.

### 5.9 Ablation discipline

After each step:

1. Regenerate the Pareto curve on both datasets.
2. Add the new curve to a cumulative plot (showing the trajectory of improvements).
3. Record: recall@10 at fixed QPS, QPS at fixed recall, memory at fixed recall.
4. Commit results to the repo with a tag for that step.

By the end of Phase 4, you'll have 7+ Pareto curves stacked, each representing one optimization layer. **This trajectory plot is the most important figure in the paper.**

### 5.10 Phase 4 deliverables

- [ ] One results CSV per optimization step
- [ ] Cumulative Pareto trajectory plot (per dataset)
- [ ] Per-step ablation table (recall, QPS, memory deltas)
- [ ] Code repo with each optimization in a separate, reviewable commit

---

## 6. Phase 5 — Final comparison and analysis

**Duration:** 2 weeks. **Goal:** Honest characterization of where optimized LSH matches HNSW, where it doesn't, and why.

### 6.1 Final head-to-head

Two summary plots — one per dataset — showing only:

1. HNSW Pareto curve (the target)
2. Vanilla random-hyperplane LSH (the "before")
3. Final optimized LSH (the "after")

These three curves tell the whole story at a glance.

### 6.2 Statistical rigor

- Run each (method, hyperparameter) point ≥3 times with different seeds where applicable. Report mean ± std.
- Bootstrap confidence intervals on recall (10,000 resamples of queries).
- For "method A beats method B at fixed recall" claims, use a paired bootstrap test on per-query latencies.
- Recall differences below 0.005 are within noise — don't over-claim.

### 6.3 Regime analysis

Even if optimized LSH doesn't dominate HNSW everywhere, find the regimes where it wins. Concretely, compare on:

| Axis | Why it might favor LSH |
|---|---|
| Memory at fixed recall | LSH's hash codes are tiny vs HNSW's graph |
| Index build time | LSH builds in one pass; HNSW is incremental and slower |
| Insertion throughput | New vectors are O(L) hashes; HNSW is O(M·log n) graph ops |
| Recall at very high QPS (low latency) | LSH has fewer sequential dependencies |
| GPU throughput | LSH is embarrassingly parallel; HNSW's graph traversal isn't |
| Robustness to distribution drift | Open question — design an experiment for it |

Pick the 2–3 axes where your method wins clearest. These become the "contribution" framing in the writeup.

### 6.4 Failure analysis

For queries where LSH gets recall@10 < 0.5 but HNSW gets ≥ 0.9, examine: what's special about them? Common patterns:

- Queries near cluster boundaries
- Queries in low-density regions
- Queries with many near-equidistant neighbors (the curse of dimensionality biting)

A short failure-case analysis in the paper builds credibility — it shows you understand your method's limits.

### 6.5 Phase 5 deliverables

- [ ] Final head-to-head plots (per dataset)
- [ ] Regime analysis table with confidence intervals
- [ ] Failure case write-up (1–2 pages with examples)
- [ ] Honest "limitations" section draft

---

## 7. Phase 6 — Writeup

**Duration:** 2 weeks. **Goal:** A paper, a clean repo, and reproducibility artifacts.

### 7.1 Paper structure

Suggested outline (8–10 pages):

1. Introduction — why LSH lost, why we revisit it
2. Background — LSH formalism, HNSW summary, related work
3. Optimizations — one subsection per Phase 4 step
4. Experimental setup — datasets, metrics, hardware
5. Results — Pareto trajectory + final head-to-head + regime analysis
6. Discussion — where LSH wins, where it loses, why
7. Limitations and future work
8. Conclusion

### 7.2 Repo hygiene

- One-command reproduction: `make reproduce` should regenerate every plot in the paper.
- Pinned dependencies, dataset download scripts, fixed seeds.
- Each experiment has a numbered notebook or script matching a section in the paper.
- Pre-computed results checkpointed so reviewers don't have to wait 8 hours to see your plots.

### 7.3 Phase 6 deliverables

- [ ] Submitted paper (or thesis chapter)
- [ ] Public code repo with one-command reproduction
- [ ] Result CSVs and pre-rendered plots checkpointed
- [ ] Hardware/environment documentation

---

## 8. Realistic expectations and risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 2 reproduction fails | Medium | Budget extra time; suspect Python/library version mismatches first |
| Optimized LSH still loses to HNSW on MS MARCO | High | Reframe around regime wins (memory, build time, GPU) — those are real contributions |
| Learned projections (Step 4.7) don't converge | Medium | Treat as a stretch; the paper survives without this step |
| Bit-packing/SIMD gives less than expected | Low | Profile and find the actual bottleneck — usually it's elsewhere by then |
| Multi-probe scoring is fiddly to tune | Medium | Use FALCONN's reference implementation as ground truth |

The honest framing: matching HNSW's recall-QPS frontier on d=384 with classical LSH (even FALCONN-grade) is hard. What's almost certainly achievable and still publishable:

- A measurable, monotone Pareto improvement at every optimization step.
- LSH wins on memory, build time, or insertion throughput at parity recall.
- A clear story about *why* LSH closes (or doesn't close) the gap on modern dense embeddings.

A paper titled *"How close can optimized LSH get to HNSW on modern neural embeddings?"* with rigorous experiments and an honest answer is more interesting and more publishable than one that overclaims a win that doesn't replicate.

---

## 9. Reading list (start here)

**Foundational:**
- Indyk & Motwani, *"Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality"* (STOC 1998)
- Datar et al., *"Locality-Sensitive Hashing Scheme Based on p-Stable Distributions"* (SoCG 2004)
- Andoni & Indyk, *"Near-Optimal Hashing Algorithms for ANN in High Dimensions"* (FOCS 2006)

**Modern LSH:**
- Lv et al., *"Multi-probe LSH"* (VLDB 2007)
- Andoni et al., *"Practical and Optimal LSH for Angular Distance"* (NeurIPS 2015) — FALCONN

**HNSW:**
- Malkov & Yashunin, *"Efficient and robust ANN search using HNSW graphs"* (TPAMI 2018)

**Learning to hash:**
- Gong & Lazebnik, *"Iterative Quantization"* (CVPR 2011)
- Wang et al., *"A Survey on Learning to Hash"* (TPAMI 2018)
- Dong et al., *"Learning Space Partitions for NN Search"* (ICLR 2020)

**Benchmarking:**
- Aumüller, Bernhardsson, Faithfull, *"ANN-Benchmarks"* (Information Systems, 2020)
- Li et al., *"ANN Search on High Dimensional Data — Experiments, Analyses, and Improvement"* (TKDE 2020)
