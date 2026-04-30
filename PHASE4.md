# Phase 4 — Layered LSH Optimizations

**Goal:** Apply optimizations to the LSH baseline one at a time, regenerate the Pareto curve after each, and watch the LSH curve climb toward HNSW.

**Discipline:** Change *one thing at a time*. Commit results after each step. Resist combining steps before each is independently validated — the ablation is the scientific contribution.

---

## Where We Stand After Phase 3

| Method | Dataset | Best Recall@10 | QPS at best recall |
|--------|---------|---------------|-------------------|
| HNSW (ef=640) | SIFT1M | 0.9992 | 1,354 |
| HNSW (ef=640) | MS MARCO | 0.9930 | 621 |
| FALCONN (L=100, probes=1000) | SIFT1M | 0.9993 | **51** |
| FALCONN (L=100, probes=1000) | MS MARCO | 0.9724 | **197** |

FALCONN matches HNSW recall on SIFT1M but is **26× slower**. On MS MARCO it is both slower *and* lower recall. Phase 4 closes this gap.

---

## The 7-Step Optimization Sequence

Each step starts from the previous step's best result. The cumulative Pareto plot is the main deliverable — it shows the trajectory of improvements.

| Step | Optimization | Expected Impact | Status |
|------|-------------|-----------------|--------|
| 4.1 | Cross-polytope hashing | Large recall lift (near-optimal ρ for angular distance) | ✅ Implemented |
| 4.2 | Multi-probe LSH | ~10× memory reduction at matched recall | ✅ Implemented |
| 4.5 | Reranking with exact distance | +0.05–0.20 recall; biggest single win | ✅ Implemented |
| 4.3 | PCA preprocessing | Moderate; cuts hash cost on MS MARCO (d=384→128) | 🔲 Stub ready |
| 4.4 | ITQ rotation | +1–3 recall points; cheap to implement | 🔲 Stub ready |
| 4.6 | Bit-packing + SIMD popcount | 10–50× QPS gain; no recall change | 🔲 Stub ready |
| 4.7 | Learned projections | Largest potential; stretch goal (2–3 weeks) | 🔲 Stretch |

> Steps 4.1, 4.2, and 4.5 are implemented in `experiments/phase4_lsh.py`. Steps 4.3–4.7 have class stubs in the same file — fill them in one at a time.

---

## Running Phase 4

```bash
# From repo root
cd /path/to/lsh-stream-filter

# Steps 4.1 + 4.2 + 4.5 on SIFT1M
python experiments/phase4_lsh.py --dataset sift1m

# Steps 4.1 + 4.2 + 4.5 on MS MARCO
python experiments/phase4_lsh.py --dataset msmarco

# Run only one step
python experiments/phase4_lsh.py --dataset sift1m --step 4.1
python experiments/phase4_lsh.py --dataset sift1m --step 4.5
```

**Outputs:**
- `results/phase4_sift1m.csv` / `results/phase4_msmarco.csv` — full sweep results
- `results/phase4_sift1m_pareto.png` / `results/phase4_msmarco_pareto.png` — cumulative Pareto plot overlaid on Phase 3 baselines

---

## Step-by-Step Implementation Guide

### Step 4.1 — Cross-polytope hashing

**What it does:** Replaces `h(x) = sign(w·x)` (random hyperplane) with a structured rotation. The rotation is `D₁ H D₂ H ... x` where `D_i` are random diagonal sign matrices and `H` is the Walsh-Hadamard matrix. The hash output is `argmax_i |z_i|` (which coordinate is largest) plus the sign of that coordinate.

**Why it's better:** Near-optimal collision probability `ρ` for angular distance. FALCONN's high recall comes from this — we're now doing it from scratch so we own the pipeline.

**Key implementation detail:** For d=384 (MS MARCO), pad to d=512 (next power of 2) before the Hadamard transform. The rotation cost stays `O(d log d)` — never use a dense random matrix (`O(d²)`).

**What to measure:** Recall@10 at matched memory vs FALCONN. They should be very similar — this validates the implementation before adding any new optimizations.

---

### Step 4.2 — Multi-probe LSH

**What it does:** Instead of L=100 tables × 1 probe, use L=8–16 tables × T perturbed probes per table.

**Probe sequence:** For a query q, compute the rotated vector z = R(q). The primary bucket is `argmax |z|`. Secondary probes replace that argmax with the 2nd, 3rd, … largest coordinates. These are exactly the highest-probability adjacent buckets.

**Parameter sweep:** T ∈ {4, 16, 64, 256} at L ∈ {8, 16, 32}.

**What to measure:** Memory drops ~10× (L=16 vs L=100) at matched recall. QPS may drop slightly (more probes per query) — this tradeoff is the headline.

---

### Step 4.3 — PCA preprocessing *(stub in `PCATransform`)*

**What it does:** Project base and query vectors to a lower-dimensional space before hashing. Fit PCA on the base set only; apply the same transform at query time.

**Suggested reductions:**
- MS MARCO: d=384 → d'=128 or d'=256
- SIFT1M: d=128 → d'=64

**Implementation:**
```python
mean = base.mean(axis=0)
_, _, Vt = np.linalg.svd(base - mean, full_matrices=False)
components = Vt[:n_components]          # shape (n_components, d)

# At query time:
q_proj = (q - mean) @ components.T     # shape (n_components,)
```

**What to measure:** Does recall hold when d' is half of d? If yes, hash compute cost drops 2× for free.

---

### Step 4.4 — ITQ rotation *(stub in `ITQRotation`)*

**What it does:** After PCA, learn a rotation R that minimises the quantization error `||sign(R·X) - R·X||`. Typically ~50 iterations of alternating optimization; no labels needed.

**Implementation (alternating SVD):**
```python
R = np.eye(n_components)
for _ in range(50):
    B = np.sign(X_pca @ R.T)           # binarize
    U, _, Vt = np.linalg.svd(B.T @ X_pca)
    R = (U @ Vt).T                     # update rotation
# Save R to disk and reuse at query time
```

**What to measure:** ITQ typically yields +1–3 recall points at the same bit budget. Cheap to implement — almost always worth including.

---

### Step 4.5 — Reranking with exact distance

**What it does:** After LSH retrieves m candidates, compute exact L2 distance to all m candidates and re-sort. Return top k.

**This is the single biggest win.** Without it, you're not giving LSH a fair fight — HNSW's `efSearch` does exactly this.

**Parameter sweep:** `rerank_m ∈ {100, 200, 500, 1000, 2000}` at `num_probes ∈ {16, 64}`.

**Implementation note:** The reranking uses vectorized NumPy (`np.einsum`) — never a Python loop over candidates.

**What to measure:** Recall@10 climbs +0.05–0.20. This step alone often changes the narrative from "LSH is hopeless" to "LSH is competitive."

---

### Step 4.6 — Bit-packing + SIMD popcount *(stub in `bit_packed_hamming_stub`)*

**What it does:** Pack binary hash signatures into `uint64` words. Replace float distance computation with fast Hamming distance via popcount.

**For prototyping (NumPy ≥ 2.0):**
```python
sigs_packed = np.packbits(binary_sigs, axis=1).view(np.uint64)  # (N, n_words)
query_packed = np.packbits(query_sig).view(np.uint64)           # (n_words,)
xor = sigs_packed ^ query_packed                                 # (N, n_words)
hamming = np.bitwise_count(xor).sum(axis=1)                     # (N,)
```

**For production:** Drop to C via `ctypes` or `pybind11`. Pure NumPy leaves 5–10× QPS on the table. On AVX-512 hardware, `vpopcntq` is the gold standard.

**Profile before and after** with `py-spy` to confirm the bottleneck moved.

**What to measure:** QPS lifts substantially with zero recall change. This is the "make it fast" step.

---

### Step 4.7 — Learned projections *(stretch goal)*

**What it does:** Replace random/cross-polytope projections with projections learned from data.

**Lightweight option:** A linear projection trained with a contrastive loss. Self-supervised positive pairs: a vector and its top-1 exact neighbor.

**Heavier option:** Small MLP outputting hash bits, trained end-to-end with sigmoid relaxation + quantization regularizer (see Neural LSH, Dong et al., ICLR 2020).

**Important:** Keep the cross-polytope variant as a baseline alongside the learned variant. The learned version is the contribution; the classical one is the floor.

**What to measure:** Does the learned variant beat cross-polytope at matched memory and QPS? If yes, this is the headline result of the paper.

**Time budget:** 2–3 weeks. Treat as a stretch goal — the paper survives without it.

---

## Ablation Discipline

After each step:

1. Regenerate the Pareto curve on **both** datasets.
2. Add the new curve to the cumulative plot (showing the trajectory of improvements).
3. Record the delta vs the previous step:
   - Recall@10 at fixed QPS (e.g., 1000 QPS)
   - QPS at fixed recall (e.g., recall=0.95)
   - Memory at fixed recall
4. Commit results to the repo with a tag for that step (e.g., `phase4-step4.1`).

The cumulative trajectory plot — 7+ Pareto curves stacked on one figure — is the most important figure in the paper.

---

## Phase 4 Deliverables

- [ ] `results/phase4_sift1m.csv` — one row per (method, parameters, step)
- [ ] `results/phase4_msmarco.csv`
- [ ] `results/phase4_sift1m_pareto.png` — cumulative Pareto trajectory
- [ ] `results/phase4_msmarco_pareto.png`
- [ ] Ablation table (recall, QPS, memory deltas per step)
- [ ] Each optimization in a separate, reviewable commit

---

## Known Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| PCA aggressively cuts recall on SIFT1M (low intrinsic dim) | Low | Try d'=96 before d'=64; skip if recall drops >2 points |
| ITQ convergence is slow on MS MARCO | Low | Cap at 50 iterations; convergence is usually fast |
| Learned projections don't converge | Medium | Treat as stretch; paper survives without Step 4.7 |
| Bit-packing gives less QPS gain than expected | Low | Profile to find real bottleneck; it's usually elsewhere by then |
