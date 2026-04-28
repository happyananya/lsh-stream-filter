# Phase 1 Reproducibility Guide

This document captures the Phase 1 setup for the LSH vs HNSW project:

- frozen environment dependencies
- dataset sources and expected file layout
- deterministic seeds and conventions
- exact ground-truth generation procedure

## 1) Environment

- Python: `3.11`
- Core packages are pinned via `requirements.txt`:
  - `numpy`, `scipy`, `scikit-learn`
  - `h5py`
  - `hnswlib`
  - `faiss-cpu` (CPU default; GPU FAISS optional if available)
  - `datasets`, `sentence-transformers`, `torch`
  - `matplotlib`, `seaborn`, `pandas`, `tqdm`

## 2) Data sources

- SIFT1M HDF5 source: [ann-benchmarks SIFT1M](http://ann-benchmarks.com/sift-128-euclidean.hdf5)
- MS MARCO source: HuggingFace dataset `ms_marco` (`v1.1`)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## 3) Determinism

- Global sampling seed: `42`
- MS MARCO subsampling and query selection use NumPy RNG with the same fixed seed.
- All embeddings are saved as `float32` and L2-normalized.

## 4) Canonical file layout

```text
data/
  sift1m/
    base.npy
    query.npy
    groundtruth.npy
  msmarco/
    base.npy
    query.npy
    groundtruth.npy
    passage_ids.npy
```

## 5) Scripts

### Convert SIFT1M HDF5 to .npy

```bash
python scripts/phase1/prepare_sift1m.py \
  --hdf5-path sift-128-euclidean.hdf5 \
  --output-dir data/sift1m
```

### Build MS MARCO embeddings (1M base, 6,980 queries)

```bash
python scripts/phase1/prepare_msmarco.py \
  --output-dir data/msmarco \
  --n-passages 1000000 \
  --n-queries 6980 \
  --seed 42
```

### Compute MS MARCO exact ground truth (top-100)

Use inner-product metric because vectors are unit-normalized (`IP == cosine`):

```bash
python scripts/phase1/compute_groundtruth.py \
  --base-path data/msmarco/base.npy \
  --query-path data/msmarco/query.npy \
  --output-path data/msmarco/groundtruth.npy \
  --metric ip \
  --k 100
```

If FAISS GPU is available in your environment, add `--use-gpu`.

## 6) Phase 1 completion checklist

- [x] Frozen dependency file exists (`requirements.txt`)
- [x] SIFT conversion script exists and supports canonical layout
- [x] MS MARCO embedding script exists and writes canonical layout
- [x] Ground-truth script exists and writes `groundtruth.npy`
- [ ] `data/sift1m/*.npy` generated locally
- [ ] `data/msmarco/*.npy` generated locally
- [ ] Hardware/software metadata captured for the final paper

## 7) Hardware/software metadata template

Fill this before benchmark phases:

- OS:
- CPU model:
- RAM:
- GPU model and VRAM:
- Python version:
- Torch version:
- FAISS version:
- Commit hash:
