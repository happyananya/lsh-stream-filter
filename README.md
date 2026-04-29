# LSH vs HNSW Research Workflow

This repository tracks an implementation-first research plan for comparing LSH and HNSW on ANN search workloads.

The active plan is documented in `LSH_vs_HNSW_Research_Plan (1).md` and currently focuses on:

- Phase 1: dataset/environment setup and ground truth
- Phase 2: reproducibility baselines on SIFT1M
- Phase 3+: broader baseline sweeps and optimization work

## Active Structure

- `scripts/phase1/` - dataset preparation and ground-truth generation
- `scripts/phase2/` - SIFT1M baseline reproduction (HNSW, FALCONN, fallback)
- `experiments/phase3_sift1m.py` - Phase 3 SIFT1M baseline runs
- `experiments/phase3_msmarco.py` - Phase 3 MS MARCO baseline runs
- `results/phase2/` - Phase 2 outputs
- `results/phase3_*.csv`, `results/phase3_*_pareto.png` - Phase 3 outputs

Legacy stream-filter prototype files (`lsh.py`, `stream.py`, `oracle.py`, `evaluate.py`, and `experiments/day*.py`) are kept for reference but are not the primary plan path now.

## Environment Setup

Recommended: Python 3.11 virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `python3.11` is not available on macOS:

```bash
brew install python@3.11
export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
```

## Phase 1 (Data + Ground Truth)

```bash
python scripts/phase1/prepare_sift1m.py --hdf5-path sift-128-euclidean.hdf5 --output-dir data/sift1m
python scripts/phase1/prepare_msmarco.py --output-dir data/msmarco --n-passages 1000000 --n-queries 6980 --seed 42
python scripts/phase1/compute_groundtruth.py --base-path data/msmarco/base.npy --query-path data/msmarco/query.npy --output-path data/msmarco/groundtruth.npy --metric ip --k 100
```

Details: `PHASE1.md`

## Phase 2 (SIFT1M Reproducibility Baselines)

One-command run:

```bash
python scripts/phase2/run_phase2_sift1m.py --data-dir data/sift1m --results-dir results/phase2
```

This executes:

- HNSW SIFT1M baseline
- FALCONN SIFT1M baseline (if installable)
- FAISS LSH fallback if FALCONN is unavailable
- consolidated CSV + recall/QPS plot + markdown report

Manual commands:

```bash
python scripts/phase2/reproduce_hnsw_sift1m.py --data-dir data/sift1m --k 10 --M 16 --ef-construction 200 --ef-search 100
python scripts/phase2/reproduce_falconn_sift1m.py --data-dir data/sift1m --k 10 --num-hash-tables 20 --num-hash-bits 18 --num-probes 64
python scripts/phase2/reproduce_faiss_lsh_sift1m.py --data-dir data/sift1m --k 10 --nbits 256
python scripts/phase2/summarize_phase2.py --results-dir results/phase2
```

Details: `PHASE2.md`

## Current Known Constraint

On modern macOS toolchains, `falconn` may fail to build from source. When that happens, Phase 2 scripts automatically fall back to FAISS `IndexLSH` so work can proceed, but those results are not equivalent to a true FALCONN baseline.

## Notes

- Keep large data artifacts in `data/` out of git (already covered in `.gitignore`).
- Use `results/phase2/` and `results/phase3_*` as the source of truth for generated metrics and figures.