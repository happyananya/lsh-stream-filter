"""
Reorganization script: splits old combined CSVs into per-method files,
archives old optimization experiments, and sets up the new directory structure.

This script does NOT delete anything — it copies and moves.
"""
import os
import shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EXPERIMENTS = os.path.join(ROOT, "experiments")

def split_csv(csv_path, output_dir):
    """Split a combined CSV into per-method CSVs."""
    df = pd.read_csv(csv_path)
    basename = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. "phase3_msmarco"
    dataset = basename.split("_", 1)[1]  # e.g. "msmarco"
    
    method_map = {
        "HNSW": f"phase2_hnsw_{dataset}.csv",
        "FALCONN": f"phase2_falconn_{dataset}.csv",
        "Vanilla LSH (FAISS)": f"phase2_vanillalsh_{dataset}.csv",
    }
    
    for method, outfile in method_map.items():
        subset = df[df["Method"] == method]
        if len(subset) > 0:
            outpath = os.path.join(output_dir, outfile)
            subset.to_csv(outpath, index=False)
            print(f"  Wrote {len(subset)} rows -> {outfile}")

def archive_results():
    """Move old optimization results and plots into results/archive/."""
    archive = os.path.join(RESULTS, "archive")
    os.makedirs(archive, exist_ok=True)
    
    # Files to archive (old phase4 stuff + old combined CSVs + old plots)
    patterns_to_archive = [
        "phase3_msmarco.csv", "phase3_sift1m.csv",
        "phase3_msmarco_pareto.png", "phase3_sift1m_pareto.png",
        "phase4_msmarco.csv", "phase4_sift1m.csv",
        "phase4_msmarco_pareto.png", "phase4_sift1m_pareto.png",
        "phase4_2_msmarco.csv", "phase4_2_sift1m.csv",
        "phase4_2_msmarco_pareto.png", "phase4_2_sift1m_pareto.png",
        "msmarco_memory_vs_qps.png", "msmarco_recall_vs_memory.png",
        "sift1m_memory_vs_qps.png", "sift1m_recall_vs_memory.png",
        "memory_over_time.png", "precision_heatmap.png",
        "recall_heatmap.png", "theory_vs_empirical.png",
        "sweep_results.csv",
    ]
    
    for f in patterns_to_archive:
        src = os.path.join(RESULTS, f)
        if os.path.exists(src):
            dst = os.path.join(archive, f)
            shutil.copy2(src, dst)
            print(f"  Archived {f}")

def archive_experiments():
    """Move old optimization experiment scripts into experiments/archive/."""
    archive = os.path.join(EXPERIMENTS, "archive")
    os.makedirs(archive, exist_ok=True)
    
    scripts_to_archive = [
        "layered_lsh.py",
        "lsh_hnsw_hybrid.py",
        "phase3_msmarco.py",
        "phase3_sift1m.py",
        "phase4_msmarco.py",
        "phase4_sift1m.py",
        "phase4_2_msmarco.py",
        "phase4_2_sift1m.py",
        "plot_phase4.py",
        "plot_phase4_2.py",
        "plot_pareto_sift.py",
    ]
    
    for f in scripts_to_archive:
        src = os.path.join(EXPERIMENTS, f)
        if os.path.exists(src):
            dst = os.path.join(archive, f)
            shutil.copy2(src, dst)
            print(f"  Archived {f}")

def rename_profiling():
    """Copy profiling script with clearer name."""
    src = os.path.join(EXPERIMENTS, "profile_falconn.py")
    dst = os.path.join(EXPERIMENTS, "phase3_profile_falconn.py")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"  Copied profile_falconn.py -> phase3_profile_falconn.py")

def main():
    print("=== Splitting CSVs ===")
    for csv_name in ["phase3_msmarco.csv", "phase3_sift1m.csv"]:
        csv_path = os.path.join(RESULTS, csv_name)
        if os.path.exists(csv_path):
            print(f"Splitting {csv_name}:")
            split_csv(csv_path, RESULTS)
    
    print("\n=== Archiving old results ===")
    archive_results()
    
    print("\n=== Archiving old experiment scripts ===")
    archive_experiments()
    
    print("\n=== Renaming profiling script ===")
    rename_profiling()
    
    print("\nDone! Directory reorganized for Falconn++ plan.")

if __name__ == "__main__":
    main()
