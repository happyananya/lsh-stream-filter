"""
Reorganization script for the LSH Memory Retention project.
Moves all ANN-benchmark related files into archive/ann_benchmarks/
while preserving data/ (MS MARCO embeddings are still needed).
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def archive_dir(src_dir, dst_dir, files):
    """Copy files from src_dir to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        src = os.path.join(src_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, f))
            print(f"  Archived {f}")

def main():
    ARCHIVE = os.path.join(ROOT, "archive", "ann_benchmarks")
    os.makedirs(ARCHIVE, exist_ok=True)
    
    # 1. Archive ALL old experiment scripts
    exp_dir = os.path.join(ROOT, "experiments")
    exp_archive = os.path.join(ARCHIVE, "experiments")
    exp_files = [
        "benchmark_harness.py", "layered_lsh.py", "lsh_hnsw_hybrid.py",
        "phase2_falconnpp.py", "phase3_msmarco.py", "phase3_sift1m.py",
        "phase3_profile_falconn.py", "phase4_msmarco.py", "phase4_sift1m.py",
        "phase4_2_msmarco.py", "phase4_2_sift1m.py",
        "plot_memory.py", "plot_pareto.py", "plot_pareto_sift.py",
        "plot_phase4.py", "plot_phase4_2.py", "plot_profiling.py",
        "profile_falconn.py",
    ]
    print("=== Archiving experiment scripts ===")
    archive_dir(exp_dir, exp_archive, exp_files)
    
    # Also archive experiments/archive/ subdirectory if it exists
    old_exp_archive = os.path.join(exp_dir, "archive")
    if os.path.isdir(old_exp_archive):
        dst = os.path.join(exp_archive, "archive")
        if not os.path.exists(dst):
            shutil.copytree(old_exp_archive, dst)
            print("  Archived experiments/archive/ subdirectory")
    
    # 2. Archive ALL old results
    res_dir = os.path.join(ROOT, "results")
    res_archive = os.path.join(ARCHIVE, "results")
    print("\n=== Archiving results ===")
    # Archive everything in results/ (files only, not subdirs)
    res_files = [f for f in os.listdir(res_dir) 
                 if os.path.isfile(os.path.join(res_dir, f))]
    archive_dir(res_dir, res_archive, res_files)
    
    # Archive results subdirs
    for subdir in ["archive", "_smoke", "_wsl_gpu_smoke", "phase2"]:
        src = os.path.join(res_dir, subdir)
        dst = os.path.join(res_archive, subdir)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"  Archived results/{subdir}/ subdirectory")
    
    # 3. Archive old scripts (not data prep)
    scripts_dir = os.path.join(ROOT, "scripts")
    scripts_archive = os.path.join(ARCHIVE, "scripts")
    scripts_files = [
        "build_falconnpp.sh", "compute_groundtruth_gpu.py",
        "reorganize.py",
    ]
    print("\n=== Archiving old scripts ===")
    archive_dir(scripts_dir, scripts_archive, scripts_files)
    
    # 4. Archive old plan documents
    docs_archive = os.path.join(ARCHIVE, "docs")
    doc_files = [
        "LSH_vs_HNSW_Research_Plan.md", "PHASE1.md", "PHASE2.md",
        "updated_with_falconnPlusPlus.md",
    ]
    print("\n=== Archiving old plan documents ===")
    archive_dir(ROOT, docs_archive, doc_files)
    
    print("\n=== Done! All ANN-benchmark files archived to archive/ann_benchmarks/ ===")
    print("Data directory preserved (MS MARCO embeddings reused).")
    print("Vendor directory preserved (FalconnPP build reused).")

if __name__ == "__main__":
    main()
