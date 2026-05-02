"""
Phase 2: Characterization Plots
=================================
Generates the five key plots from the characterization experiments.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

RESULTS_DIR = 'results/phase2_characterization'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


def plot_exp1_steady_state():
    """Plot 1: |M| over time for various configs."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp1_steady_state.csv'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for config in df['config'].unique():
        subset = df[df['config'] == config]
        ax1.plot(subset['t'], subset['kept'], label=config, linewidth=1.5)
        ax2.plot(subset['t'], subset['retention_rate'], label=config, linewidth=1.5)
    
    ax1.set_xlabel('Stream position (t)')
    ax1.set_ylabel('Kept set size |M|')
    ax1.set_title('Steady-State Memory Bound')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    ax2.set_xlabel('Stream position (t)')
    ax2.set_ylabel('Retention rate |M|/t')
    ax2.set_title('Retention Rate Over Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'exp1_steady_state.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved exp1_steady_state.png")


def plot_exp2_bucket_distributions():
    """Plot 2: Bucket-occupancy histograms at different checkpoints."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    checkpoints = [100000, 500000, 1000000]
    
    for ax, t in zip(axes, checkpoints):
        fpath = os.path.join(RESULTS_DIR, f'exp2_bucket_hist_t{t}.npy')
        if not os.path.exists(fpath):
            continue
        hist = np.load(fpath)
        
        # Clip for readability
        max_val = int(np.percentile(hist, 99))
        ax.hist(hist, bins=min(50, max_val + 1), range=(0, max_val + 1),
                color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_title(f't = {t:,}')
        ax.set_xlabel('Bucket occupancy')
        ax.set_ylabel('Number of buckets')
        ax.axvline(np.median(hist), color='red', linestyle='--', label=f'median={np.median(hist):.0f}')
        ax.axvline(np.mean(hist), color='orange', linestyle='--', label=f'mean={np.mean(hist):.1f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Bucket-Occupancy Distributions (K=10, L=8)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'exp2_bucket_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved exp2_bucket_distributions.png")


def plot_exp3_duplicate_sweep():
    """Plot 3: Retention metrics vs. duplicate rate."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp3_duplicate_sweep.csv'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = range(len(df))
    labels = df['dup_rate'].tolist()
    
    ax1.bar(x, df['dup_discard_rate'] * 100, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Duplicate injection rate')
    ax1.set_ylabel('Duplicate discard rate (%)')
    ax1.set_title('Duplicate Detection Accuracy')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, df['final_kept'], color='#3498db', alpha=0.8, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Duplicate injection rate')
    ax2.set_ylabel('Final kept set size')
    ax2.set_title('Retained Items vs. Duplication')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'exp3_duplicate_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved exp3_duplicate_sweep.png")


def plot_exp4_k_sweep():
    """Plot 4: Recall@10 and kept size vs. K."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp4_k_sweep.csv'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(df['K'], df['recall_at_10'], 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax1.fill_between(df['K'], df['recall_p10'], df['recall_p90'], alpha=0.2, color='#2ecc71')
    ax1.set_xlabel('K (hash bits)')
    ax1.set_ylabel('Recall@10')
    ax1.set_title('Recall vs. Hash Granularity')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['K'], df['kept'], 'o-', color='#9b59b6', linewidth=2, markersize=8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['K'], df['throughput'], 's--', color='#e67e22', linewidth=1.5, markersize=6)
    ax2.set_xlabel('K (hash bits)')
    ax2.set_ylabel('Kept set size', color='#9b59b6')
    ax2_twin.set_ylabel('Throughput (items/s)', color='#e67e22')
    ax2.set_title('Memory & Throughput vs. K')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'exp4_k_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved exp4_k_sweep.png")


def plot_exp5_drift_response():
    """Plot 5: Windowed retention rate vs. stream position on drift stream."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp5_drift_response.csv'))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    midpoints = (df['window_start'] + df['window_end']) / 2
    
    ax1.plot(midpoints, df['window_retention_rate'], color='#2c3e50', linewidth=1.5)
    ax1.set_ylabel('Windowed retention rate')
    ax1.set_title('Retention Rate Under Distribution Drift')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(midpoints, df['dominant_cluster'], 'o', color='#e74c3c', markersize=3, alpha=0.7)
    ax2.set_xlabel('Stream position')
    ax2.set_ylabel('Dominant cluster ID')
    ax2.set_title('Topic Drift (cluster transitions)')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'exp5_drift_response.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved exp5_drift_response.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Generating Phase 2 characterization plots...")
    plot_exp1_steady_state()
    plot_exp2_bucket_distributions()
    plot_exp3_duplicate_sweep()
    plot_exp4_k_sweep()
    plot_exp5_drift_response()
    print(f"\nAll plots saved to {PLOTS_DIR}/")

if __name__ == "__main__":
    main()
