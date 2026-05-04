"""
Phase 4: Framing 1 Plots
=========================
Generates the recall vs memory budget plots for Phase 4 experiments.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

RESULTS_DIR = 'results/phase4_framing1'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


def plot_budget_sweep(stream_name):
    csv_file = os.path.join(RESULTS_DIR, f'phase4_sweep_{stream_name}.csv')
    if not os.path.exists(csv_file):
        print(f"Skipping {stream_name} (results not found)")
        return
        
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'BucketOccupancy': '#2ecc71',
        'FIFO': '#e74c3c',
        'FIFO (Oracle Upper Bound)': '#e74c3c',
        'Reservoir': '#3498db',
        'Stream-LSH': '#9b59b6',
        'SemanticDedup (eps=0.1)': '#f39c12'
    }
    
    markers = {
        'BucketOccupancy': 'o',
        'FIFO': 's',
        'FIFO (Oracle Upper Bound)': 's',
        'Reservoir': '^',
        'Stream-LSH': 'D',
        'SemanticDedup (eps=0.1)': 'x'
    }
    
    for policy in df['policy'].unique():
        subset = df[df['policy'] == policy].sort_values('budget_fraction')
        c = colors.get(policy, 'black')
        m = markers.get(policy, 'o')
        
        ax1.plot(subset['budget_fraction'] * 100, subset['recall@10_mean'], 
                 marker=m, label=policy, color=c, linewidth=2, markersize=8)
        
        # Add error bars (p10 to p90)
        if 'recall@10_p10' in subset.columns and 'recall@10_p90' in subset.columns:
            ax1.fill_between(subset['budget_fraction'] * 100, 
                             subset['recall@10_p10'], 
                             subset['recall@10_p90'], 
                             color=c, alpha=0.1)
                 
        ax2.plot(subset['budget_fraction'] * 100, subset['throughput_items_per_sec'], 
                 marker=m, label=policy, color=c, linewidth=2, markersize=8)
    
    # Ax 1: Recall
    ax1.set_xlabel('Memory Budget B/N (%)')
    ax1.set_ylabel('Recall@10')
    ax1.set_title(f'Recall vs. Memory Budget ({stream_name})')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Ax 2: Throughput
    ax2.set_xlabel('Memory Budget B/N (%)')
    ax2.set_ylabel('Throughput (items/sec)')
    ax2.set_title(f'Ingestion Throughput ({stream_name})')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'plot_{stream_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot_{stream_name}.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Generating Phase 4 plots...")
    for stream in ["HeavyDuplication_50", "CleanStream_0", "TopicDrift"]:
        plot_budget_sweep(stream)
    
    print(f"\nAll plots saved to {PLOTS_DIR}/")

if __name__ == "__main__":
    main()
