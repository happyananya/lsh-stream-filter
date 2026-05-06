import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phase6():
    # Load data
    csv_path = 'results/phase6_jl/phase6_jl_results.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Set aesthetics
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    # Colors
    colors = {
        'BucketOccupancy (Baseline, d=384)': '#00ffcc',  # Cyan
        'FIFO (Baseline, d=384)': '#ff3366',            # Pink
        'BucketOccupancy + JL (d=128)': '#ffcc00',      # Yellow
        'BucketOccupancy + JL (d=64)': '#ccff33'        # Lime
    }
    
    # We want to plot Recall@10 vs Memory Budget Fraction (which represents fixed float budget)
    # The x-axis is the fraction of the *original* stream's memory footprint.
    
    sns.lineplot(
        data=df,
        x='budget_fraction',
        y='recall@10_mean',
        hue='policy',
        palette=colors,
        marker='o',
        markersize=10,
        linewidth=3
    )
    
    # Format axes
    plt.title('Phase 6: JL Projection Memory-Recall Tradeoff', fontsize=16, pad=20, color='white')
    plt.xlabel('Memory Budget (Fraction of Full Uncompressed Stream)', fontsize=12, labelpad=10)
    plt.ylabel('Recall@10', fontsize=12, labelpad=10)
    
    # Set x-ticks to match our budgets
    plt.xticks([0.01, 0.05, 0.1, 0.25, 0.5], ['1%', '5%', '10%', '25%', '50%'])
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    plt.legend(title='Policy & Dimension', frameon=True, facecolor='black', edgecolor='white')
    
    # Save the plot
    output_path = 'results/phase6_jl/phase6_jl_tradeoff.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_phase6()
