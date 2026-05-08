import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_cosine_diversity():
    # Load data
    csv_path = 'results/phase4_cosine/cosine_sweep_HeavyDuplication_50.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Set aesthetics
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Custom palette
    palette = {
        'FIFO': '#ff3366',                 # Pink
        'Reservoir': '#33ccff',            # Light Blue
        'BucketOccupancy': '#00ffcc'       # Cyan
    }
    
    # 1. Cosine Diversity Score (Higher = More spread out)
    sns.lineplot(
        data=df,
        x='budget_fraction',
        y='cosine_diversity_score',
        hue='policy',
        palette=palette,
        marker='o',
        markersize=10,
        linewidth=3,
        ax=ax1
    )
    ax1.set_title('Cosine Diversity Score (Higher is Better)', fontsize=14, pad=15)
    ax1.set_xlabel('Memory Budget (B/N)', fontsize=12)
    ax1.set_ylabel('Score (1 - Mean NN Cosine)', fontsize=12)
    ax1.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5])
    ax1.set_xticklabels(['1%', '5%', '10%', '25%', '50%'])
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(title='Policy')

    # 2. Cosine Coverage Radius (Lower = Better Coverage)
    sns.lineplot(
        data=df,
        x='budget_fraction',
        y='cosine_coverage_radius',
        hue='policy',
        palette=palette,
        marker='o',
        markersize=10,
        linewidth=3,
        ax=ax2
    )
    ax2.set_title('Cosine Coverage Radius (Lower is Better)', fontsize=14, pad=15)
    ax2.set_xlabel('Memory Budget (B/N)', fontsize=12)
    ax2.set_ylabel('Angular Gap (1 - Min Similarity)', fontsize=12)
    ax2.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5])
    ax2.set_xticklabels(['1%', '5%', '10%', '25%', '50%'])
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(title='Policy')

    plt.suptitle('Phase 4: Cosine-Based Diversity Metrics (HeavyDuplication_50)', fontsize=18, y=1.05)
    plt.tight_layout()
    
    # Save the plot
    output_path = 'results/phase4_cosine/plot_cosine_diversity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_cosine_diversity()
