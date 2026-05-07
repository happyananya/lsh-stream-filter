import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_jaccard_comparison():
    # Load data
    csv_path = 'results/phase4_framing1/phase4_sweep_HeavyDuplication_50.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter to the policies we want to compare
    target_policies = [
        'FIFO', 
        'Reservoir', 
        'BucketOccupancy', 
        'BucketOccupancy (Jaccard)'
    ]
    df_filtered = df[df['policy'].isin(target_policies)].copy()
    
    # Clean up policy names for the legend
    df_filtered['policy'] = df_filtered['policy'].replace({
        'BucketOccupancy': 'BucketOccupancy (Median)',
        'BucketOccupancy (Jaccard)': 'BucketOccupancy (Jaccard)'
    })
    
    # Set aesthetics
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    # Custom palette
    palette = {
        'FIFO': '#ff3366',                 # Pink
        'Reservoir': '#33ccff',            # Light Blue
        'BucketOccupancy (Median)': '#00ffcc',  # Cyan
        'BucketOccupancy (Jaccard)': '#ffcc00'  # Gold
    }
    
    # Plot Recall@10
    sns.lineplot(
        data=df_filtered,
        x='budget_fraction',
        y='recall@10_mean',
        hue='policy',
        palette=palette,
        marker='o',
        markersize=10,
        linewidth=3
    )
    
    # Format axes
    plt.title('Jaccard vs Median Aggregator (HeavyDuplication_50)', fontsize=16, pad=20, color='white')
    plt.xlabel('Memory Budget (B/N)', fontsize=12, labelpad=10)
    plt.ylabel('Recall@10', fontsize=12, labelpad=10)
    
    # Set x-ticks to match our budgets
    plt.xticks([0.01, 0.05, 0.1, 0.25, 0.5], ['1%', '5%', '10%', '25%', '50%'])
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    plt.legend(title='Policy Strategy', frameon=True, facecolor='black', edgecolor='white')
    
    # Save the plot
    output_path = 'results/phase4_framing1/plot_jaccard_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_jaccard_comparison()
