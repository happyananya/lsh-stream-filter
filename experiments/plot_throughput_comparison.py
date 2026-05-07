import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_throughput_comparison():
    # Load data
    csv_path = 'results/phase4_framing1/phase4_sweep_HeavyDuplication_50.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter to the policies we want to compare
    # Note: Reservoir is so fast it often skews the scale, but we'll include it.
    target_policies = [
        'FIFO', 
        'Reservoir', 
        'BucketOccupancy', 
        'BucketOccupancy (Jaccard)'
    ]
    df_filtered = df[df['policy'].isin(target_policies)].copy()
    
    # Clean up policy names
    df_filtered['policy'] = df_filtered['policy'].replace({
        'BucketOccupancy': 'BucketOccupancy (Median)',
        'BucketOccupancy (Jaccard)': 'BucketOccupancy (Jaccard)'
    })
    
    # Set aesthetics
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    palette = {
        'FIFO': '#ff3366',
        'Reservoir': '#33ccff',
        'BucketOccupancy (Median)': '#00ffcc',
        'BucketOccupancy (Jaccard)': '#ffcc00'
    }
    
    # Plot Throughput on primary y-axis
    sns.lineplot(
        data=df_filtered,
        x='budget_fraction',
        y='throughput_items_per_sec',
        hue='policy',
        palette=palette,
        marker='o',
        markersize=10,
        linewidth=3,
        ax=ax1
    )
    
    # Format axes
    plt.title('Throughput Performance Comparison', fontsize=16, pad=20, color='white')
    ax1.set_xlabel('Memory Budget (B/N)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Throughput (Items / Second)', fontsize=12, labelpad=10)
    
    # Log scale for throughput because Reservoir is 100x faster than Jaccard
    ax1.set_yscale('log')
    
    # Set x-ticks
    plt.xticks([0.01, 0.05, 0.1, 0.25, 0.5], ['1%', '5%', '10%', '25%', '50%'])
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.3, which='both')
    
    # Legend
    ax1.legend(title='Policy Strategy', frameon=True, facecolor='black', edgecolor='white')
    
    # Save the plot
    output_path = 'results/phase4_framing1/plot_throughput_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_throughput_comparison()
