import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phase5():
    # Load data
    csv_path = 'results/phase5_locomo/phase5_locomo_results.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['accuracy'])
    
    # Set aesthetics
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    # Colors and markers
    colors = {
        'BucketOccupancy': '#00ffcc',  # Cyan/Teal
        'FIFO': '#ff3366',            # Pink/Red
        'Reservoir': '#ccff33'        # Lime
    }
    
    # Group by policy and budget for plotting
    # We'll use a lineplot with error bars (standard error)
    sns.lineplot(
        data=df,
        x='budget_fraction',
        y='accuracy',
        hue='policy',
        palette=colors,
        marker='o',
        markersize=10,
        linewidth=3,
        errorbar=('se', 1)
    )
    
    # Format axes
    plt.title('Phase 5: LoCoMo LLM-QA Accuracy vs Memory Budget', fontsize=16, pad=20, color='white')
    plt.xlabel('Memory Budget (% of Conversation Turns)', fontsize=12, labelpad=10)
    plt.ylabel('QA Accuracy (Mean across Conversations)', fontsize=12, labelpad=10)
    
    # Set x-ticks to match our budgets
    plt.xticks([0.1, 0.25, 0.5, 1.0], ['10%', '25%', '50%', '100%'])
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    plt.legend(title='Retention Policy', frameon=True, facecolor='black', edgecolor='white')
    
    # Save the plot
    output_path = 'results/phase5_locomo/phase5_accuracy_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_phase5()
