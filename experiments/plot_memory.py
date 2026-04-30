import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_memory_graphs(csv_path, dataset_name):
    if not os.path.exists(csv_path):
        print(f"Skipping {dataset_name}: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    if 'Index_Size_MB' not in df.columns:
        print(f"Skipping {dataset_name}: 'Index_Size_MB' column not found. Did you rerun the sweep?")
        return

    sns.set_style("whitegrid")
    colors = {
        'HNSW': '#d62728',
        'Vanilla LSH (FAISS)': '#1f77b4',
        'FALCONN': '#ff7f0e'
    }

    # 1. Recall vs Memory (Memory on X, Recall on Y)
    plt.figure(figsize=(10, 8))
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        color = colors.get(method, '#333333')
        # Since efSearch/probes don't change memory but change recall, 
        # we will plot a scatter plot of all points.
        plt.scatter(method_df['Index_Size_MB'], method_df['recall'], 
                    label=method, color=color, alpha=0.7, s=80, edgecolors='k')
        
    plt.xscale('log') # Memory can span orders of magnitude
    plt.xlabel('Index Size (MB)', fontsize=14)
    plt.ylabel('Recall@10', fontsize=14)
    plt.title(f'{dataset_name} Baseline: Recall vs Memory', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    out_recall = f"results/{dataset_name.lower()}_recall_vs_memory.png"
    plt.savefig(out_recall, dpi=300)
    print(f"Saved {out_recall}")
    plt.close()

    # 2. Memory vs QPS (Memory on X, QPS on Y)
    plt.figure(figsize=(10, 8))
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        color = colors.get(method, '#333333')
        plt.scatter(method_df['Index_Size_MB'], method_df['qps'], 
                    label=method, color=color, alpha=0.7, s=80, edgecolors='k')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Index Size (MB)', fontsize=14)
    plt.ylabel('Queries per second (QPS)', fontsize=14)
    plt.title(f'{dataset_name} Baseline: Memory vs QPS', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    out_qps = f"results/{dataset_name.lower()}_memory_vs_qps.png"
    plt.savefig(out_qps, dpi=300)
    print(f"Saved {out_qps}")
    plt.close()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    plot_memory_graphs("results/phase3_msmarco.csv", "MSMARCO")
    plot_memory_graphs("results/phase3_sift1m.csv", "SIFT1M")
