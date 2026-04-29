import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plot_pareto import get_pareto_frontier

def plot_pareto_sift(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    colors = {
        'HNSW': '#d62728',
        'Vanilla LSH (FAISS)': '#1f77b4',
        'FALCONN': '#ff7f0e'
    }
    
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        Xs = method_df['recall'].values
        Ys = method_df['qps'].values
        
        frontier = get_pareto_frontier(Xs, Ys)
        
        color = colors.get(method, '#333333')
        plt.plot(frontier[:, 0], frontier[:, 1], marker='o', linewidth=2, label=method, color=color, markersize=8)
        
    plt.yscale('log')
    plt.xlabel('Recall@10', fontsize=14)
    plt.ylabel('Queries per second (QPS)', fontsize=14)
    plt.title('Phase 3: SIFT1M (d=128) Baseline Pareto Frontier', fontsize=16)
    plt.xlim(0.0, 1.05)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Pareto plot saved to {output_path}")

if __name__ == "__main__":
    plot_pareto_sift("results/phase3_sift1m.csv", "results/phase3_sift1m_pareto.png")
