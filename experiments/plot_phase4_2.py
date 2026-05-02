import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from plot_pareto import get_pareto_frontier

def plot_phase4_2_comparison(phase3_csv, phase4_2_csv, output_path, dataset_name):
    if not os.path.exists(phase3_csv) or not os.path.exists(phase4_2_csv):
        print(f"Skipping {dataset_name}: CSVs not found.")
        return
        
    df3 = pd.read_csv(phase3_csv)
    df4 = pd.read_csv(phase4_2_csv)
    
    # Combine dataframes
    df = pd.concat([df3, df4], ignore_index=True)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    colors = {
        'HNSW': '#d62728',
        'Vanilla LSH (FAISS)': '#1f77b4',
        'FALCONN': '#ff7f0e',
        'LSH-HNSW Hybrid (FAISS)': '#9467bd' # Purple for Hybrid
    }
    
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        Xs = method_df['recall'].values
        Ys = method_df['qps'].values
        
        frontier = get_pareto_frontier(Xs, Ys)
        
        color = colors.get(method, '#333333')
        linewidth = 3 if method == 'LSH-HNSW Hybrid (FAISS)' else 2
        markersize = 10 if method == 'LSH-HNSW Hybrid (FAISS)' else 8
        zorder = 5 if method == 'LSH-HNSW Hybrid (FAISS)' else 3
        
        plt.plot(frontier[:, 0], frontier[:, 1], marker='o', linewidth=linewidth, 
                 label=method, color=color, markersize=markersize, zorder=zorder)
        
    plt.yscale('log')
    plt.xlabel('Recall@10', fontsize=14)
    plt.ylabel('Queries per second (QPS)', fontsize=14)
    plt.title(f'Phase 4.2: {dataset_name} Pareto Frontier (LSH-HNSW Hybrid)', fontsize=16)
    plt.xlim(0.0, 1.05)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Phase 4.2 comparison plot saved to {output_path}")

if __name__ == "__main__":
    plot_phase4_2_comparison("results/phase3_msmarco.csv", "results/phase4_2_msmarco.csv", "results/phase4_2_msmarco_pareto.png", "MSMARCO")
    plot_phase4_2_comparison("results/phase3_sift1m.csv", "results/phase4_2_sift1m.csv", "results/phase4_2_sift1m_pareto.png", "SIFT1M")
