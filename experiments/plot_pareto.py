import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_pareto_frontier(Xs, Ys):
    """
    Given a list of X (Recall) and Y (QPS) points, computes the Pareto frontier.
    We want to maximize both Recall and QPS.
    """
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=True) # sort by Recall descending
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        # If the QPS is greater than the previous best QPS, it's Pareto optimal
        # (since we are moving down in Recall)
        if pair[1] >= pareto_front[-1][1]:
            pareto_front.append(pair)
            
    # Sort back by Recall ascending for plotting lines left-to-right
    frontier = np.array(pareto_front)
    return frontier[frontier[:,0].argsort()]

def plot_pareto(csv_path, output_path):
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
    plt.title('Phase 3: MS MARCO Baseline Pareto Frontier', fontsize=16)
    plt.xlim(0.0, 1.05)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Pareto plot saved to {output_path}")

if __name__ == "__main__":
    plot_pareto("results/phase3_msmarco.csv", "results/phase3_msmarco_pareto.png")
