import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_profiling(csv_path, output_prefix, dataset_name):
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_path}: File not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Convert seconds to microseconds for easier reading
    df['lsh_us'] = df['avg_lsh_time_s'] * 1e6
    df['table_us'] = df['avg_hash_table_time_s'] * 1e6
    df['distance_us'] = df['avg_distance_time_s'] * 1e6
    
    # 1. Plot Absolute Time Stacked Bar Chart (X-axis: Probes)
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df['probes']))
    width = 0.6
    
    p1 = plt.bar(x, df['lsh_us'], width, color='#1f77b4', edgecolor='white', label='Hashing Time')
    p2 = plt.bar(x, df['table_us'], width, bottom=df['lsh_us'], color='#ff7f0e', edgecolor='white', label='Hash Table Lookup Time')
    p3 = plt.bar(x, df['distance_us'], width, bottom=df['lsh_us'] + df['table_us'], color='#d62728', edgecolor='white', label='Exact Distance Time')
    
    plt.ylabel('Microseconds per Query (\u03BCs)', fontsize=14)
    plt.xlabel('Number of Probes', fontsize=14)
    plt.title(f'FALCONN Query Time Breakdown - {dataset_name}', fontsize=16)
    plt.xticks(x, df['probes'])
    plt.legend(loc='upper left', fontsize=12)
    
    # Add recall annotations on top of the bars
    for i, (p, recall) in enumerate(zip(p3, df['recall'])):
        height = p.get_y() + p.get_height()
        plt.text(p.get_x() + p.get_width() / 2., height + (height * 0.05),
                 f'Rec: {recall:.2f}', ha='center', va='bottom', rotation=90)
                 
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_absolute.png", dpi=300)
    plt.close()
    
    # 2. Plot Relative Percentage Stacked Area Chart (X-axis: Recall)
    # This shows what percentage of the query is spent doing what, as recall increases
    plt.figure(figsize=(10, 6))
    
    total = df['lsh_us'] + df['table_us'] + df['distance_us']
    pct_lsh = (df['lsh_us'] / total) * 100
    pct_table = (df['table_us'] / total) * 100
    pct_dist = (df['distance_us'] / total) * 100
    
    plt.stackplot(df['recall'], pct_lsh, pct_table, pct_dist, 
                  labels=['Hashing Time', 'Hash Table Lookup Time', 'Exact Distance Time'],
                  colors=['#1f77b4', '#ff7f0e', '#d62728'], alpha=0.8)
                  
    plt.ylabel('Percentage of Total Query Time (%)', fontsize=14)
    plt.xlabel('Recall@10', fontsize=14)
    plt.title(f'FALCONN Bottleneck Distribution vs Recall - {dataset_name}', fontsize=16)
    plt.xlim(df['recall'].min(), df['recall'].max())
    plt.ylim(0, 100)
    plt.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_percentage.png", dpi=300)
    plt.close()
    
    print(f"Generated profiling plots for {dataset_name}")

if __name__ == "__main__":
    plot_profiling("results/profiling_msmarco.csv", "results/profiling_msmarco", "MS MARCO")
    plot_profiling("results/profiling_sift1m.csv", "results/profiling_sift1m", "SIFT1M")
