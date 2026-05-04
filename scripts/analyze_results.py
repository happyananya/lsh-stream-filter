"""Quick analysis of the Phase 4 results CSVs."""
import pandas as pd

for stream in ["HeavyDuplication_50", "CleanStream_0", "TopicDrift"]:
    print(f"\n{'='*70}")
    print(f"  {stream}")
    print(f"{'='*70}")
    df = pd.read_csv(f"results/phase4_framing1/phase4_sweep_{stream}.csv")
    
    for frac in sorted(df['budget_fraction'].unique()):
        subset = df[df['budget_fraction'] == frac]
        print(f"\n  B/N = {frac:.0%}:")
        for _, row in subset.iterrows():
            name = row['policy']
            r10 = row['recall@10_mean']
            r1 = row.get('recall@1_mean', 0)
            r100 = row.get('recall@100_mean', 0)
            cov10 = row.get('coverage@10', 0)
            cov100 = row.get('coverage@100', 0)
            kcr = row.get('k_center_radius', 0)
            intra = row.get('mean_intra_set_distance', 0)
            tp = row['throughput_items_per_sec']
            p99 = row.get('p99_us', 0)
            print(f"    {name:30s}  R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}  "
                  f"Cov@10={cov10:.4f}  Cov@100={cov100:.4f}  "
                  f"k-center={kcr:.4f}  intra-dist={intra:.4f}  "
                  f"TP={tp:,.0f}  p99={p99:.1f}us")
