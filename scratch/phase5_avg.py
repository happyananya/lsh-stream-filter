import pandas as pd
df = pd.read_csv('results/phase5_locomo/phase5_locomo_results.csv')
summary = df.groupby(['policy','budget_fraction'])['accuracy'].mean().unstack(0)
print(summary.to_string(float_format='%.6f'))
