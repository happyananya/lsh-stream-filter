import pandas as pd
import numpy as np

df = pd.read_csv('results/phase5_locomo/phase5_locomo_results.csv')
# The user might have a trailing empty line or something, let's clean
df = df.dropna(subset=['accuracy'])

# Group by policy and budget
summary = df.groupby(['policy', 'budget_fraction'])['accuracy'].mean().unstack(level=0)
print("--- Mean Accuracy by Policy and Budget ---")
print(summary)

# Also check raw counts for sanity
totals = df.groupby(['policy', 'budget_fraction'])[['correct', 'total']].sum()
totals['overall_accuracy'] = totals['correct'] / totals['total']
print("\n--- Overall Accuracy (Sum of correct / Sum of total) ---")
print(totals.unstack(level=0)['overall_accuracy'])
