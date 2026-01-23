"""
Compare different correlation methods:
1. Spearman
2. Pearson
3. Kendall's tau
4. Partial correlation (controlling for age and PMI)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from _paths import SEGMENTATION_REFINED_ROOT, FULLDATA_ROOT

# Try to import pingouin for partial correlation
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Note: pingouin not installed, skipping partial correlation")
    print("Install with: pip install pingouin")

# Load data
df = pd.read_csv(SEGMENTATION_REFINED_ROOT / 'donor_level_aggregation_refined.csv')

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']
clinical_vars = ['cogdx', 'braaksc', 'ceradsc', 'nft', 'plaq_d', 'plaq_n', 'gpath']
covariates = ['age_death', 'pmi']

# Determine ratio column
if 'GFAP_ratio_from_sum' in df.columns:
    RATIO_COL = lambda m: f'{m}_ratio_from_sum'
else:
    RATIO_COL = lambda m: f'{m}_marker_positive_ratio_refined_mean'

print(f"Loaded {len(df)} donors\n")
print("="*100)
print("COMPARISON OF CORRELATION METHODS")
print("="*100)

# Store results
results = []

for marker in markers:
    marker_col = RATIO_COL(marker)
    if marker_col not in df.columns:
        continue
    
    for clin in clinical_vars:
        if clin not in df.columns:
            continue
        
        # Get complete cases
        mask = df[[marker_col, clin]].notna().all(axis=1)
        if mask.sum() < 10:
            continue
        
        x = df.loc[mask, marker_col].values
        y = df.loc[mask, clin].values
        n = mask.sum()
        
        # 1. Spearman
        r_spear, p_spear = stats.spearmanr(x, y)
        
        # 2. Pearson
        r_pear, p_pear = stats.pearsonr(x, y)
        
        # 3. Kendall
        r_kend, p_kend = stats.kendalltau(x, y)
        
        # 4. Pearson on log-transformed data (if all positive)
        if (x > 0).all() and (y > 0).all():
            r_log, p_log = stats.pearsonr(np.log(x), np.log(y))
        else:
            r_log, p_log = np.nan, np.nan
        
        # 5. Partial correlation (controlling for age and PMI)
        r_partial, p_partial = np.nan, np.nan
        if HAS_PINGOUIN:
            mask_partial = df[[marker_col, clin] + covariates].notna().all(axis=1)
            if mask_partial.sum() >= 10:
                try:
                    result = pg.partial_corr(
                        data=df[mask_partial], 
                        x=marker_col, 
                        y=clin, 
                        covar=covariates, 
                        method='spearman'
                    )
                    r_partial = result['r'].values[0]
                    p_partial = result['p-val'].values[0]
                except:
                    pass
        
        results.append({
            'Marker': marker,
            'Clinical': clin,
            'n': n,
            'Spearman_r': r_spear,
            'Spearman_p': p_spear,
            'Pearson_r': r_pear,
            'Pearson_p': p_pear,
            'Kendall_tau': r_kend,
            'Kendall_p': p_kend,
            'LogPearson_r': r_log,
            'LogPearson_p': p_log,
            'Partial_r': r_partial,
            'Partial_p': p_partial,
        })

# Create DataFrame
df_results = pd.DataFrame(results)

# Add significance columns
df_results['Spearman_sig'] = df_results['Spearman_p'] < 0.05
df_results['Pearson_sig'] = df_results['Pearson_p'] < 0.05
df_results['Kendall_sig'] = df_results['Kendall_p'] < 0.05
df_results['LogPearson_sig'] = df_results['LogPearson_p'] < 0.05
df_results['Partial_sig'] = df_results['Partial_p'] < 0.05

# Print summary
print("\n" + "="*100)
print("SIGNIFICANT RESULTS BY METHOD (p < 0.05)")
print("="*100)

methods = ['Spearman', 'Pearson', 'Kendall', 'LogPearson', 'Partial']
for method in methods:
    sig_col = f'{method}_sig'
    r_col = f'{method}_r' if method != 'Kendall' else 'Kendall_tau'
    p_col = f'{method}_p'
    
    sig_results = df_results[df_results[sig_col] == True]
    print(f"\n{method}: {len(sig_results)} significant associations")
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            r_val = row[r_col]
            p_val = row[p_col]
            print(f"  {row['Marker']:6s} - {row['Clinical']:8s}: r={r_val:.3f}, p={p_val:.4f}")

# Comparison table for key associations
print("\n" + "="*100)
print("DETAILED COMPARISON - Key Associations")
print("="*100)

key_pairs = [('GFAP', 'ceradsc'), ('iba1', 'braaksc'), ('iba1', 'nft')]

for marker, clin in key_pairs:
    row = df_results[(df_results['Marker'] == marker) & (df_results['Clinical'] == clin)]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    
    print(f"\n{marker} vs {clin} (n={row['n']}):")
    print(f"  {'Method':<15} {'r/tau':>8} {'p-value':>10} {'Sig?':>6}")
    print(f"  {'-'*40}")
    print(f"  {'Spearman':<15} {row['Spearman_r']:>8.3f} {row['Spearman_p']:>10.4f} {'*' if row['Spearman_sig'] else '':>6}")
    print(f"  {'Pearson':<15} {row['Pearson_r']:>8.3f} {row['Pearson_p']:>10.4f} {'*' if row['Pearson_sig'] else '':>6}")
    print(f"  {'Kendall':<15} {row['Kendall_tau']:>8.3f} {row['Kendall_p']:>10.4f} {'*' if row['Kendall_sig'] else '':>6}")
    if not np.isnan(row['LogPearson_r']):
        print(f"  {'Log-Pearson':<15} {row['LogPearson_r']:>8.3f} {row['LogPearson_p']:>10.4f} {'*' if row['LogPearson_sig'] else '':>6}")
    if not np.isnan(row['Partial_r']):
        print(f"  {'Partial (adj)':<15} {row['Partial_r']:>8.3f} {row['Partial_p']:>10.4f} {'*' if row['Partial_sig'] else '':>6}")

# Save full results
out_path = FULLDATA_ROOT / 'analysis_refined' / 'correlation_methods_comparison.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
df_results.to_csv(out_path, index=False)
print(f"\n\n✓ Full results saved to: {out_path}")

# Summary statistics
print("\n" + "="*100)
print("SUMMARY: Number of significant associations by method")
print("="*100)
summary = {
    'Spearman': df_results['Spearman_sig'].sum(),
    'Pearson': df_results['Pearson_sig'].sum(),
    'Kendall': df_results['Kendall_sig'].sum(),
    'Log-Pearson': df_results['LogPearson_sig'].sum(),
    'Partial (adj)': df_results['Partial_sig'].sum(),
}
for method, count in summary.items():
    print(f"  {method:<15}: {count}")
