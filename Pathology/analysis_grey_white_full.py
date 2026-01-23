"""
Comprehensive Statistical Analysis - Cellpose vs Watershed
No figures, only statistics and tables
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

from _paths import FULLDATA_ROOT
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_PATH = FULLDATA_ROOT
out_dir = BASE_PATH / 'analysis_results'
out_dir.mkdir(exist_ok=True)

MARKERS = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']
ALGORITHMS = ['cellpose', 'watershed']
REGIONS = ['grey', 'white', 'full']

clinical_vars = ['cogdx', 'braaksc', 'ceradsc', 'plaq_d', 'plaq_n', 'nft', 'gpath']

# ============================================================
# Load Data
# ============================================================
print("="*60)
print("Loading data...")
print("="*60)

data = {}
for algo in ALGORITHMS:
    algo_dir = BASE_PATH / f'aggregation_{algo}'
    if not algo_dir.exists():
        print(f"  ⚠️ {algo} not found")
        continue
    
    data[algo] = {}
    
    for region in REGIONS:
        if region == 'full':
            file_path = algo_dir / 'donor_level_aggregation.csv'
        else:
            file_path = algo_dir / f'donor_{region}_aggregation.csv'
        
        if file_path.exists():
            data[algo][region] = pd.read_csv(file_path)
            print(f"  {algo} {region}: {len(data[algo][region])} donors")

# ============================================================
# Helper Functions
# ============================================================
def get_ratio_col(marker, df):
    """Find the ratio column for a marker."""
    candidates = [
        f'{marker}_marker_positive_ratio',
        f'{marker}_ratio_from_sum',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None

def get_count_col(marker, df, count_type='marker'):
    """Find the cell count column."""
    candidates = [
        f'{marker}_{count_type}_cell_count_sum',
        f'{marker}_n_cells_sum',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None

def compute_correlation(df, marker_col, clin_col):
    """Compute Spearman correlation."""
    if marker_col is None or marker_col not in df.columns or clin_col not in df.columns:
        return np.nan, 1.0, 0
    mask = df[[marker_col, clin_col]].notna().all(axis=1)
    n = mask.sum()
    if n < 3:
        return np.nan, 1.0, n
    r, p = stats.spearmanr(df.loc[mask, marker_col], df.loc[mask, clin_col])
    return r, p, n

# ============================================================
# 1. Compute All Correlations
# ============================================================
print("\n" + "="*60)
print("Computing correlations...")
print("="*60)

results = []

for algo in ALGORITHMS:
    if algo not in data:
        continue
    for region in REGIONS:
        if region not in data[algo]:
            continue
        df = data[algo][region]
        
        for marker in MARKERS:
            # Ratio
            ratio_col = get_ratio_col(marker, df)
            for clin in clinical_vars:
                r, p, n = compute_correlation(df, ratio_col, clin)
                results.append({
                    'algorithm': algo,
                    'region': region,
                    'marker': marker,
                    'metric': 'ratio',
                    'clinical_var': clin,
                    'spearman_r': r,
                    'p_value': p,
                    'n': n,
                    'significant_005': p < 0.05,
                    'significant_001': p < 0.01,
                    'significant_0001': p < 0.001
                })
            
            # Marker cell count
            count_col = get_count_col(marker, df, 'marker')
            for clin in clinical_vars:
                r, p, n = compute_correlation(df, count_col, clin)
                results.append({
                    'algorithm': algo,
                    'region': region,
                    'marker': marker,
                    'metric': 'marker_count',
                    'clinical_var': clin,
                    'spearman_r': r,
                    'p_value': p,
                    'n': n,
                    'significant_005': p < 0.05,
                    'significant_001': p < 0.01,
                    'significant_0001': p < 0.001
                })

df_results = pd.DataFrame(results)
df_results = df_results[df_results['n'] > 0]  # Remove empty rows

# Save full correlation table
df_results.to_csv(out_dir / 'correlation_table_full.csv', index=False)
print(f"\n✓ Saved: correlation_table_full.csv ({len(df_results)} rows)")

# ============================================================
# 2. Summary Tables
# ============================================================
print("\n" + "="*60)
print("Generating summary tables...")
print("="*60)

# 2a. Significant correlations summary
sig_summary = df_results[df_results['significant_005']].groupby(
    ['algorithm', 'region', 'metric']
).size().reset_index(name='n_significant')
sig_summary.to_csv(out_dir / 'significant_count_summary.csv', index=False)
print(f"✓ Saved: significant_count_summary.csv")

# 2b. Pivot table: Algorithm x Region x Marker (for ratio, Braak)
pivot_braak = df_results[
    (df_results['metric'] == 'ratio') & 
    (df_results['clinical_var'] == 'braaksc')
].pivot_table(
    index='marker',
    columns=['algorithm', 'region'],
    values=['spearman_r', 'p_value']
)
pivot_braak.to_csv(out_dir / 'pivot_ratio_vs_braak.csv')
print(f"✓ Saved: pivot_ratio_vs_braak.csv")

# 2c. Pivot table: Algorithm x Region x Marker (for ratio, cogdx)
pivot_cogdx = df_results[
    (df_results['metric'] == 'ratio') & 
    (df_results['clinical_var'] == 'cogdx')
].pivot_table(
    index='marker',
    columns=['algorithm', 'region'],
    values=['spearman_r', 'p_value']
)
pivot_cogdx.to_csv(out_dir / 'pivot_ratio_vs_cogdx.csv')
print(f"✓ Saved: pivot_ratio_vs_cogdx.csv")

# 2d. Best correlations per marker
best_per_marker = df_results[df_results['metric'] == 'ratio'].loc[
    df_results[df_results['metric'] == 'ratio'].groupby('marker')['p_value'].idxmin()
]
best_per_marker.to_csv(out_dir / 'best_correlation_per_marker.csv', index=False)
print(f"✓ Saved: best_correlation_per_marker.csv")

# ============================================================
# 3. Algorithm Comparison
# ============================================================
print("\n" + "="*60)
print("Algorithm comparison...")
print("="*60)

algo_comparison = []

for region in REGIONS:
    for marker in MARKERS:
        for clin in clinical_vars:
            row = {'region': region, 'marker': marker, 'clinical_var': clin}
            
            for algo in ALGORITHMS:
                subset = df_results[
                    (df_results['algorithm'] == algo) &
                    (df_results['region'] == region) &
                    (df_results['marker'] == marker) &
                    (df_results['metric'] == 'ratio') &
                    (df_results['clinical_var'] == clin)
                ]
                if len(subset) > 0:
                    row[f'{algo}_r'] = subset['spearman_r'].values[0]
                    row[f'{algo}_p'] = subset['p_value'].values[0]
                else:
                    row[f'{algo}_r'] = np.nan
                    row[f'{algo}_p'] = np.nan
            
            # Compute difference
            if 'cellpose_r' in row and 'watershed_r' in row:
                row['r_diff'] = row['cellpose_r'] - row['watershed_r']
            
            algo_comparison.append(row)

df_algo_comp = pd.DataFrame(algo_comparison)
df_algo_comp.to_csv(out_dir / 'algorithm_comparison.csv', index=False)
print(f"✓ Saved: algorithm_comparison.csv")

# ============================================================
# 4. Region Comparison
# ============================================================
print("\n" + "="*60)
print("Region comparison...")
print("="*60)

region_comparison = []

for algo in ALGORITHMS:
    for marker in MARKERS:
        for clin in clinical_vars:
            row = {'algorithm': algo, 'marker': marker, 'clinical_var': clin}
            
            for region in REGIONS:
                subset = df_results[
                    (df_results['algorithm'] == algo) &
                    (df_results['region'] == region) &
                    (df_results['marker'] == marker) &
                    (df_results['metric'] == 'ratio') &
                    (df_results['clinical_var'] == clin)
                ]
                if len(subset) > 0:
                    row[f'{region}_r'] = subset['spearman_r'].values[0]
                    row[f'{region}_p'] = subset['p_value'].values[0]
                else:
                    row[f'{region}_r'] = np.nan
                    row[f'{region}_p'] = np.nan
            
            # Compute grey-white difference
            if 'grey_r' in row and 'white_r' in row:
                row['grey_white_diff'] = row['grey_r'] - row['white_r']
            
            region_comparison.append(row)

df_region_comp = pd.DataFrame(region_comparison)
df_region_comp.to_csv(out_dir / 'region_comparison.csv', index=False)
print(f"✓ Saved: region_comparison.csv")

# ============================================================
# 5. Descriptive Statistics
# ============================================================
print("\n" + "="*60)
print("Descriptive statistics...")
print("="*60)

desc_stats = []

for algo in ALGORITHMS:
    if algo not in data:
        continue
    for region in REGIONS:
        if region not in data[algo]:
            continue
        df = data[algo][region]
        
        for marker in MARKERS:
            ratio_col = get_ratio_col(marker, df)
            count_col = get_count_col(marker, df, 'marker')
            
            row = {
                'algorithm': algo,
                'region': region,
                'marker': marker,
                'n_donors': len(df)
            }
            
            if ratio_col and ratio_col in df.columns:
                vals = df[ratio_col].dropna()
                row['ratio_mean'] = vals.mean()
                row['ratio_std'] = vals.std()
                row['ratio_median'] = vals.median()
                row['ratio_min'] = vals.min()
                row['ratio_max'] = vals.max()
                row['ratio_n'] = len(vals)
            
            if count_col and count_col in df.columns:
                vals = df[count_col].dropna()
                row['count_mean'] = vals.mean()
                row['count_std'] = vals.std()
                row['count_median'] = vals.median()
                row['count_sum'] = vals.sum()
                row['count_n'] = len(vals)
            
            desc_stats.append(row)

df_desc = pd.DataFrame(desc_stats)
df_desc.to_csv(out_dir / 'descriptive_statistics.csv', index=False)
print(f"✓ Saved: descriptive_statistics.csv")

# ============================================================
# 6. Print Summary Report
# ============================================================
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

print("\n📊 Data Overview:")
for algo in ALGORITHMS:
    if algo not in data:
        continue
    print(f"\n  {algo.upper()}:")
    for region in REGIONS:
        if region in data[algo]:
            print(f"    {region}: {len(data[algo][region])} donors")

print("\n📈 Significant Correlations (p < 0.05):")
for algo in ALGORITHMS:
    print(f"\n  {algo.upper()}:")
    for region in REGIONS:
        subset = df_results[
            (df_results['algorithm'] == algo) &
            (df_results['region'] == region) &
            (df_results['significant_005'])
        ]
        n_ratio = len(subset[subset['metric'] == 'ratio'])
        n_count = len(subset[subset['metric'] == 'marker_count'])
        print(f"    {region}: {n_ratio} ratio, {n_count} count")

print("\n🏆 Top Correlations (Ratio vs Clinical):")
top_corr = df_results[
    (df_results['metric'] == 'ratio') & 
    (df_results['significant_005'])
].nsmallest(10, 'p_value')

for _, row in top_corr.iterrows():
    print(f"  {row['algorithm']}/{row['region']}: {row['marker']} vs {row['clinical_var']}: "
          f"r={row['spearman_r']:.3f}, p={row['p_value']:.4f}")

print("\n🔬 Grey vs White Matter Differences (|Δr| > 0.1):")
big_diff = df_region_comp[
    (df_region_comp['grey_white_diff'].abs() > 0.1) &
    (df_region_comp['grey_white_diff'].notna())
].sort_values('grey_white_diff', key=abs, ascending=False)

for _, row in big_diff.head(10).iterrows():
    print(f"  {row['algorithm']}: {row['marker']} vs {row['clinical_var']}: "
          f"grey r={row['grey_r']:.3f}, white r={row['white_r']:.3f}, Δ={row['grey_white_diff']:.3f}")

print("\n🔄 Algorithm Differences (|Δr| > 0.1):")
big_algo_diff = df_algo_comp[
    (df_algo_comp['r_diff'].abs() > 0.1) &
    (df_algo_comp['r_diff'].notna())
].sort_values('r_diff', key=abs, ascending=False)

for _, row in big_algo_diff.head(10).iterrows():
    print(f"  {row['region']}: {row['marker']} vs {row['clinical_var']}: "
          f"cellpose r={row['cellpose_r']:.3f}, watershed r={row['watershed_r']:.3f}, Δ={row['r_diff']:.3f}")

# ============================================================
# 7. Output Summary
# ============================================================
print("\n" + "="*60)
print("OUTPUT FILES")
print("="*60)
print(f"\nDirectory: {out_dir}")
print("\nFiles:")
print("  - correlation_table_full.csv        (all correlations)")
print("  - significant_count_summary.csv     (count of significant)")
print("  - pivot_ratio_vs_braak.csv          (ratio vs Braak pivot)")
print("  - pivot_ratio_vs_cogdx.csv          (ratio vs CogDx pivot)")
print("  - best_correlation_per_marker.csv   (best per marker)")
print("  - algorithm_comparison.csv          (cellpose vs watershed)")
print("  - region_comparison.csv             (grey vs white vs full)")
print("  - descriptive_statistics.csv        (mean, std, etc.)")
print("="*60)
