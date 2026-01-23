"""
Combined Multi-Panel Figure
A: Stacked barplot (all markers)
B: Scatter vs Braak (all 5 markers)
C: Scatter vs CERAD (all 5 markers)
D: Scatter vs NFT (all 5 markers)

Using Theil-Sen regression (robust, matches Spearman correlation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import theilslopes
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from _paths import SEGMENTATION_REFINED_ROOT, FULLDATA_ROOT

# ============================================================
# Style Settings
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Color palette
COLORS = {
    'NeuN': '#1f77b4',
    'iba1': '#ff7f0e',
    'PECAM': '#2ca02c',
    'GFAP': '#d62728',
    'Olig2': '#9467bd',
}

LABELS = {
    'NeuN': 'Neurons',
    'iba1': 'Microglia',
    'PECAM': 'Endothelial',
    'GFAP': 'Astrocytes',
    'Olig2': 'Oligodendrocytes',
}

TITLES = {
    'NeuN': 'NeuN',
    'iba1': 'IBA1',
    'PECAM': 'PECAM',
    'GFAP': 'GFAP',
    'Olig2': 'OLIG2',
}

# Load data
df = pd.read_csv(SEGMENTATION_REFINED_ROOT / 'donor_level_aggregation_refined.csv')
out_dir = FULLDATA_ROOT / 'analysis_refined'
out_dir.mkdir(parents=True, exist_ok=True)

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

RATIO_COL = lambda m: f'{m}_marker_positive_ratio_refined_mean'

print(f"Loaded {len(df)} donors")

# ============================================================
# Helper Functions
# ============================================================
def add_corr(ax, x, y, fontsize=7): 
    """Add Spearman correlation to plot, with * only when p < 0.05"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    r, p = stats.spearmanr(x[mask], y[mask])
    sig = '*' if p < 0.05 else ''
    ax.text(0.95, 0.95, f'ρ={r:.2f}{sig}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=fontsize)

def add_regline(ax, x, y):
    """Add Theil-Sen regression line (robust, matches Spearman)"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    slope, intercept, _, _ = theilslopes(y[mask], x[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, '--', color='#444444', alpha=0.6, linewidth=0.8)

# ============================================================
# Create Figure
# ============================================================
fig = plt.figure(figsize=(10, 8.5))

gs = fig.add_gridspec(4, 5, height_ratios=[1.3, 0.9, 0.9, 0.9], 
                      hspace=0.4, wspace=0.35,
                      left=0.07, right=0.98, top=0.93, bottom=0.05)

# ============================================================
# Panel A: Stacked Bar Plot (NORMALIZED)
# ============================================================
ax_a = fig.add_subplot(gs[0, :])

ratio_cols = [RATIO_COL(m) for m in markers]
df_stack = df[['projid'] + ratio_cols].dropna().copy()
df_stack.columns = ['projid'] + markers

# Normalize
row_sums = df_stack[markers].sum(axis=1)
for m in markers:
    df_stack[m] = df_stack[m] / row_sums

df_stack = df_stack.sort_values('NeuN', ascending=False).reset_index(drop=True)

stack_order = ['NeuN', 'iba1', 'PECAM', 'GFAP', 'Olig2']
x = np.arange(len(df_stack))
bottom = np.zeros(len(df_stack))

for marker in stack_order:
    values = df_stack[marker].values
    ax_a.bar(x, values, bottom=bottom, width=1.0,
             color=COLORS[marker], edgecolor='none')
    bottom += values

ax_a.set_xlim(-0.5, len(df_stack) - 0.5)
ax_a.set_ylim(0, 1)
ax_a.set_ylabel('Proportion')
ax_a.set_xticks([])
ax_a.set_xlabel(f'Donors')
ax_a.set_yticks([0, 0.5, 1])

handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in stack_order]
ax_a.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=5, frameon=False, fontsize=8)
ax_a.text(-0.05, 1.02, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='bottom')

# ============================================================
# Panel B: Scatter vs Braak Stage
# ============================================================
for i, marker in enumerate(markers):
    ax = fig.add_subplot(gs[1, i])
    col = RATIO_COL(marker)
    
    mask = df[[col, 'braaksc']].notna().all(axis=1)
    x_data = df.loc[mask, 'braaksc'].values
    y_data = df.loc[mask, col].values
    
    ax.scatter(x_data, y_data, c=COLORS[marker], alpha=0.7, s=20, 
               edgecolors='white', linewidth=0.4)
    add_regline(ax, x_data, y_data)
    add_corr(ax, x_data, y_data)
    
    ax.set_title(TITLES[marker], fontsize=9, fontweight='bold')
    ax.set_xticks([0, 3, 6])
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylabel('Cell Fraction')
    
    if i == 2:
        ax.set_xlabel('Braak stage')
    
    if i == 0:
        ax.text(-0.3, 1.02, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')

# ============================================================
# Panel C: Scatter vs CERAD Score
# ============================================================
for i, marker in enumerate(markers):
    ax = fig.add_subplot(gs[2, i])
    col = RATIO_COL(marker)
    
    mask = df[[col, 'ceradsc']].notna().all(axis=1)
    x_data = df.loc[mask, 'ceradsc'].values
    y_data = df.loc[mask, col].values
    
    ax.scatter(x_data, y_data, c=COLORS[marker], alpha=0.7, s=20,
               edgecolors='white', linewidth=0.4)
    add_regline(ax, x_data, y_data)
    add_corr(ax, x_data, y_data)
    
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylabel('Cell Fraction')
    
    if i == 2:
        ax.set_xlabel('CERAD score')
    
    if i == 0:
        ax.text(-0.3, 1.02, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')

# ============================================================
# Panel D: Scatter vs NFT
# ============================================================
for i, marker in enumerate(markers):
    ax = fig.add_subplot(gs[3, i])
    col = RATIO_COL(marker)
    
    mask = df[[col, 'nft']].notna().all(axis=1)
    x_data = df.loc[mask, 'nft'].values
    y_data = df.loc[mask, col].values
    
    ax.scatter(x_data, y_data, c=COLORS[marker], alpha=0.7, s=20,
               edgecolors='white', linewidth=0.4)
    add_regline(ax, x_data, y_data)
    add_corr(ax, x_data, y_data)
    
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Cell Fraction')
    
    if i == 2:
        ax.set_xlabel('Neurofibrillary tangles')
    
    if i == 0:
        ax.text(-0.3, 1.02, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')

# ============================================================
# Save
# ============================================================
plt.savefig(out_dir / 'figure_main.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_dir / 'figure_main.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"\n✓ Saved: {out_dir / 'figure_main.pdf'}")
print(f"✓ Saved: {out_dir / 'figure_main.png'}")

# ============================================================
# Print exact p-values for all correlations
# ============================================================
print("\n" + "="*70)
print("EXACT P-VALUES FOR SPEARMAN CORRELATIONS")
print("="*70)

headers = ["Marker", "Vs Braak", "", "Vs CERAD", "", "Vs NFT", ""]
print(f"{headers[0]:<10} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12} {headers[6]:<12}")
print("-"*70)

for marker in markers:
    col = RATIO_COL(marker)

    # Braak
    mask = df[[col, 'braaksc']].notna().all(axis=1)
    if mask.sum() >= 3:
        r_braak, p_braak = stats.spearmanr(df.loc[mask, 'braaksc'], df.loc[mask, col])
        braak_str = f"ρ={r_braak:.3f}"
        braak_p = f"p={p_braak:.4f}"
    else:
        braak_str = "N/A"
        braak_p = "N/A"

    # CERAD
    mask = df[[col, 'ceradsc']].notna().all(axis=1)
    if mask.sum() >= 3:
        r_cerad, p_cerad = stats.spearmanr(df.loc[mask, 'ceradsc'], df.loc[mask, col])
        cerad_str = f"ρ={r_cerad:.3f}"
        cerad_p = f"p={p_cerad:.4f}"
    else:
        cerad_str = "N/A"
        cerad_p = "N/A"

    # NFT
    mask = df[[col, 'nft']].notna().all(axis=1)
    if mask.sum() >= 3:
        r_nft, p_nft = stats.spearmanr(df.loc[mask, 'nft'], df.loc[mask, col])
        nft_str = f"ρ={r_nft:.3f}"
        nft_p = f"p={p_nft:.4f}"
    else:
        nft_str = "N/A"
        nft_p = "N/A"

    print(f"{marker:<10} {braak_str:<12} {braak_p:<12} {cerad_str:<12} {cerad_p:<12} {nft_str:<12} {nft_p:<12}")

print("="*70)
