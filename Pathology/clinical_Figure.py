"""
Combined Figure: Cell Type Analysis
A: Stacked bar plot
B, C: Heatmaps (CellposeSAM, Watershed)
D, E, F, G: Scatter/Box plots

Author: Liang
Date: 2025
Updated: 2026-01-21 - Metrics now grouped in blocks for better readability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import matplotlib.gridspec as gridspec
from scipy import stats

# ============================================================
# ==================== ADJUSTABLE PARAMETERS =================
# ============================================================

# === FILE PATHS ===
DATA_FILE = 'full_aggregation.csv'
CORR_FILE = 'correlation_analysis.csv'
OUTPUT_PNG = 'combined_figure.png'
OUTPUT_PDF = 'combined_figure.pdf'

# === FIGURE SIZE AND LAYOUT ===
FIG_WIDTH = 18
FIG_HEIGHT = 14
DPI = 300

# Row heights: [Panel A, Panel B/C, Spacer, Panel D-G]
ROW_HEIGHTS = [0.9, 2.4, 0.15, 1.5]
ROW_HSPACE = 0.25           # Vertical space between rows
FIG_TOP = 0.95              # Top margin
FIG_BOTTOM = 0.05           # Bottom margin

# === PANEL A: STACKED BAR ===
PANEL_A_YLABEL_SIZE = 14
PANEL_A_YTICK_SIZE = 13
PANEL_A_LEGEND_SIZE = 13
PANEL_A_LEGEND_NCOL = 5
PANEL_A_LEGEND_Y = 1.28     # Legend vertical position

# === PANEL B/C: HEATMAPS ===
HEATMAP_WIDTH_RATIOS = [1, 1, 0.05, 0.28]  # [B, C, colorbar, legend]
HEATMAP_WSPACE = 0.15       # Horizontal space between heatmaps

HEATMAP_TITLE_SIZE = 16
HEATMAP_XTICK_SIZE = 14
HEATMAP_XTICK_ROTATION = 45

HEATMAP_VMIN = -0.4         # Colorbar min
HEATMAP_VMAX = 0.4          # Colorbar max
COLORBAR_TICK_SIZE = 11

# Side bar settings
SIDEBAR_WIDTH = 0.34
SIDEBAR_GAP = 0.06
SIDEBAR_LABEL_SIZE = 13
SIDEBAR_LABEL_ROTATION = 45

# Legend settings
LEGEND_REGION_SIZE = 12
LEGEND_METRIC_SIZE = 12
LEGEND_PHENO_SIZE = 11
LEGEND_TITLE_SIZE = 13

# Significance asterisk
SIG_MARKER = '*'
SIG_COLOR = 'black'
SIG_SIZE = 14

# === PANEL D-G: SCATTER/BOX PLOTS ===
SCATTER_WSPACE = 0.35       # Horizontal space between plots

SCATTER_TITLE_SIZE = 16
SCATTER_XLABEL_SIZE = 13
SCATTER_YLABEL_SIZE = 13
SCATTER_TICK_SIZE = 11
SCATTER_PVAL_SIZE = 12

# Scatter plot points
SCATTER_POINT_SIZE = 50
SCATTER_POINT_ALPHA = 0.7

# Box plot settings
BOX_WIDTH = 0.6
BOX_ALPHA = 0.7
BOX_JITTER_SIZE = 20
BOX_JITTER_ALPHA = 0.5

# Regression line
REG_LINE_WIDTH = 2.5
REG_CI_ALPHA = 0.15
REG_N_BOOTSTRAP = 500

# === PANEL LABELS ===
PANEL_LABEL_SIZE = 20
PANEL_LABEL_FONTWEIGHT = 'bold'

# === COLORS ===
# Cell type colors (consistent across all panels)
COLORS = {
    'GFAP': '#FFA726',   # Red
    'iba1': '#4DBBD5',   # Cyan
    'NeuN': '#00A087',   # Teal
    'Olig2': '#3C5488',  # Blue
    'PECAM': '#AB47BC'   # Salmon
}

# Display labels (standardized names)
LABELS = {
    'GFAP': 'GFAP',
    'iba1': 'IBA1',
    'NeuN': 'NeuN',
    'Olig2': 'OLIG2',
    'PECAM': 'PECAM'
}

# Region colors
REGION_COLORS = {
    'grey': '#7E6148',
    'white': '#B0ACA4'
}

# Region display labels
REGION_LABELS = {
    'grey': 'Grey Matter',
    'white': 'White Matter'
}

# Phenotype colors (adjusted to differentiate CERAD and Neuritic Plaque)
PHENOTYPE_COLORS = {
    'Braak': '#3C5488',           # Blue
    'CERAD': '#FF9933',           # Orange (was salmon-ish, now distinct)
    'NFT': '#8491B4',             # Light blue
    'Diffuse Plaque': '#B09C85',  # Tan
    'Neuritic Plaque': '#DC3545', # Red (distinct from orange)
    'Cognitive Dx': '#FFCA28',    # Sage green
    'Global Path': '#26A69A'      # Teal
}

# Metric colors
METRIC_COLORS = {
    'fraction': '#8B5CF6',   # Purple
    'count': '#F59E0B'       # Orange
}

# Metric display labels
METRIC_LABELS = {
    'fraction': 'Fraction',
    'count': 'Count'
}

# === DATA SETTINGS ===
MARKERS = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']
MARKERS_DISPLAY = ['GFAP', 'IBA1', 'NeuN', 'OLIG2', 'PECAM']
PHENOTYPES = ['Braak', 'CERAD', 'NFT', 'Diffuse Plaque', 'Neuritic Plaque', 'Cognitive Dx', 'Global Path']
REGIONS = ['grey', 'white']
METRICS = ['fraction', 'count']  # Changed from ratio to fraction

# Stacked bar order (bottom to top)
STACK_ORDER = ['NeuN', 'iba1', 'PECAM', 'GFAP', 'Olig2']

# Clinical variable mapping
CLINICAL_MAP = {
    'braaksc': 'Braak',
    'ceradsc': 'CERAD',
    'nft': 'NFT',
    'plaq_d': 'Diffuse Plaque',
    'plaq_n': 'Neuritic Plaque',
    'cogdx': 'Cognitive Dx',
    'gpath': 'Global Path'
}

# Clinical variable info: (display_name, is_discrete)
CLINICAL_INFO = {
    'cogdx': ('Cognitive Diagnosis', True),
    'braaksc': ('Braak Stage', True),
    'ceradsc': ('CERAD Score', True),
    'nft': ('NFT Density', False),
    'plaq_n': ('Neuritic Plaque', False),
    'plaq_d': ('Diffuse Plaque', False),
    'gpath': ('Global Pathology', False)
}

# ============================================================
# ==================== END OF PARAMETERS =====================
# ============================================================


# ============================================================
# Helper Functions
# ============================================================
def build_heatmap_data(df_filtered, algo):
    """Build correlation matrix and significance matrix for heatmap"""
    n_cells = len(MARKERS)
    total_rows = len(REGIONS) * len(PHENOTYPES) * len(METRICS)
    corr_matrix = np.zeros((total_rows, n_cells))
    sig_matrix = np.zeros((total_rows, n_cells), dtype=bool)
    row_info = []
    
    row_idx = 0
    # Changed loop order: region -> metric -> phenotype
    # This groups all fractions together, then all counts together
    for region in REGIONS:
        for metric in METRICS:
            for phenotype in PHENOTYPES:
                for k, marker in enumerate(MARKERS):
                    # Map 'fraction' back to 'ratio' for data lookup
                    data_metric = 'ratio' if metric == 'fraction' else metric
                    subset = df_filtered[
                        (df_filtered['Region'] == region) &
                        (df_filtered['Phenotype'] == phenotype) &
                        (df_filtered['Algorithm'] == algo) &
                        (df_filtered['Marker'] == marker) &
                        (df_filtered['Metric'] == data_metric)
                    ]
                    if len(subset) > 0:
                        corr_matrix[row_idx, k] = subset['Spearman_r'].values[0]
                        sig_matrix[row_idx, k] = subset['Spearman_sig'].values[0]
                    else:
                        corr_matrix[row_idx, k] = np.nan
                row_info.append({'region': region, 'phenotype': phenotype, 'metric': metric})
                row_idx += 1
    
    return corr_matrix, sig_matrix, row_info


def find_blocks(row_info, key):
    """Find consecutive blocks of same value for side bars"""
    blocks = []
    current_value = None
    start_idx = None
    for i, info in enumerate(row_info):
        value = info[key]
        if value != current_value:
            if current_value is not None:
                blocks.append((current_value, start_idx, i))
            current_value = value
            start_idx = i
    if current_value is not None:
        blocks.append((current_value, start_idx, len(row_info)))
    return blocks


def robust_regression_with_ci(x, y):
    """Theil-Sen robust regression with bootstrap confidence interval"""
    x_pred = np.linspace(x.min(), x.max(), 100)
    res = stats.theilslopes(y, x)
    slope, intercept = res[0], res[1]
    y_pred = slope * x_pred + intercept
    
    n = len(x)
    boot_curves = []
    np.random.seed(42)
    for _ in range(REG_N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        try:
            res_boot = stats.theilslopes(y[idx], x[idx])
            y_boot = res_boot[0] * x_pred + res_boot[1]
            boot_curves.append(y_boot)
        except:
            continue
    
    boot_curves = np.array(boot_curves)
    ci_low = np.percentile(boot_curves, 2.5, axis=0)
    ci_high = np.percentile(boot_curves, 97.5, axis=0)
    
    return x_pred, y_pred, ci_low, ci_high


# ============================================================
# Main Script
# ============================================================
if __name__ == '__main__':
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    corr_df = pd.read_csv(CORR_FILE)
    
    # Prepare filtered correlation data
    df_filtered = corr_df[corr_df['Region'].isin(REGIONS)].copy()
    df_filtered['Phenotype'] = df_filtered['Clinical'].map(CLINICAL_MAP)
    
    # ============================================================
    # Create Figure
    # ============================================================
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    gs_main = gridspec.GridSpec(4, 1, figure=fig, height_ratios=ROW_HEIGHTS, 
                                hspace=ROW_HSPACE, top=FIG_TOP, bottom=FIG_BOTTOM)
    
    # ============================================================
    # Panel A: Stacked Bar Plot
    # ============================================================
    ax_a = fig.add_subplot(gs_main[0])
    
    df_main = df[(df['algorithm'] == 'cellpose') & (df['region'] == 'grey')].copy()
    ratio_cols = [f'{m}_ratio' for m in MARKERS]
    df_stack = df_main[['participant_id'] + ratio_cols].dropna().copy()
    df_stack.columns = ['projid'] + MARKERS
    
    row_sums = df_stack[MARKERS].sum(axis=1)
    for m in MARKERS:
        df_stack[m] = df_stack[m] / row_sums
    
    df_stack = df_stack.sort_values('NeuN', ascending=False).reset_index(drop=True)
    
    x = np.arange(len(df_stack))
    bottom = np.zeros(len(df_stack))
    
    for marker in STACK_ORDER:
        values = df_stack[marker].values
        ax_a.bar(x, values, bottom=bottom, width=1.0,
                 color=COLORS[marker], edgecolor='none')
        bottom += values
    
    ax_a.set_xlim(-0.5, len(df_stack) - 0.5)
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel('Proportion', fontsize=PANEL_A_YLABEL_SIZE)
    ax_a.set_xticks([])
    ax_a.set_yticks([0, 0.5, 1])
    
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.tick_params(axis='y', length=0, labelsize=PANEL_A_YTICK_SIZE)
    
    handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in STACK_ORDER]
    ax_a.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, PANEL_A_LEGEND_Y),
                ncol=PANEL_A_LEGEND_NCOL, frameon=False, fontsize=PANEL_A_LEGEND_SIZE)
    ax_a.text(-0.02, 1.08, 'A', transform=ax_a.transAxes, 
              fontsize=PANEL_LABEL_SIZE, fontweight=PANEL_LABEL_FONTWEIGHT, va='bottom')
    
    # ============================================================
    # Panels B, C: Heatmaps
    # ============================================================
    gs_heatmaps = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1], 
                                                    width_ratios=HEATMAP_WIDTH_RATIOS, 
                                                    wspace=HEATMAP_WSPACE)
    
    algorithms = ['cellpose', 'watershed']
    titles = ['CellposeSAM', 'Watershed']
    panel_labels = ['B', 'C']
    
    cmap = plt.cm.RdYlBu_r
    im = None
    
    for idx, (algo, title, panel) in enumerate(zip(algorithms, titles, panel_labels)):
        ax = fig.add_subplot(gs_heatmaps[idx])
        
        corr_matrix, sig_matrix, row_info = build_heatmap_data(df_filtered, algo)
        total_rows = len(row_info)
        
        mask = np.isnan(corr_matrix)
        im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
        
        # Significance markers
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if sig_matrix[i, j] and not mask[i, j]:
                    ax.text(float(j), float(i) + 0.1, SIG_MARKER, 
                           ha='center', va='center', color=SIG_COLOR, 
                           fontsize=SIG_SIZE, fontweight='bold')
        
        ax.set_xticks(np.arange(len(MARKERS)))
        ax.set_xticklabels(MARKERS_DISPLAY, fontsize=HEATMAP_XTICK_SIZE, 
                          fontweight='bold', rotation=HEATMAP_XTICK_ROTATION, ha='right')
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
        
        ax.set_title(title, fontsize=HEATMAP_TITLE_SIZE, fontweight='bold', pad=10)
        ax.text(-0.1, 1.02, panel, transform=ax.transAxes, 
                fontsize=PANEL_LABEL_SIZE, fontweight=PANEL_LABEL_FONTWEIGHT, va='bottom')
        
        # Side bars - ORDER: Region, Metric, Phenotype (left to right)
        xlim = ax.get_xlim()
        heatmap_left = xlim[0]
        region_offset = heatmap_left - 3 * SIDEBAR_WIDTH - 3 * SIDEBAR_GAP
        metric_offset = heatmap_left - 2 * SIDEBAR_WIDTH - 2 * SIDEBAR_GAP  # Swapped
        pheno_offset = heatmap_left - SIDEBAR_WIDTH - SIDEBAR_GAP           # Swapped
        
        for region, start, end in find_blocks(row_info, 'region'):
            ax.add_patch(Rectangle((region_offset, start - 0.5), SIDEBAR_WIDTH, end - start, 
                                    color=REGION_COLORS[region], clip_on=False, linewidth=0))
        
        for metric, start, end in find_blocks(row_info, 'metric'):
            ax.add_patch(Rectangle((metric_offset, start - 0.5), SIDEBAR_WIDTH, end - start, 
                                    color=METRIC_COLORS[metric], clip_on=False, linewidth=0))
        
        for phenotype, start, end in find_blocks(row_info, 'phenotype'):
            ax.add_patch(Rectangle((pheno_offset, start - 0.5), SIDEBAR_WIDTH, end - start, 
                                    color=PHENOTYPE_COLORS[phenotype], clip_on=False, linewidth=0))
        
        ax.set_xlim(region_offset - 0.1, xlim[1])
        
        bar_label_y = total_rows - 0.5 + 0.6
        ax.text(region_offset + SIDEBAR_WIDTH/2, bar_label_y, 'Region', ha='right', va='top', 
                fontsize=SIDEBAR_LABEL_SIZE, fontweight='bold', rotation=SIDEBAR_LABEL_ROTATION)
        ax.text(metric_offset + SIDEBAR_WIDTH/2, bar_label_y, 'Metric', ha='right', va='top', 
                fontsize=SIDEBAR_LABEL_SIZE, fontweight='bold', rotation=SIDEBAR_LABEL_ROTATION)
        ax.text(pheno_offset + SIDEBAR_WIDTH/2, bar_label_y, 'Phenotype', ha='right', va='top', 
                fontsize=SIDEBAR_LABEL_SIZE, fontweight='bold', rotation=SIDEBAR_LABEL_ROTATION)
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs_heatmaps[2])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([HEATMAP_VMIN, HEATMAP_VMIN/2, 0, HEATMAP_VMAX/2, HEATMAP_VMAX])
    cbar.ax.tick_params(size=0, labelsize=COLORBAR_TICK_SIZE)
    cbar.outline.set_visible(False)
    
    # Legend for heatmap
    ax_leg = fig.add_subplot(gs_heatmaps[3])
    ax_leg.axis('off')
    
    # Region legend with "matter" labels
    region_patches = [Patch(facecolor=REGION_COLORS[r], label=REGION_LABELS[r], edgecolor='none') for r in REGIONS]
    leg_region = ax_leg.legend(handles=region_patches, loc='upper left', bbox_to_anchor=(0.0, 0.98),
                               ncol=1, frameon=False, fontsize=LEGEND_REGION_SIZE, 
                               title='Region', title_fontsize=LEGEND_TITLE_SIZE,
                               handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
    leg_region.get_title().set_fontweight('bold')
    ax_leg.add_artist(leg_region)
    
    # Metric legend with Fraction/Count labels
    metric_patches = [Patch(facecolor=METRIC_COLORS[m], label=METRIC_LABELS[m], edgecolor='none') for m in METRICS]
    leg_metric = ax_leg.legend(handles=metric_patches, loc='upper left', bbox_to_anchor=(0.0, 0.78),
                               ncol=1, frameon=False, fontsize=LEGEND_METRIC_SIZE, 
                               title='Metric', title_fontsize=LEGEND_TITLE_SIZE,
                               handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
    leg_metric.get_title().set_fontweight('bold')
    ax_leg.add_artist(leg_metric)
    
    phenotype_patches = [Patch(facecolor=PHENOTYPE_COLORS[p], label=p, edgecolor='none') for p in PHENOTYPES]
    leg_pheno = ax_leg.legend(handles=phenotype_patches, loc='upper left', bbox_to_anchor=(0.0, 0.58),
                              ncol=1, frameon=False, fontsize=LEGEND_PHENO_SIZE, 
                              title='Phenotype', title_fontsize=LEGEND_TITLE_SIZE,
                              handlelength=1.5, handletextpad=0.4, labelspacing=0.2)
    leg_pheno.get_title().set_fontweight('bold')
    ax_leg.add_artist(leg_pheno)
    
    # ============================================================
    # Panels D, E, F, G: Scatter/Box plots
    # ============================================================
    gs_scatter = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[3], wspace=SCATTER_WSPACE)
    
    sig = corr_df[(corr_df['Spearman_sig'] == True) & 
                  (corr_df['Algorithm'] == 'cellpose') &
                  (corr_df['Marker'] != 'PECAM') &
                  (corr_df['Region'] == 'grey') &
                  (corr_df['Metric'] == 'count')].copy()
    
    scatter_labels = ['D', 'E', 'F', 'G']
    
    for idx, (_, row) in enumerate(sig.iterrows()):
        ax = fig.add_subplot(gs_scatter[idx])
        
        marker = row['Marker']
        clinical = row['Clinical']
        p = row['Spearman_p']
        
        subset = df[(df['algorithm'] == 'cellpose') & (df['region'] == 'grey')].copy()
        
        y_col = f'{marker}_positive_sum'
        y_label = 'Cell Count'
        
        subset = subset.dropna(subset=[clinical, y_col])
        
        clin_label, is_discrete = CLINICAL_INFO[clinical]
        marker_color = COLORS[marker]
        
        if is_discrete:
            groups = sorted(subset[clinical].dropna().unique())
            box_data = [subset[subset[clinical] == g][y_col].values for g in groups]
            
            bp = ax.boxplot(box_data, positions=range(len(groups)), widths=BOX_WIDTH, 
                            patch_artist=True, flierprops=dict(marker='o', markersize=4, alpha=0.5))
            
            for patch in bp['boxes']:
                patch.set_facecolor(marker_color)
                patch.set_alpha(BOX_ALPHA)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            
            np.random.seed(42)
            for i, g in enumerate(groups):
                y_data = subset[subset[clinical] == g][y_col].values
                x_jitter = np.random.normal(i, 0.1, len(y_data))
                ax.scatter(x_jitter, y_data, color='black', s=BOX_JITTER_SIZE, 
                          alpha=BOX_JITTER_ALPHA, zorder=3)
            
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([str(int(g)) for g in groups], fontsize=SCATTER_TICK_SIZE)
            
        else:
            x_vals = subset[clinical].values
            y_vals = subset[y_col].values
            
            ax.scatter(x_vals, y_vals, color=marker_color, s=SCATTER_POINT_SIZE, 
                      alpha=SCATTER_POINT_ALPHA, edgecolor='white', linewidth=0.5)
            
            x_pred, y_pred, ci_low, ci_high = robust_regression_with_ci(x_vals, y_vals)
            ax.plot(x_pred, y_pred, color='black', linewidth=REG_LINE_WIDTH, zorder=5)
            # CI removed per request
        
        ax.set_title(f'{LABELS[marker]}', fontsize=SCATTER_TITLE_SIZE, fontweight='bold')
        ax.set_xlabel(clin_label, fontsize=SCATTER_XLABEL_SIZE)
        ax.set_ylabel(y_label, fontsize=SCATTER_YLABEL_SIZE)
        ax.tick_params(labelsize=SCATTER_TICK_SIZE)
        
        p_str = f'p={p:.3f}' if p >= 0.001 else f'p={p:.1e}'
        ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
                ha='right', va='top', fontsize=SCATTER_PVAL_SIZE, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.text(-0.15, 1.05, scatter_labels[idx], transform=ax.transAxes, 
                fontsize=PANEL_LABEL_SIZE, fontweight=PANEL_LABEL_FONTWEIGHT, va='bottom')
    
    # ============================================================
    # Save Figure
    # ============================================================
    plt.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Done! Saved to {OUTPUT_PNG} and {OUTPUT_PDF}")