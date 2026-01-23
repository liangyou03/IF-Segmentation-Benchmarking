#!/usr/bin/env python3
"""
plot_comprehensive_boxplot.py

Create comprehensive per-cell-type box plots for metrics:
- Rows: Pooled, OLIG2, NEUN, IBA1, GFAP, Nuclei
- Cols: Precision, Recall, Boundary F-score

Publication ready - no titles, no Chinese text.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Import shared configuration
from config import (
    RESULTS_DIR, PLOTS_DIR, PNG_SUBDIR_NAME, FIGURE_DPI, TRANSPARENT_BG,
    CELL_TYPE_GROUPS, ALGORITHM_COLORS, FONT_SIZES,
    get_algorithm_display_name, save_figure_with_no_legend
)

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = PLOTS_DIR / PNG_SUBDIR_NAME
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================
try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "legend.fontsize": FONT_SIZES["legend"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "figure.dpi": 200,
    })


def compute_precision_recall(df):
    """Compute precision and recall at IoU@0.50 from TP, FP, FN."""
    df = df.copy()
    df['Precision@0.50'] = df['TP@0.50'] / (df['TP@0.50'] + df['FP@0.50'])
    df['Precision@0.50'] = df['Precision@0.50'].fillna(0)
    df['Recall@0.50'] = df['TP@0.50'] / (df['TP@0.50'] + df['FN@0.50'])
    df['Recall@0.50'] = df['Recall@0.50'].fillna(0)
    return df


def plot_comprehensive_boxplot():
    """
    Create comprehensive box plots grid:
    - 6 rows: Pooled (cell), OLIG2, NEUN, IBA1, GFAP, Nuclei
    - 3 cols: Precision, Recall, Boundary F-score
    """
    print("Creating comprehensive per-cell-type box plots with nuclei...")

    # Load cell data
    cell_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    cell_img = compute_precision_recall(cell_img)

    # Load nuclei data (adjust path if needed)
    nuclei_path = RESULTS_DIR / "nuclei_per_image.parquet"
    if nuclei_path.exists():
        nuclei_img = pd.read_parquet(nuclei_path)
        nuclei_img = compute_precision_recall(nuclei_img)
        has_nuclei = True
    else:
        print(f"⚠ Nuclei data not found at {nuclei_path}")
        has_nuclei = False
        nuclei_img = None

    # Get all algorithms
    all_algorithms = sorted(cell_img['algorithm'].unique())

    # Metrics to plot
    metrics = [
        ('Precision@0.50', 'Precision @ IoU=0.50'),
        ('Recall@0.50', 'Recall @ IoU=0.50'),
        ('BF_bestF', 'Boundary F1-score'),
    ]

    # Row definitions: (label, data_source, filter_pattern)
    # data_source: 'cell' or 'nuclei'
    rows = []
    if has_nuclei:
        rows.append(('Nuclei', 'nuclei', None))
    rows.extend([
        ('All Cells', 'cell', None),
        ('OLIG2', 'cell', 'olig2'),
        ('NeuN', 'cell', 'neun'),
        ('IBA1', 'cell', 'iba1'),
        ('GFAP', 'cell', 'gfap'),
    ])

    # Subplot labels for each row
    row_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][:len(rows)]

    n_rows = len(rows)
    n_cols = len(metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)

    print("\n" + "="*80)
    print("MEDIAN VALUES FOR ALL SUBPLOTS")
    print("="*80)

    for row_idx, (row_label, data_source, pattern) in enumerate(rows):
        # Select data source
        if data_source == 'nuclei':
            base_data = nuclei_img
        else:
            base_data = cell_img

        # Apply filter if needed
        if pattern is not None:
            row_data = base_data[base_data["base"].str.contains(pattern, case=False, na=False)]
        else:
            row_data = base_data

        for col_idx, (metric_col, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # Print header for this subplot
            print(f"\n[{row_labels[row_idx]}] {row_label} - {metric_name}")
            print("-" * 60)

            # Get available algorithms in this subset
            available_algos = row_data['algorithm'].unique()
            algorithms = [a for a in all_algorithms if a in available_algos]

            if len(algorithms) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                print("  No data available")
                continue

            # Sort by median
            medians = row_data.groupby('algorithm')[metric_col].median()
            algorithms = medians.loc[algorithms].sort_values(ascending=False).index.tolist()

            # Print medians for all algorithms (sorted)
            for algo in algorithms:
                median_val = medians[algo]
                display_name = get_algorithm_display_name(algo)
                print(f"  {display_name:30s}: {median_val:.4f}")

            # Prepare box data
            box_data = [row_data[row_data['algorithm'] == algo][metric_col].values
                        for algo in algorithms]

            # Create boxplot
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)

            # Color boxes
            for patch, algo in zip(bp['boxes'], algorithms):
                color = ALGORITHM_COLORS.get(algo, '#808080')
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)

            for element in ['whiskers', 'fliers', 'caps']:
                plt.setp(bp[element], color='black', linewidth=1)
            plt.setp(bp['medians'], color='black', linewidth=2)

            # X-axis labels
            ticks = range(1, len(algorithms) + 1)
            display_labels = [get_algorithm_display_name(algo) for algo in algorithms]
            ax.set_xticks(ticks)
            ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)

            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)

            # Top row: metric names
            if row_idx == 0:
                ax.set_title(metric_name, fontsize=12, fontweight='bold')

            # First column: row labels + subplot label (A, B, C, ...)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight='bold')
                # Add subplot label outside top-left
                ax.text(
                    -0.18, 1, row_labels[row_idx],
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=15, fontweight="bold"
                )

    print("\n" + "="*80)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.08, bottom=0.1)

    # Save
    out_pdf = PLOTS_DIR / "metrics_boxplot_comprehensive.pdf"
    out_png = PNG_DIR / "metrics_boxplot_comprehensive.png"
    save_figure_with_no_legend(fig, out_pdf, out_png, dpi=FIGURE_DPI, transparent=TRANSPARENT_BG)
    print(f"\n✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    plot_comprehensive_boxplot()
