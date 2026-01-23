#!/usr/bin/env python3
"""
plot_combined_with_legend.py

2x4 layout:
  Row 1: Nuclei | Cell (marker only) | Cell (DAPI+marker) | [Unified Legend]
  Row 2: OLIG2  | NeuN               | IBA1               | GFAP

All panels share one clean legend in the top-right corner.
Bottom 4 panels share label "D" in the leftmost position.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

# Apply science style
try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass

# Sans-serif font
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

from config import (
    RESULTS_DIR, PLOTS_DIR, PNG_SUBDIR_NAME,
    FIGURE_DPI, TRANSPARENT_BG,
    CELL_TYPE_GROUPS,
    ALGORITHM_COLORS, ALGORITHM_LINESTYLES, ALGORITHM_MARKERS,
    get_algorithm_display_name,
    save_figure_with_no_legend
)

# ============================================================================
# HELPERS
# ============================================================================

def infer_ap_cols(df):
    ap = [c for c in df.columns if re.match(r"^AP@\d\.\d{2}$", c)]
    thr = np.array([float(c.split("@")[1]) for c in ap])
    order = np.argsort(thr)
    return thr[order], [ap[i] for i in order]


def plot_ap_curves(ax, df, panel_label, show_ylabel=False, show_xlabel=True, subplot_label=None):
    """Plot AP curves without legend."""
    thr, ap_cols = infer_ap_cols(df)
    curve = df.groupby("algorithm")[ap_cols].mean()
    AP = curve.mean(axis=1)
    curve = curve.loc[AP.sort_values(ascending=False).index]

    for algo, row in curve.iterrows():
        ax.plot(
            thr, row.values,
            color=ALGORITHM_COLORS.get(algo, "#000000"),
            linestyle=ALGORITHM_LINESTYLES.get(algo, "-"),
            marker=ALGORITHM_MARKERS.get(algo, "o"),
            linewidth=1.8,
            markersize=4,
        )

    ax.set_xlim(thr.min(), thr.max())
    ax.set_ylim(0, 1.0)
    if show_xlabel:
        ax.set_xlabel("IoU")
    if show_ylabel:
        ax.set_ylabel("Precision")
    ax.minorticks_on()
    ax.grid(alpha=0.25, linewidth=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Subplot label A, B, etc. - top left outside
    if subplot_label:
        ax.text(
            -0.12, 1.05, subplot_label,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=12, fontweight="bold"
        )

    # Panel title - inside top left
    ax.text(
        0.03, 0.97, panel_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold"
    )

    return AP


def create_unified_legend(ax, algo_order, AP_dict):
    """
    Create a clean unified legend showing all algorithms in 2 columns.
    """
    ax.axis("off")
    
    handles = []
    labels = []
    
    for algo in algo_order:
        handle = mlines.Line2D(
            [], [],
            color=ALGORITHM_COLORS.get(algo, "#000000"),
            linestyle=ALGORITHM_LINESTYLES.get(algo, "-"),
            marker=ALGORITHM_MARKERS.get(algo, "o"),
            linewidth=2.5,
            markersize=8,
        )
        handles.append(handle)
        
        display_name = get_algorithm_display_name(algo)
        labels.append(display_name)
    
    ax.legend(
        handles, labels,
        loc="center",
        frameon=False,
        fontsize=11,
        title="Methods",
        title_fontsize=12,
        handlelength=2.5,
        labelspacing=0.8,
        ncol=2,
        columnspacing=1.5,
    )


def add_AP_annotations(ax, AP_series, n_top=None, y_pos=1.0):
    """Add AP annotations inside the panel."""
    sorted_algos = AP_series.sort_values(ascending=False)
    if n_top:
        sorted_algos = sorted_algos.head(n_top)
    
    text_lines = []
    for algo, val in sorted_algos.items():
        short_name = get_algorithm_display_name(algo)
        # Don't truncate - show full name
        text_lines.append(f"{short_name}: {val:.2f}")
    
    annotation = "\n".join(text_lines)
    ax.text(
        0.97, y_pos, annotation,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        family="sans-serif",
        linespacing=1.2,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading data...")
    
    df_nuclei = pd.read_parquet(RESULTS_DIR / "nuclei_per_image.parquet")
    df_cell_2ch = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    df_cell_marker = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")

    # Create 2x4 figure
    fig, axes = plt.subplots(
        2, 4,
        figsize=(14, 7),
        gridspec_kw={"hspace": 0.3, "wspace": 0.25}
    )

    # -------------------------------------------------------------------------
    # Row 1: Overall performance (no x-axis labels)
    # SWITCHED ORDER: B and C are swapped
    # -------------------------------------------------------------------------
    AP_nuclei = plot_ap_curves(axes[0, 0], df_nuclei, "Nuclei", 
                                show_ylabel=True, show_xlabel=False, subplot_label="A")
    AP_marker = plot_ap_curves(axes[0, 1], df_cell_marker, "Cell (marker only)", 
                                show_xlabel=False, subplot_label="B")
    AP_2ch = plot_ap_curves(axes[0, 2], df_cell_2ch, "Cell (DAPI + marker)", 
                             show_xlabel=False, subplot_label="C")

    # Add AP annotations to each panel
    add_AP_annotations(axes[0, 0], AP_nuclei, y_pos=1.0)
    add_AP_annotations(axes[0, 1], AP_marker, y_pos=0.8)
    add_AP_annotations(axes[0, 2], AP_2ch, y_pos=0.8)

    # -------------------------------------------------------------------------
    # Top-right: Unified legend (axes[0, 3])
    # -------------------------------------------------------------------------
    # Get algorithm order from nuclei (best to worst)
    algo_order = AP_nuclei.sort_values(ascending=False).index.tolist()
    create_unified_legend(axes[0, 3], algo_order, AP_nuclei.to_dict())

    # -------------------------------------------------------------------------
    # Row 2: Per cell type (with x-axis labels)
    # Share label "D" for all 4 panels in bottom row
    # -------------------------------------------------------------------------
    cell_type_panels = list(CELL_TYPE_GROUPS.items())
    
    for i, (gname, pattern) in enumerate(cell_type_panels):
        ax = axes[1, i]
        sub = df_cell_2ch[
            df_cell_2ch["base"].str.contains(pattern, case=False, regex=True, na=False)
        ]
        
        if sub.empty:
            ax.text(0.5, 0.5, f"{gname}\nNo data",
                    transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            continue
        
        # Only add "D" label to the first panel (leftmost)
        subplot_label = "D" if i == 0 else None
        
        AP_ct = plot_ap_curves(ax, sub, gname, 
                               show_ylabel=(i == 0), show_xlabel=True, 
                               subplot_label=subplot_label)
        add_AP_annotations(ax, AP_ct)

    plt.tight_layout()

    # Save
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR = PLOTS_DIR / PNG_SUBDIR_NAME
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    out_pdf = PLOTS_DIR / "combined_ap_with_legend.pdf"
    out_png = PNG_DIR / "combined_ap_with_legend.png"

    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight", transparent=TRANSPARENT_BG)
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight", transparent=TRANSPARENT_BG)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()