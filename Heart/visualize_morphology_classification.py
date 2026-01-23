"""
Heart Dataset – Morphology Classification Visualization
Integrates segmentation evaluation and morphology-based cell classification results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from _paths import HEART_DATA_ROOT, NUC_MOFO_ROOT

# ============================================================================
# Paths
# ============================================================================

# Segmentation evaluation data
EVAL_CSV = HEART_DATA_ROOT / "evaluation_results.csv"

# Morphology classification results
MORPH_RESULTS_DIR = NUC_MOFO_ROOT / "results"

# Output directory
PLOTS_DIR = HEART_DATA_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Style Configuration
# ============================================================================

ALGORITHM_COLORS = {
    "cellpose": "#1f77b4",
    "cellpose_sam": "#1f77b4",
    "stardist": "#ff7f0e",
    "omnipose": "#8c564b",
    "watershed": "#9467bd",
    "mesmer": "#d62728",
    "lacss": "#e377c2",
    "splinedist": "#7f7f7f",
    "microsam": "#bcbd22",
    "cellsam": "#2ca02c",
    "InstanSeg": "#17becf",
}

ALGORITHM_DISPLAY_NAMES = {
    "cellpose": "Cellpose",
    "cellpose_sam": "CellposeSAM",
    "stardist": "StarDist",
    "omnipose": "Omnipose",
    "watershed": "Watershed",
    "mesmer": "Mesmer",
    "lacss": "LACSS",
    "splinedist": "SplineDist",
    "microsam": "MicroSAM",
    "cellsam": "CellSAM",
    "instanseg": "InstanSeg",
}

# Cell type colors for morphology classification
CELLTYPE_COLORS = {
    "epi": "#e74c3c",          # red
    "immune cell": "#3498db",  # blue
    "immune": "#3498db",
    "mural cell": "#2ecc71",   # green
    "mural": "#2ecc71",
    "cm": "#9b59b6",           # purple (cardiomyocytes)
    "ec": "#f39c12",           # orange (endothelial cells)
    "fb": "#1abc9c",           # teal (fibroblasts)
}

CELLTYPE_DISPLAY_NAMES = {
    "epi": "Epithelial",
    "immune cell": "Immune",
    "immune": "Immune",
    "mural cell": "Mural",
    "mural": "Mural",
    "cm": "Cardiomyocyte",
    "ec": "Endothelial",
    "fb": "Fibroblast",
}

plt.style.use("default")
sns.set_style("whitegrid")

FONT_SIZES = {
    "title": 18,
    "subtitle": 14,
    "label": 16,
    "legend": 12,
    "tick": 14,
    "annotation": 11,
}

# ============================================================================
# Helper functions
# ============================================================================

def save_figure(fig, name, bbox_inches='tight'):
    pdf_path = PLOTS_DIR / f"{name}.pdf"
    png_path = PLOTS_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches=bbox_inches)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches=bbox_inches)
    print(f"  Saved: {pdf_path.name}")


def get_display_name(algo):
    return ALGORITHM_DISPLAY_NAMES.get(algo, algo)


def get_celltype_display(name):
    return CELLTYPE_DISPLAY_NAMES.get(name.lower(), name)


def get_celltype_color(name):
    return CELLTYPE_COLORS.get(name.lower(), "#7f7f7f")


# ============================================================================
# Load Data
# ============================================================================

def load_segmentation_data():
    """Load segmentation evaluation results."""
    print(f"Loading segmentation data: {EVAL_CSV}")
    df = pd.read_csv(EVAL_CSV)
    df["algorithm_display"] = df["algorithm"].apply(get_display_name)
    return df


def load_morphology_data():
    """Load morphology classification results from all model directories."""
    results = {}

    # Define result directories to load
    result_dirs = [
        ("section01_rf", "Section 01 RF", Path(MORPH_RESULTS_DIR) / "section01_rf"),
        ("section02_rf", "Section 02 RF", Path(MORPH_RESULTS_DIR) / "section02_rf"),
        ("section01_logit", "Section 01 Logit", Path(MORPH_RESULTS_DIR) / "section01_logit"),
        ("section02_logit", "Section 02 Logit", Path(MORPH_RESULTS_DIR) / "section02_logit"),
    ]

    for key, display_name, path in result_dirs:
        summary_path = path / "metrics_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["model"] = display_name
            df["model_key"] = key
            results[key] = df
            print(f"  Loaded {display_name}: {len(df)} rows")

    return results


def load_feature_importance():
    """Load feature importance from Random Forest models."""
    importance_data = []

    for section in ["section01_rf", "section02_rf"]:
        path = Path(MORPH_RESULTS_DIR) / section / "feature_importance_long.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["section"] = section.replace("section", "Section ")
            importance_data.append(df)
            print(f"  Loaded feature importance: {section}")

    if importance_data:
        return pd.concat(importance_data, ignore_index=True)
    return None


# ============================================================================
# Plot Functions
# ============================================================================

def plot_segmentation_boxplot(df):
    """Plot segmentation evaluation boxplots (original style)."""
    print("\n📊 Plotting: Segmentation Performance Boxplots")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    metrics = [
        ("object_recall", "Object Recall (%)"),
        ("pixel_recall", "Pixel Recall (%)"),
    ]

    df_plot = df.copy()

    for ax, (metric, ylabel) in zip(axes, metrics):
        df_plot_metric = df_plot.copy()
        df_plot_metric[metric] = df_plot_metric[metric] * 100

        # Sort by median value
        order = (
            df_plot_metric.groupby("algorithm_display")[metric]
            .median()
            .sort_values(ascending=False)
            .index
        )

        data = [
            df_plot_metric[df_plot_metric["algorithm_display"] == algo][metric].values
            for algo in order
        ]

        bp = ax.boxplot(
            data,
            tick_labels=order,
            patch_artist=True,
            showmeans=False,
        )

        for patch, algo_disp in zip(bp["boxes"], order):
            orig_algo = df_plot[df_plot["algorithm_display"] == algo_disp]["algorithm"].iloc[0]
            patch.set_facecolor(ALGORITHM_COLORS.get(orig_algo, "#333333"))
            patch.set_alpha(0.75)

        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
        ax.set_title(ylabel.replace(" (%)", ""), fontsize=FONT_SIZES["title"])
        ax.tick_params(axis="x", rotation=45, labelsize=FONT_SIZES["tick"])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, "01_segmentation_boxplot")
    plt.close()


def plot_morphology_auroc(morph_results):
    """Plot AUROC for morphology-based cell classification."""
    print("\n📊 Plotting: Morphology Classification AUROC")

    # Only plot first 2 models (RF models)
    rf_models = {k: v for k, v in morph_results.items() if "rf" in k.lower()}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (key, df) in enumerate(rf_models.items()):
        ax = axes[idx]

        # Filter AUROC only
        auroc_df = df[df["metric"] == "AUROC"].copy()
        auroc_df["cell_type_display"] = auroc_df["cell_type"].apply(get_celltype_display)

        # Sort by mean AUROC
        auroc_df = auroc_df.sort_values("mean", ascending=False)

        x_pos = np.arange(len(auroc_df))
        colors = [get_celltype_color(ct) for ct in auroc_df["cell_type"]]

        # Bar plot with error bars (95% CI)
        ax.bar(x_pos, auroc_df["mean"] * 100,
               yerr=(auroc_df["mean"] - auroc_df["ci95_low"]) * 100,
               capsize=5, alpha=0.8, color=colors,
               error_kw={'linewidth': 1.5})

        ax.set_xlabel("Cell Type", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("AUROC (%)", fontsize=FONT_SIZES["label"])
        ax.set_title(df["model"].iloc[0], fontsize=FONT_SIZES["title"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(auroc_df["cell_type_display"], rotation=0, ha="center")
        ax.set_ylim([50, 100])
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (idx_row, row) in enumerate(auroc_df.iterrows()):
            ax.text(i, row["mean"] * 100 + 2,
                   f"{row['mean']*100:.1f}%",
                   ha='center', va='bottom', fontsize=FONT_SIZES["annotation"])

    plt.tight_layout()
    save_figure(fig, "02_morphology_auroc")
    plt.close()


def plot_morphology_auprc(morph_results):
    """Plot AUPRC for morphology-based cell classification."""
    print("\n📊 Plotting: Morphology Classification AUPRC")

    # Only plot first 2 models (RF models)
    rf_models = {k: v for k, v in morph_results.items() if "rf" in k.lower()}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (key, df) in enumerate(rf_models.items()):
        ax = axes[idx]

        auprc_df = df[df["metric"] == "AUPRC"].copy()
        auprc_df["cell_type_display"] = auprc_df["cell_type"].apply(get_celltype_display)
        auprc_df = auprc_df.sort_values("mean", ascending=False)

        x_pos = np.arange(len(auprc_df))
        colors = [get_celltype_color(ct) for ct in auprc_df["cell_type"]]

        ax.bar(x_pos, auprc_df["mean"] * 100,
               yerr=(auprc_df["mean"] - auprc_df["ci95_low"]) * 100,
               capsize=5, alpha=0.8, color=colors,
               error_kw={'linewidth': 1.5})

        ax.set_xlabel("Cell Type", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("AUPRC (%)", fontsize=FONT_SIZES["label"])
        ax.set_title(df["model"].iloc[0], fontsize=FONT_SIZES["title"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(auprc_df["cell_type_display"], rotation=0, ha="center")
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis="y")

        for i, (idx_row, row) in enumerate(auprc_df.iterrows()):
            ax.text(i, row["mean"] * 100 + 2,
                   f"{row['mean']*100:.1f}%",
                   ha='center', va='bottom', fontsize=FONT_SIZES["annotation"])

    plt.tight_layout()
    save_figure(fig, "03_morphology_auprc")
    plt.close()


def plot_combined_metrics(morph_results):
    """Plot combined AUROC and AUPRC comparison across models."""
    print("\n📊 Plotting: Combined Metrics Comparison")

    # Only use RF models for cleaner comparison
    rf_models = {k: v for k, v in morph_results.items() if "rf" in k.lower()}

    # Combine all results
    all_data = []
    for key, df in rf_models.items():
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # Get unique cell types and metrics
    cell_types = sorted(combined["cell_type"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, metric in enumerate(["AUROC", "AUPRC"]):
        ax = axes[ax_idx]

        metric_data = combined[combined["metric"] == metric].copy()
        metric_data["cell_type_display"] = metric_data["cell_type"].apply(get_celltype_display)

        # Group by cell type and model
        plot_data = []
        for ct in cell_types:
            ct_data = metric_data[metric_data["cell_type"] == ct]
            for _, row in ct_data.iterrows():
                plot_data.append({
                    "cell_type": get_celltype_display(ct),
                    "model": row["model"],
                    "mean": row["mean"] * 100,
                    "ci_low": row["ci95_low"] * 100,
                    "ci_high": row["ci95_high"] * 100,
                    "color": get_celltype_color(ct)
                })

        plot_df = pd.DataFrame(plot_data)

        # Create grouped bar chart
        models = sorted(plot_df["model"].unique())
        n_models = len(models)
        n_ct = len(cell_types)
        x = np.arange(n_ct)
        width = 0.35

        for i, model in enumerate(models):
            model_data = plot_df[plot_df["model"] == model]

            # Get ordered means and colors
            ordered_ct = [get_celltype_display(ct) for ct in cell_types]
            means = []
            colors = []
            for ct_disp in ordered_ct:
                ct_rows = model_data[model_data["cell_type"] == ct_disp]
                if len(ct_rows) > 0:
                    means.append(ct_rows["mean"].values[0])
                    colors.append(ct_rows["color"].values[0])
                else:
                    means.append(0)
                    colors.append("#7f7f7f")

            means = np.array(means)
            colors = np.array(colors)

            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(x + offset, means, width, label=model, alpha=0.8)

            # Color bars by cell type
            for bar, color in zip(bars, colors):
                if pd.notna(color):
                    bar.set_color(color)
                    bar.set_alpha(0.8)

        ax.set_xlabel("Cell Type", fontsize=FONT_SIZES["label"])
        ax.set_ylabel(f"{metric} (%)", fontsize=FONT_SIZES["label"])
        ax.set_title(f"{metric} by Model", fontsize=FONT_SIZES["title"])
        ax.set_xticks(x)
        ax.set_xticklabels([get_celltype_display(ct) for ct in cell_types])
        ax.legend(fontsize=FONT_SIZES["legend"], loc="lower right")
        ax.grid(True, alpha=0.3, axis="y")

        if metric == "AUROC":
            ax.set_ylim([60, 100])
        else:
            ax.set_ylim([0, 100])

    plt.tight_layout()
    save_figure(fig, "04_combined_metrics_comparison")
    plt.close()


def plot_feature_importance(importance_df, n_top=15):
    """Plot feature importance from Random Forest models."""
    print("\n📊 Plotting: Feature Importance")

    # Aggregate importance by feature and cell type
    agg_df = importance_df.groupby(["cell_type", "feature"])["importance"].mean().reset_index()
    agg_df = agg_df.sort_values(["cell_type", "importance"], ascending=[True, False])

    # Get top features overall
    top_features = agg_df.groupby("feature")["importance"].mean().nlargest(n_top).index
    plot_df = agg_df[agg_df["feature"].isin(top_features)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (section, section_df) in enumerate(plot_df.groupby("cell_type")):
        if idx >= 2:
            break
        ax = axes[idx]

        section_data = section_df.sort_values("importance", ascending=True)
        colors = [get_celltype_color(section)] * len(section_data)

        ax.barh(section_data["feature"], section_data["importance"], color=colors, alpha=0.8)
        ax.set_xlabel("Mean Importance", fontsize=FONT_SIZES["label"])
        ax.set_title(f"{get_celltype_display(section)} - Top {n_top} Features", fontsize=FONT_SIZES["title"])
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure(fig, "05_feature_importance")
    plt.close()


def plot_comprehensive_overview(df_seg, morph_results, importance_df=None):
    """Create a comprehensive multi-panel figure combining all results."""
    print("\n📊 Plotting: Comprehensive Overview")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # === Panel 1-2: Segmentation Boxplots (top row) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax, (metric, ylabel) in zip([ax1, ax2],
                                     [("object_recall", "Object Recall (%)"),
                                      ("pixel_recall", "Pixel Recall (%)")]):
        df_plot = df_seg.copy()
        df_plot[metric] = df_plot[metric] * 100
        order = df_plot.groupby("algorithm_display")[metric].median().sort_values(ascending=False).index
        data = [df_plot[df_plot["algorithm_display"] == algo][metric].values for algo in order]

        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=False)
        for patch, algo_disp in zip(bp["boxes"], order):
            orig_algo = df_seg[df_seg["algorithm_display"] == algo_disp]["algorithm"].iloc[0]
            patch.set_facecolor(ALGORITHM_COLORS.get(orig_algo, "#333"))
            patch.set_alpha(0.75)

        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
        ax.set_title(metric.replace("_", " ").title(), fontsize=FONT_SIZES["title"])
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # === Panel 3: Morphology AUROC (top right) ===
    ax3 = fig.add_subplot(gs[0, 2])
    all_auroc = []
    for key, df in morph_results.items():
        auroc_df = df[df["metric"] == "AUROC"].copy()
        auroc_df["model"] = df["model"].iloc[0]
        all_auroc.append(auroc_df)

    combined_auroc = pd.concat(all_auroc)
    combined_auroc["cell_type_display"] = combined_auroc["cell_type"].apply(get_celltype_display)

    for model in combined_auroc["model"].unique():
        model_data = combined_auroc[combined_auroc["model"] == model].sort_values("mean")
        x = np.arange(len(model_data))
        ax3.plot(x, model_data["mean"] * 100, 'o-', label=model, linewidth=2, markersize=8)
        ax3.fill_between(x,
                        model_data["ci95_low"] * 100,
                        model_data["ci95_high"] * 100,
                        alpha=0.2)

    ax3.set_ylabel("AUROC (%)", fontsize=FONT_SIZES["label"])
    ax3.set_title("Cell Classification AUROC", fontsize=FONT_SIZES["title"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_data["cell_type_display"], rotation=45, ha="right")
    ax3.legend(fontsize=FONT_SIZES["legend"])
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([60, 100])

    # === Panel 4-5: Morphology AUROC per section (middle row) ===
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])

    for idx, (key, df) in enumerate(morph_results.items()):
        if idx >= 2:
            break
        ax = [ax4, ax5][idx]

        auroc_df = df[df["metric"] == "AUROC"].copy()
        auroc_df["cell_type_display"] = auroc_df["cell_type"].apply(get_celltype_display)
        auroc_df = auroc_df.sort_values("mean")

        x = np.arange(len(auroc_df))
        colors = [get_celltype_color(ct) for ct in auroc_df["cell_type"]]

        bars = ax.barh(x, auroc_df["mean"] * 100,
                      xerr=[(auroc_df["mean"] - auroc_df["ci95_low"]) * 100,
                            (auroc_df["ci95_high"] - auroc_df["mean"]) * 100],
                      capsize=4, color=colors, alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(auroc_df["cell_type_display"])
        ax.set_xlabel("AUROC (%)", fontsize=FONT_SIZES["label"])
        ax.set_title(df["model"].iloc[0], fontsize=FONT_SIZES["title"])
        ax.set_xlim([60, 100])
        ax.grid(True, alpha=0.3, axis='x')

        for i, (idx_row, row) in enumerate(auroc_df.iterrows()):
            ax.text(row["mean"] * 100 + 1, i, f"{row['mean']*100:.1f}%",
                   va='center', fontsize=9)

    # === Panel 6: Feature Importance (middle right) ===
    ax6 = fig.add_subplot(gs[1, 2])
    if importance_df is not None:
        agg_df = importance_df.groupby(["cell_type", "feature"])["importance"].mean().reset_index()
        top_features = agg_df.groupby("feature")["importance"].mean().nlargest(8).index
        plot_df = agg_df[agg_df["feature"].isin(top_features)]

        for ct in plot_df["cell_type"].unique():
            ct_data = plot_df[plot_df["cell_type"] == ct].set_index("feature")
            ct_data = ct_data.reindex(top_features).fillna(0)
            ax6.plot(np.arange(len(top_features)), ct_data["importance"],
                    'o-', label=get_celltype_display(ct), linewidth=2, markersize=6)

        ax6.set_xticks(np.arange(len(top_features)))
        ax6.set_xticklabels([f.replace('_', ' ')[:15] for f in top_features], rotation=45, ha='right', fontsize=9)
        ax6.set_ylabel("Mean Importance", fontsize=FONT_SIZES["label"])
        ax6.set_title("Top Morphological Features", fontsize=FONT_SIZES["title"])
        ax6.legend(fontsize=FONT_SIZES["legend"])
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "Feature importance data not available",
                ha='center', va='center', transform=ax6.transAxes)

    # === Panel 7-9: AUPRC and summary (bottom row) ===
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    # AUPRC comparison
    for idx, (key, df) in enumerate(morph_results.items()):
        if idx >= 2:
            break
        ax = [ax7, ax8][idx]

        auprc_df = df[df["metric"] == "AUPRC"].copy()
        auprc_df["cell_type_display"] = auprc_df["cell_type"].apply(get_celltype_display)
        auprc_df = auprc_df.sort_values("mean")

        x = np.arange(len(auprc_df))
        colors = [get_celltype_color(ct) for ct in auprc_df["cell_type"]]

        bars = ax.barh(x, auprc_df["mean"] * 100,
                      xerr=[(auprc_df["mean"] - auprc_df["ci95_low"]) * 100,
                            (auprc_df["ci95_high"] - auprc_df["mean"]) * 100],
                      capsize=4, color=colors, alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(auprc_df["cell_type_display"])
        ax.set_xlabel("AUPRC (%)", fontsize=FONT_SIZES["label"])
        ax.set_title(f"{df['model'].iloc[0]} - AUPRC", fontsize=FONT_SIZES["title"])
        ax.set_xlim([0, 100])
        ax.grid(True, alpha=0.3, axis='x')

        for i, (idx_row, row) in enumerate(auprc_df.iterrows()):
            ax.text(row["mean"] * 100 + 2, i, f"{row['mean']*100:.1f}%",
                   va='center', fontsize=9)

    # Summary statistics panel
    ax9.axis('off')

    summary_text = "Morphology Classification Summary\n\n"
    for key, df in morph_results.items():
        model_name = df["model"].iloc[0]
        auroc_mean = df[df["metric"] == "AUROC"]["mean"].mean() * 100
        auprc_mean = df[df["metric"] == "AUPRC"]["mean"].mean() * 100
        n_ct = df["cell_type"].nunique()

        summary_text += f"{model_name}:\n"
        summary_text += f"  Cell Types: {n_ct}\n"
        summary_text += f"  Mean AUROC: {auroc_mean:.1f}%\n"
        summary_text += f"  Mean AUPRC: {auprc_mean:.1f}%\n\n"

    # Add segmentation summary
    seg_auroc = df_seg["object_recall"].mean() * 100
    summary_text += f"Segmentation (CellposeSAM):\n"
    summary_text += f"  Mean Object Recall: {seg_auroc:.1f}%\n"

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=FONT_SIZES["label"], verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle("Heart Tissue Analysis: Segmentation & Morphology-Based Cell Classification",
                fontsize=20, fontweight='bold')

    save_figure(fig, "00_comprehensive_overview")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*60)
    print("Heart Dataset – Morphology Classification Visualization")
    print("="*60)

    # Load data
    df_seg = load_segmentation_data()
    print(f"  Segmentation: {len(df_seg)} rows, {df_seg['algorithm'].nunique()} algorithms")

    morph_results = load_morphology_data()
    importance_df = load_feature_importance()

    # Generate plots
    plot_segmentation_boxplot(df_seg)
    plot_morphology_auroc(morph_results)
    plot_morphology_auprc(morph_results)
    plot_combined_metrics(morph_results)
    if importance_df is not None:
        plot_feature_importance(importance_df)
    plot_comprehensive_overview(df_seg, morph_results, importance_df)

    print("\n" + "="*60)
    print(f"All plots saved to: {PLOTS_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
