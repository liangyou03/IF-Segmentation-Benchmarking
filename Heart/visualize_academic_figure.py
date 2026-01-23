"""
Heart Dataset – Academic Quality Figure for Genome Biology
Figure: Morphology-based Cell Classification Performance Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from _paths import HEART_DATA_ROOT, NUC_MOFO_ROOT

# ============================================================================
# Configuration
# ============================================================================

# Paths
EVAL_CSV = HEART_DATA_ROOT / "evaluation_results.csv"
MORPH_RESULTS_DIR = NUC_MOFO_ROOT / "results"
PLOTS_DIR = HEART_DATA_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Academic style configuration
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme (colorblind-friendly, publication quality)
CELLTYPE_COLORS = {
    "epi": "#E64B35",           # red
    "immune cell": "#4DBBD5",   # blue
    "immune": "#4DBBD5",
    "mural cell": "#00A087",    # green
    "mural": "#00A087",
    "cm": "#F39B7F",            # orange (cardiomyocytes)
    "ec": "#3C5488",            # dark blue (endothelial cells)
    "fb": "#8491B4",            # purple-gray (fibroblasts)
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

MODEL_COLORS = {
    "Section 01 RF": "#E64B35",
    "Section 02 RF": "#4DBBD5",
    "Section 01 Logit": "#00A087",
    "Section 02 Logit": "#F39B7F",
}

FEATURE_DISPLAY_NAMES = {
    "area": "Area",
    "area_um2": "Area (μm²)",
    "perimeter": "Perimeter",
    "aspect_ratio": "Aspect ratio",
    "circularity": "Circularity",
    "roundness": "Roundness",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "extent": "Extent",
    "major_axis_length": "Major axis",
    "minor_axis_length": "Minor axis",
    "feret_diameter_max": "Feret diameter",
    "equivalent_diameter_area": "Equiv. diameter",
    "convex_area": "Convex area",
    "filled_area": "Filled area",
    "centroid-0": "Centroid X",
    "centroid-1": "Centroid Y",
    "orientation": "Orientation",
    "bbox-0": "BBox Y0",
    "bbox-1": "BBox X0",
    "bbox-2": "BBox Y1",
    "bbox-3": "BBox X1",
    "moments_hu-0": "Hu moment 0",
    "moments_hu-1": "Hu moment 1",
    "moments_hu-2": "Hu moment 2",
    "moments_hu-3": "Hu moment 3",
    "moments_hu-4": "Hu moment 4",
    "moments_hu-5": "Hu moment 5",
    "moments_hu-6": "Hu moment 6",
    "inertia_tensor_eigvals-0": "Inertia λ0",
    "inertia_tensor_eigvals-1": "Inertia λ1",
    "euler_number": "Euler number",
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_celltype_display(name):
    return CELLTYPE_DISPLAY_NAMES.get(name.lower(), name)


def get_celltype_color(name):
    return CELLTYPE_COLORS.get(name.lower(), "#7f7f7f")


def get_feature_display(name):
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# ============================================================================
# Load Data
# ============================================================================

def load_predictions():
    """Load prediction data for ROC curves."""
    predictions = {}

    # Section 01
    path1 = MORPH_RESULTS_DIR / "section01_rf" / "predictions_last_repeat.csv"
    if path1.exists():
        df1 = pd.read_csv(path1)
        df1["section"] = "Section 01"
        predictions["section01"] = df1
        print(f"Loaded Section 01 predictions: {len(df1)} rows")

    # Section 02
    path2 = MORPH_RESULTS_DIR / "section02_rf" / "predictions_last_repeat.csv"
    if path2.exists():
        df2 = pd.read_csv(path2)
        df2["section"] = "Section 02"
        predictions["section02"] = df2
        print(f"Loaded Section 02 predictions: {len(df2)} rows")

    return predictions


def load_feature_importance():
    """Load and aggregate feature importance."""
    importance_data = []

    for section_key, section_name, section_path in [
        ("section01", "Section 01", MORPH_RESULTS_DIR / "section01_rf" / "feature_importance_long.csv"),
        ("section02", "Section 02", MORPH_RESULTS_DIR / "section02_rf" / "feature_importance_long.csv"),
    ]:
        if section_path.exists():
            df = pd.read_csv(section_path)
            df["section"] = section_name
            df["section_key"] = section_key
            importance_data.append(df)
            print(f"Loaded {section_name} feature importance: {len(df)} rows")

    if importance_data:
        return pd.concat(importance_data, ignore_index=True)
    return None


def load_metrics_summary():
    """Load metrics summary for stats."""
    summaries = {}

    for section_key, section_name, path in [
        ("section01_rf", "Section 01 RF", MORPH_RESULTS_DIR / "section01_rf" / "metrics_summary.csv"),
        ("section02_rf", "Section 02 RF", MORPH_RESULTS_DIR / "section02_rf" / "metrics_summary.csv"),
        ("section01_logit", "Section 01 Logit", MORPH_RESULTS_DIR / "section01_logit" / "metrics_summary.csv"),
        ("section02_logit", "Section 02 Logit", MORPH_RESULTS_DIR / "section02_logit" / "metrics_summary.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = section_name
            summaries[section_key] = df

    return summaries


# ============================================================================
# Plot Functions
# ============================================================================

def plot_roc_curves(predictions_dict):
    """Plot ROC curves for all cell types in both sections."""
    print("\n📊 Plotting: ROC Curves")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, (section_key, ax) in enumerate([("section01", axes[0]), ("section02", axes[1])]):
        if section_key not in predictions_dict:
            continue

        df = predictions_dict[section_key]
        cell_types = df["cell_type"].unique()

        for ct in cell_types:
            ct_data = df[df["cell_type"] == ct]
            y_true = ct_data["y_true"].values
            y_score = ct_data["y_score"].values

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            color = get_celltype_color(ct)
            label = f"{get_celltype_display(ct)} (AUC = {roc_auc:.3f})"

            ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
            ax.fill_between(fpr, tpr, alpha=0.15, color=color)

        # Diagonal line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)

        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title(f"{'Section 01' if section_key == 'section01' else 'Section 02'} - ROC Curves",
                    fontweight="bold")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc="lower right", frameon=True, fancybox=False, shadow=False)

    plt.tight_layout()
    pdf_path = PLOTS_DIR / "figure_roc_curves.pdf"
    png_path = PLOTS_DIR / "figure_roc_curves.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path.name}")
    plt.close()


def plot_feature_importance(importance_df):
    """Plot feature importance as horizontal bar charts."""
    print("\n📊 Plotting: Feature Importance")

    # Aggregate by cell type and feature
    agg_df = importance_df.groupby(["section_key", "cell_type", "feature"])["importance"] \
        .agg(["mean", "std"]).reset_index()
    agg_df = agg_df.sort_values(["section_key", "cell_type", "mean"], ascending=[True, True, False])

    # Get top 10 features per cell type
    top_n = 10
    top_features = set()
    for (section, ct), group in agg_df.groupby(["section_key", "cell_type"]):
        top_features.update(group.nlargest(top_n, "mean")["feature"].tolist())

    plot_df = agg_df[agg_df["feature"].isin(top_features)].copy()
    plot_df["feature_display"] = plot_df["feature"].apply(get_feature_display)

    # Get unique cell types per section
    section01_ct = plot_df[plot_df["section_key"] == "section01"]["cell_type"].unique()
    section02_ct = plot_df[plot_df["section_key"] == "section02"]["cell_type"].unique()

    # Create figure with dynamic subplots
    n_ct1 = len(section01_ct)
    n_ct2 = len(section02_ct)
    max_ct = max(n_ct1, n_ct2)

    fig = plt.figure(figsize=(16, 3.5 * max_ct))
    gs = fig.add_gridspec(max_ct, 2, hspace=0.4, wspace=0.25)

    # Section 01 plots
    for idx, ct in enumerate(sorted(section01_ct)):
        ax = fig.add_subplot(gs[idx, 0])

        ct_data = plot_df[(plot_df["section_key"] == "section01") &
                         (plot_df["cell_type"] == ct)].nlargest(top_n, "mean")

        y_pos = np.arange(len(ct_data))
        colors = [get_celltype_color(ct)] * len(ct_data)

        bars = ax.barh(y_pos, ct_data["mean"], color=colors, alpha=0.8,
                      xerr=ct_data["std"], capsize=3, error_kw={'linewidth': 1})

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ct_data["feature_display"])
        ax.set_xlabel("Mean Importance", fontweight="bold")
        ax.set_title(f"Section 01 - {get_celltype_display(ct)}", fontweight="bold")
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, ct_data["mean"].max() * 1.15)

        # Add value labels
        for i, (idx_row, row) in enumerate(ct_data.iterrows()):
            ax.text(row["mean"] + row["std"] + 0.002, i,
                   f"{row['mean']:.3f}", va='center', fontsize=8)

    # Section 02 plots
    for idx, ct in enumerate(sorted(section02_ct)):
        ax = fig.add_subplot(gs[idx, 1])

        ct_data = plot_df[(plot_df["section_key"] == "section02") &
                         (plot_df["cell_type"] == ct)].nlargest(top_n, "mean")

        y_pos = np.arange(len(ct_data))
        colors = [get_celltype_color(ct)] * len(ct_data)

        bars = ax.barh(y_pos, ct_data["mean"], color=colors, alpha=0.8,
                      xerr=ct_data["std"], capsize=3, error_kw={'linewidth': 1})

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ct_data["feature_display"])
        ax.set_xlabel("Mean Importance", fontweight="bold")
        ax.set_title(f"Section 02 - {get_celltype_display(ct)}", fontweight="bold")
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, ct_data["mean"].max() * 1.15)

        for i, (idx_row, row) in enumerate(ct_data.iterrows()):
            ax.text(row["mean"] + row["std"] + 0.002, i,
                   f"{row['mean']:.3f}", va='center', fontsize=8)

    plt.tight_layout()
    pdf_path = PLOTS_DIR / "figure_feature_importance.pdf"
    png_path = PLOTS_DIR / "figure_feature_importance.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path.name}")
    plt.close()


def plot_combined_importance(importance_df):
    """Plot feature importance as compact heatmap-style comparison."""
    print("\n📊 Plotting: Combined Feature Importance")

    # Aggregate
    agg_df = importance_df.groupby(["section_key", "cell_type", "feature"])["importance"] \
        .mean().reset_index()
    agg_df["cell_type_display"] = agg_df["cell_type"].apply(get_celltype_display)
    agg_df["feature_display"] = agg_df["feature"].apply(get_feature_display)

    # Get top features overall
    overall_top = agg_df.groupby("feature")["importance"].mean().nlargest(15).index
    plot_df = agg_df[agg_df["feature"].isin(overall_top)].copy()

    # Create pivot table
    pivot = plot_df.pivot_table(
        index="feature_display",
        columns=["section_key", "cell_type_display"],
        values="importance"
    )

    # Reorder columns by section
    pivot = pivot.sort_index(axis=1, level=[0, 1])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create horizontal bar plot
    y_pos = np.arange(len(pivot.index))
    section_offset = 0.35

    for idx, (section, cell_type) in enumerate(pivot.columns):
        values = pivot[(section, cell_type)].values
        color = get_celltype_color(cell_type.split()[0].lower())

        offset = (idx % 3 - 1) * section_offset / 3
        bar_pos = y_pos + offset

        ax.barh(bar_pos, values, height=section_offset/3.5,
               color=color, alpha=0.8, label=f"{section} {cell_type}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Mean Feature Importance", fontweight="bold")
    ax.set_title("Top Morphological Features for Cell Classification",
                fontweight="bold")
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom legend
    handles = []
    for section in ["Section 01", "Section 02"]:
        for ct in pivot.columns.get_level_values(1).unique():
            color = get_celltype_color(ct.split()[0].lower())
            handles.append(mpatches.Patch(color=color, alpha=0.8,
                                         label=f"{section} - {ct}"))

    ax.legend(handles=handles, loc="lower right", ncol=2,
             frameon=True, fancybox=False, fontsize=8)

    plt.tight_layout()
    pdf_path = PLOTS_DIR / "figure_combined_importance.pdf"
    png_path = PLOTS_DIR / "figure_combined_importance.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path.name}")
    plt.close()


def plot_performance_summary(metrics_summaries):
    """Plot performance metrics summary with error bars."""
    print("\n📊 Plotting: Performance Summary")

    # Combine all metrics
    all_data = []
    for key, df in metrics_summaries.items():
        df_plot = df.copy()
        df_plot["model_key"] = key
        all_data.append(df_plot)

    combined = pd.concat(all_data, ignore_index=True)
    combined["cell_type_display"] = combined["cell_type"].apply(get_celltype_display)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: AUROC by cell type
    ax = axes[0, 0]
    auroc_df = combined[combined["metric"] == "AUROC"].copy()

    cell_types = sorted(auroc_df["cell_type_display"].unique())
    models = sorted(auroc_df["model"].unique())

    x = np.arange(len(cell_types))
    width = 0.2

    for i, model in enumerate(models):
        model_data = auroc_df[auroc_df["model"] == model]
        means = []
        errors = []
        colors = []

        for ct in cell_types:
            ct_rows = model_data[model_data["cell_type_display"] == ct]
            if len(ct_rows) > 0:
                means.append(ct_rows["mean"].values[0] * 100)
                errors.append((ct_rows["mean"].values[0] - ct_rows["ci95_low"].values[0]) * 100)
                # Get original cell_type name for color
                orig_ct = ct_rows["cell_type"].values[0]
                colors.append(get_celltype_color(orig_ct))
            else:
                means.append(0)
                errors.append(0)
                colors.append("#7f7f7f")

        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=errors,
                     label=model, capsize=3, alpha=0.85,
                     error_kw={'linewidth': 1})

        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.85)

    ax.set_ylabel("AUROC (%)", fontweight="bold")
    ax.set_title("A. AUROC by Cell Type and Model", fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, rotation=45, ha="right")
    ax.set_ylim([50, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: AUPRC by cell type
    ax = axes[0, 1]
    auprc_df = combined[combined["metric"] == "AUPRC"].copy()

    for i, model in enumerate(models):
        model_data = auprc_df[auprc_df["model"] == model]
        means = []
        errors = []
        colors = []

        for ct in cell_types:
            ct_rows = model_data[model_data["cell_type_display"] == ct]
            if len(ct_rows) > 0:
                means.append(ct_rows["mean"].values[0] * 100)
                errors.append((ct_rows["mean"].values[0] - ct_rows["ci95_low"].values[0]) * 100)
                orig_ct = ct_rows["cell_type"].values[0]
                colors.append(get_celltype_color(orig_ct))
            else:
                means.append(0)
                errors.append(0)
                colors.append("#7f7f7f")

        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=errors,
                     label=model, capsize=3, alpha=0.85,
                     error_kw={'linewidth': 1})

        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.85)

    ax.set_ylabel("AUPRC (%)", fontweight="bold")
    ax.set_title("B. AUPRC by Cell Type and Model", fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, rotation=45, ha="right")
    ax.set_ylim([0, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: Model comparison (mean performance)
    ax = axes[1, 0]

    model_stats = []
    for model in models:
        model_auroc = auroc_df[auroc_df["model"] == model]["mean"].mean() * 100
        model_auprc = auprc_df[auprc_df["model"] == model]["mean"].mean() * 100
        model_stats.append({"model": model, "AUROC": model_auroc, "AUPRC": model_auprc})

    stats_df = pd.DataFrame(model_stats)

    x_pos = np.arange(len(stats_df))
    width = 0.35

    ax.bar(x_pos - width/2, stats_df["AUROC"], width, label="AUROC",
          color="#E64B35", alpha=0.85, capsize=3)
    ax.bar(x_pos + width/2, stats_df["AUPRC"], width, label="AUPRC",
          color="#4DBBD5", alpha=0.85, capsize=3)

    ax.set_ylabel("Mean Performance (%)", fontweight="bold")
    ax.set_title("C. Model Comparison", fontweight="bold", loc="left")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace("Section ", "S") for m in stats_df["model"]])
    ax.set_ylim([0, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: Cell type difficulty ranking
    ax = axes[1, 1]

    # Calculate mean AUROC across models for each cell type
    ct_difficulty = []
    for ct in cell_types:
        ct_rows = auroc_df[auroc_df["cell_type_display"] == ct]
        mean_auroc = ct_rows["mean"].mean() * 100
        std_auroc = ct_rows["mean"].std() * 100
        ct_difficulty.append({
            "cell_type": ct,
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc
        })

    diff_df = pd.DataFrame(ct_difficulty).sort_values("mean_auroc")

    y_pos = np.arange(len(diff_df))
    colors = [get_celltype_color(ct.split()[0].lower()) for ct in diff_df["cell_type"]]

    ax.barh(y_pos, diff_df["mean_auroc"], xerr=diff_df["std_auroc"],
           color=colors, alpha=0.85, capsize=3, error_kw={'linewidth': 1})

    ax.set_xlabel("AUROC (%)", fontweight="bold")
    ax.set_title("D. Cell Type Classification Difficulty", fontweight="bold", loc="left")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(diff_df["cell_type"])
    ax.invert_yaxis()
    ax.set_xlim([50, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    pdf_path = PLOTS_DIR / "figure_performance_summary.pdf"
    png_path = PLOTS_DIR / "figure_performance_summary.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path.name}")
    plt.close()


def plot_main_figure(predictions_dict, importance_df, metrics_summaries):
    """Create the main multi-panel figure for publication."""
    print("\n📊 Plotting: Main Publication Figure")

    # Get cell types
    section01_ct = predictions_dict["section01"]["cell_type"].unique()
    section02_ct = predictions_dict["section02"]["cell_type"].unique()

    # Create figure with 3 rows, 4 columns
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35,
                         left=0.08, right=0.95, top=0.94, bottom=0.06)

    # ========================================================================
    # Row 1: ROC Curves
    # ========================================================================

    # Panel A: Section 01 ROC
    ax = fig.add_subplot(gs[0, 0:2])

    df = predictions_dict["section01"]
    for ct in section01_ct:
        ct_data = df[df["cell_type"] == ct]
        y_true = ct_data["y_true"].values
        y_score = ct_data["y_score"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        color = get_celltype_color(ct)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
               label=f"{get_celltype_display(ct)} (AUC={roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.2, color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("A. Section 01 - ROC Curves", fontweight="bold", loc="left", fontsize=13)
    ax.legend(loc="lower right", frameon=True, fancybox=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Section 02 ROC
    ax = fig.add_subplot(gs[0, 2:4])

    df = predictions_dict["section02"]
    for ct in section02_ct:
        ct_data = df[df["cell_type"] == ct]
        y_true = ct_data["y_true"].values
        y_score = ct_data["y_score"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        color = get_celltype_color(ct)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
               label=f"{get_celltype_display(ct)} (AUC={roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.2, color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("B. Section 02 - ROC Curves", fontweight="bold", loc="left", fontsize=13)
    ax.legend(loc="lower right", frameon=True, fancybox=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ========================================================================
    # Row 2: Feature Importance
    # ========================================================================

    # Aggregate feature importance
    agg_df = importance_df.groupby(["section_key", "cell_type", "feature"])["importance"] \
        .agg(["mean", "std"]).reset_index()
    agg_df["feature_display"] = agg_df["feature"].apply(get_feature_display)

    top_n = 8

    # Panel C: Section 01 Feature Importance
    ax = fig.add_subplot(gs[1, 0:2])

    ct_labels = []
    y_positions = []
    importances = []
    errors = []
    colors_list = []

    y_offset = 0
    for ct_idx, ct in enumerate(sorted(section01_ct)):
        ct_data = agg_df[(agg_df["section_key"] == "section01") &
                         (agg_df["cell_type"] == ct)].nlargest(top_n, "mean")

        for i, (idx_row, row) in enumerate(ct_data.iterrows()):
            y_positions.append(y_offset)
            ct_labels.append(row["feature_display"])
            importances.append(row["mean"])
            errors.append(row["std"])
            colors_list.append(get_celltype_color(ct))

            # Add cell type label for first feature
            if i == 0:
                ax.text(-0.01, y_offset, get_celltype_display(ct),
                       ha="right", va="center", fontweight="bold", fontsize=9)

            y_offset += 1

        y_offset += 0.5  # Gap between cell types

    y_positions = np.array(y_positions)

    ax.barh(y_positions, importances, color=colors_list, alpha=0.85,
           xerr=errors, capsize=2.5, error_kw={'linewidth': 1})

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ct_labels, fontsize=8)
    ax.set_xlabel("Feature Importance", fontweight="bold")
    ax.set_title("C. Section 01 - Top Features", fontweight="bold", loc="left", fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, max(importances) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel D: Section 02 Feature Importance
    ax = fig.add_subplot(gs[1, 2:4])

    ct_labels = []
    y_positions = []
    importances = []
    errors = []
    colors_list = []

    y_offset = 0
    for ct_idx, ct in enumerate(sorted(section02_ct)):
        ct_data = agg_df[(agg_df["section_key"] == "section02") &
                         (agg_df["cell_type"] == ct)].nlargest(top_n, "mean")

        for i, (idx_row, row) in enumerate(ct_data.iterrows()):
            y_positions.append(y_offset)
            ct_labels.append(row["feature_display"])
            importances.append(row["mean"])
            errors.append(row["std"])
            colors_list.append(get_celltype_color(ct))

            if i == 0:
                ax.text(-0.01, y_offset, get_celltype_display(ct),
                       ha="right", va="center", fontweight="bold", fontsize=9)

            y_offset += 1

        y_offset += 0.5

    y_positions = np.array(y_positions)

    ax.barh(y_positions, importances, color=colors_list, alpha=0.85,
           xerr=errors, capsize=2.5, error_kw={'linewidth': 1})

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ct_labels, fontsize=8)
    ax.set_xlabel("Feature Importance", fontweight="bold")
    ax.set_title("D. Section 02 - Top Features", fontweight="bold", loc="left", fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, max(importances) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ========================================================================
    # Row 3: Performance Summary
    # ========================================================================

    # Prepare combined data
    all_data = []
    for key, df in metrics_summaries.items():
        df_plot = df.copy()
        df_plot["model_key"] = key
        all_data.append(df_plot)
    combined = pd.concat(all_data, ignore_index=True)
    combined["cell_type_display"] = combined["cell_type"].apply(get_celltype_display)

    # Panel E: AUROC comparison
    ax = fig.add_subplot(gs[2, 0:2])

    auroc_df = combined[combined["metric"] == "AUROC"].copy()
    cell_types = sorted(auroc_df["cell_type_display"].unique())
    models = sorted([m for m in auroc_df["model"].unique() if "RF" in m])

    x = np.arange(len(cell_types))
    width = 0.35

    for i, model in enumerate(models):
        model_data = auroc_df[auroc_df["model"] == model]
        means = []
        errors = []
        colors = []

        for ct in cell_types:
            ct_rows = model_data[model_data["cell_type_display"] == ct]
            if len(ct_rows) > 0:
                means.append(ct_rows["mean"].values[0] * 100)
                errors.append((ct_rows["mean"].values[0] - ct_rows["ci95_low"].values[0]) * 100)
                orig_ct = ct_rows["cell_type"].values[0]
                colors.append(get_celltype_color(orig_ct))
            else:
                means.append(0)
                errors.append(0)
                colors.append("#7f7f7f")

        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=errors,
                     label=model.replace("Section ", "S"), capsize=3, alpha=0.85,
                     error_kw={'linewidth': 1.2})

        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_edgecolor('white')
            bar.set_linewidth(0.5)

    ax.set_ylabel("AUROC (%)", fontweight="bold")
    ax.set_title("E. AUROC by Cell Type", fontweight="bold", loc="left", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, rotation=45, ha="right")
    ax.set_ylim([60, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", linewidth=0.5)

    # Panel F: Summary statistics table
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis("off")

    # Build summary table
    table_data = []
    for ct in sorted(auroc_df["cell_type_display"].unique()):
        ct_rows = auroc_df[auroc_df["cell_type_display"] == ct]

        # Get RF model results
        rf_rows = ct_rows[ct_rows["model"].str.contains("RF")]

        if len(rf_rows) > 0:
            mean_auroc = rf_rows["mean"].mean() * 100
            ci_low = rf_rows["ci95_low"].mean() * 100
            ci_high = rf_rows["ci95_high"].mean() * 100

            table_data.append([
                ct,
                f"{mean_auroc:.1f}",
                f"[{ci_low:.1f}, {ci_high:.1f}]"
            ])

    table = ax.table(cellText=table_data,
                     colLabels=["Cell Type", "Mean AUROC (%)", "95% CI"],
                     cellLoc="center",
                     loc="center",
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor("#E5E5E5")
        table[(0, i)].set_text_props(weight="bold")

    # Color rows by cell type
    for i, (ct, _, _) in enumerate(table_data):
        color = get_celltype_color(ct.split()[0].lower())
        for j in range(3):
            table[(i + 1, j)].set_facecolor(color)
            table[(i + 1, j)].set_alpha(0.3)

    ax.set_title("F. Performance Summary", fontweight="bold", loc="left", fontsize=13)

    # Main figure title
    fig.suptitle("Morphology-Based Cell Classification Performance Analysis",
                fontsize=16, fontweight="bold", y=0.98)

    # Save
    pdf_path = PLOTS_DIR / "figure_main_publication.pdf"
    png_path = PLOTS_DIR / "figure_main_publication.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path.name}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("Academic Figure Generation for Genome Biology")
    print("="*70)

    # Load data
    predictions = load_predictions()
    importance = load_feature_importance()
    metrics = load_metrics_summary()

    # Generate figures
    plot_roc_curves(predictions)
    plot_feature_importance(importance)
    plot_combined_importance(importance)
    plot_performance_summary(metrics)
    plot_main_figure(predictions, importance, metrics)

    print("\n" + "="*70)
    print(f"All figures saved to: {PLOTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
