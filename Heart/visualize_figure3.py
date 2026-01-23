"""
Figure 3: Morphology-based Classification Performance

Panels:
    A. Object Recall by algorithm
    B. Pixel Recall by algorithm
    C. Example annotations for each heart cell type
    D. ROC curves for all cell types

Style: Genome Biology academic standards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from PIL import Image
import roifile
import warnings

from _paths import HEART_DATA_ROOT, NUC_MOFO_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================

EVAL_CSV = HEART_DATA_ROOT / "evaluation_results.csv"
MORPH_RESULTS_DIR = NUC_MOFO_ROOT / "results"
PROCESSED_DIR = HEART_DATA_ROOT / "processed"
GT_MASKS_DIR = HEART_DATA_ROOT / "ground_truth_masks"
RAW_DATA_DIR = NUC_MOFO_ROOT / "data" / "raw"
PLOTS_DIR = HEART_DATA_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Cell types: (short, display, marker, color, marker_file, section, roi_subdir)
CELLTYPES = [
    ("Epi", "Epicardial", "ALDH1A2", "#E64B35", "aldh1a2", "01_section", "epi"),
    ("Immune", "Immune", "CD45", "#4DBBD5", "cd45", "01_section", "Immune cell"),
    ("Mural", "Mural", "PDGFRB", "#00A087", "pdgfrb", "01_section", "Mural cell"),
    ("CM", "Cardiomyocyte", "WGA", "#F39B7F", "cm", "02_section", "cm"),
    ("EC", "Endothelial", "PECAM", "#3C5488", "ec", "02_section", "ec"),
    ("FB", "Fibroblast", "VIM", "#8491B4", "fb", "02_section", "fb"),
]

# Style settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

ALGORITHM_COLORS = {
    "cellpose": "#0072B2", "cellpose_sam": "#0072B2",
    "stardist": "#E69F00", "omnipose": "#CC79A7",
    "watershed": "#9467BD", "mesmer": "#D55E00",
    "lacss": "#F0E442", "splinedist": "#999999",
    "microsam": "#56B4E9", "cellsam": "#009E73",
    "instanseg": "#17BECF",
}

ALGORITHM_DISPLAY = {
    "cellpose": "Cellpose", "cellpose_sam": "CellposeSAM",
    "stardist": "StarDist", "omnipose": "Omnipose",
    "watershed": "Watershed", "mesmer": "MESMER",
    "lacss": "LACSS", "splinedist": "SplineDist",
    "microsam": "MicroSAM", "cellsam": "CellSAM",
    "instanseg": "InstanSeg",
}

CELLTYPE_COLORS = {
    "epi": "#E64B35", "epithelial": "#E64B35",
    "immune cell": "#4DBBD5", "immune": "#4DBBD5",
    "mural cell": "#00A087", "mural": "#00A087",
    "cm": "#F39B7F", "cardiomyocyte": "#F39B7F",
    "ec": "#3C5488", "endothelial": "#3C5488",
    "fb": "#8491B4", "fibroblast": "#8491B4",
}

CELLTYPE_DISPLAY = {
    "epi": "Epicardial", "epithelial": "Epicardial",
    "immune cell": "Immune", "immune": "Immune",
    "mural cell": "Mural", "mural": "Mural",
    "cm": "Cardiomyocyte", "cardiomyocyte": "Cardiomyocyte",
    "ec": "Endothelial", "endothelial": "Endothelial",
    "fb": "Fibroblast", "fibroblast": "Fibroblast",
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_algo_display(name: str) -> str:
    return ALGORITHM_DISPLAY.get(name.lower(), name)


def get_celltype_display(name: str) -> str:
    return CELLTYPE_DISPLAY.get(name.lower(), name.title())


def get_celltype_color(name: str) -> str:
    return CELLTYPE_COLORS.get(name.lower(), "#666666")


def add_panel_label(ax, label: str, x: float = -0.15, y: float = 1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


def load_ome_tiff_channel(ome_path, channel_idx=0):
    """Load a specific channel from OME-TIFF file."""
    img = Image.open(ome_path)
    try:
        img.seek(channel_idx)
        return np.array(img, dtype=np.float32)
    except:
        return None


def normalize_image(img):
    """Normalize image to 0-1."""
    if img is None:
        return None
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1), 0, 1)
    else:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def make_composite(dapi, marker):
    """Create RGB composite: DAPI=blue, Marker=red."""
    d = normalize_image(dapi) if dapi is not None else None
    m = normalize_image(marker) if marker is not None else None

    if d is None and m is None:
        return None

    if d is not None and m is not None:
        h, w = min(d.shape[0], m.shape[0]), min(d.shape[1], m.shape[1])
        d, m = d[:h, :w], m[:h, :w]
    elif d is not None:
        h, w = d.shape
    else:
        h, w = m.shape

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if d is not None:
        rgb[..., 2] = d  # Blue
    if m is not None:
        rgb[..., 0] = m  # Red
        rgb[..., 1] = m * 0.2  # Slight green

    return rgb


def read_roi_polygon(roi_path):
    """Read ImageJ ROI polygon coordinates and return (x_coords, y_coords)."""
    try:
        roi = roifile.roiread(roi_path)
        coords = roi.coordinates()
        if coords is not None and len(coords) > 0:
            # roifile returns (x, y)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            return x_coords, y_coords
        return None, None
    except Exception as e:
        print(f"[WARNING] Failed to read ROI {roi_path}: {e}")
        return None, None


# ============================================================================
# Data Loading
# ============================================================================

def load_segmentation_data() -> pd.DataFrame | None:
    if not EVAL_CSV.exists():
        print(f"[WARNING] Segmentation data not found: {EVAL_CSV}")
        return None
    df = pd.read_csv(EVAL_CSV)
    df["algorithm_display"] = df["algorithm"].apply(get_algo_display)
    print(f"[OK] Loaded segmentation data: {len(df)} rows")
    return df


def load_predictions() -> pd.DataFrame | None:
    sections = [
        ("Section 01", MORPH_RESULTS_DIR / "section01_rf" / "predictions_last_repeat.csv"),
        ("Section 02", MORPH_RESULTS_DIR / "section02_rf" / "predictions_last_repeat.csv"),
    ]
    dfs = []
    for name, path in sections:
        if path.exists():
            df = pd.read_csv(path)
            df["section"] = name
            dfs.append(df)
            print(f"[OK] Loaded {name} predictions: {len(df)} rows")
    if not dfs:
        print("[WARNING] No prediction files found")
        return None
    return pd.concat(dfs, ignore_index=True)


def load_metrics() -> pd.DataFrame | None:
    sections = [
        ("Section 01", MORPH_RESULTS_DIR / "section01_rf" / "metrics_summary.csv"),
        ("Section 02", MORPH_RESULTS_DIR / "section02_rf" / "metrics_summary.csv"),
    ]
    dfs = []
    for name, path in sections:
        if path.exists():
            df = pd.read_csv(path)
            df["section"] = name
            dfs.append(df)
            print(f"[OK] Loaded {name} metrics: {len(df)} rows")
    if not dfs:
        print("[WARNING] No metrics files found")
        return None
    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Plot Functions
# ============================================================================

def plot_recall_boxplot(df: pd.DataFrame, metric: str, ylabel: str, ax):
    """Create boxplot for recall metrics."""
    df_plot = df.copy()
    df_plot[metric] = df_plot[metric] * 100
    order = (
        df_plot.groupby("algorithm_display")[metric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [df_plot[df_plot["algorithm_display"] == algo][metric].dropna().values
            for algo in order]

    bp = ax.boxplot(
        data, tick_labels=order, patch_artist=True, showmeans=False,
        showfliers=True, widths=0.65,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        medianprops=dict(linewidth=0),
        whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
    )

    for patch, algo_disp in zip(bp["boxes"], order):
        orig = df_plot[df_plot["algorithm_display"] == algo_disp]["algorithm"].iloc[0]
        color = ALGORITHM_COLORS.get(orig.lower(), "#666666")
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 105])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", rotation=50, labelsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)


def plot_cell_examples(ax):
    """Plot example annotations for 6 heart cell types (2x3 grid)."""
    ax.axis('off')

    ROWS, COLS = 2, 3
    PADDING = 0.02
    TITLE_SPACE = 0.06  # Extra room for panel titles
    CELL_WIDTH = (1.0 - (COLS + 1) * PADDING) / COLS
    CELL_HEIGHT = (1.0 - (ROWS + 1) * PADDING - ROWS * TITLE_SPACE) / ROWS

    SECTION01_CHANNELS = {"Epi": 2, "Immune": 3, "Mural": 4}
    SECTION02_CHANNELS = {"CM": 1, "EC": 2, "FB": 3}

    section01_dir = RAW_DATA_DIR / "01_section"
    section02_dir = RAW_DATA_DIR / "02_section"
    
    section01_omes = list(section01_dir.glob("*.ome.tif*"))
    section02_omes = list(section02_dir.glob("*.ome.tif*"))
    
    if not section01_omes or not section02_omes:
        print("[WARNING] OME-TIFF files not found")
        return
    
    section01_ome = section01_omes[0]
    section02_ome = section02_omes[0]

    crop_h = 350  # Height
    crop_w = 250  # Width

    for idx, (cell_type, display_name, marker_name, color, marker_file,
              section, roi_subdir) in enumerate(CELLTYPES):
        row = idx // COLS
        col = idx % COLS
        y_pos = 1.0 - (row + 1) * (CELL_HEIGHT + TITLE_SPACE) - (row + 1) * PADDING
        x_pos = col * CELL_WIDTH + (col + 1) * PADDING

        if section == "01_section":
            ome_path = section01_ome
            roi_dir = section01_dir / roi_subdir
            dapi_idx, marker_idx = 1, SECTION01_CHANNELS[cell_type]
        else:
            ome_path = section02_ome
            roi_dir = section02_dir / roi_subdir
            dapi_idx, marker_idx = 0, SECTION02_CHANNELS[cell_type]

        print(f"[INFO] Loading {cell_type} from {ome_path.name}...")
        dapi = load_ome_tiff_channel(str(ome_path), dapi_idx)
        marker = load_ome_tiff_channel(str(ome_path), marker_idx)

        if dapi is None or marker is None:
            print(f"[WARNING] Failed to load images for {cell_type}")
            continue

        roi_files = list(roi_dir.glob("*.roi"))
        roi_files = [f for f in roi_files if '-' in f.stem]

        if not roi_files:
            print(f"[WARNING] No ROI files found for {cell_type} in {roi_dir}")
            continue

        print(f"[INFO] Found {len(roi_files)} ROI files for {cell_type}")

        # Read all ROI coordinates (x, y)
        all_rois = []
        for rf in roi_files:
            x_coords, y_coords = read_roi_polygon(rf)
            if x_coords is not None:
                all_rois.append((x_coords, y_coords))

        if not all_rois:
            print(f"[WARNING] No valid ROI coordinates for {cell_type}")
            continue

        # Debug info
        first_x, first_y = all_rois[0]
        print(f"[DEBUG] First ROI range: x=[{first_x.min():.0f}, {first_x.max():.0f}], "
              f"y=[{first_y.min():.0f}, {first_y.max():.0f}]")
        print(f"[DEBUG] Image shape (h, w): {dapi.shape}")

        # Find the region with the densest ROIs
        # Image coordinates: shape = (height, width) i.e., (rows, cols)
        # ROI coordinates: x maps to width (cols), y maps to height (rows)
        img_h, img_w = dapi.shape
        
        # Compute each ROI center (center_x, center_y)
        centers = np.array([[np.mean(x), np.mean(y)] for x, y in all_rois])
        
        # Slide a window to identify the best crop location
        best_count = 0
        best_x1, best_y1 = 0, 0
        
        step_y = crop_h // 4
        step_x = crop_w // 4
        for start_y in range(0, max(1, img_h - crop_h), step_y):
            for start_x in range(0, max(1, img_w - crop_w), step_x):
                # centers[:, 0] = x, centers[:, 1] = y
                in_region = ((centers[:, 0] >= start_x) & (centers[:, 0] < start_x + crop_w) &
                            (centers[:, 1] >= start_y) & (centers[:, 1] < start_y + crop_h))
                count = np.sum(in_region)
                if count > best_count:
                    best_count = count
                    best_x1, best_y1 = start_x, start_y

        print(f"[DEBUG] Best region: x1={best_x1}, y1={best_y1}, contains {best_count} ROIs")

        # Crop: [y1:y2, x1:x2]
        y1, y2 = best_y1, min(best_y1 + crop_h, img_h)
        x1, x2 = best_x1, min(best_x1 + crop_w, img_w)

        dapi_crop = dapi[y1:y2, x1:x2]
        marker_crop = marker[y1:y2, x1:x2]
        composite = make_composite(dapi_crop, marker_crop)

        inset = ax.inset_axes([x_pos, y_pos, CELL_WIDTH, CELL_HEIGHT])
        inset.imshow(composite)

        # Draw ROI outlines
        roi_count = 0
        for roi_x, roi_y in all_rois:
            # Convert to crop coordinates
            roi_x_crop = roi_x - x1
            roi_y_crop = roi_y - y1

            # Ensure the ROI center resides within the crop
            center_x = np.mean(roi_x_crop)
            center_y = np.mean(roi_y_crop)
            
            actual_h, actual_w = y2 - y1, x2 - x1
            if 0 <= center_x < actual_w and 0 <= center_y < actual_h:
                # Close the polygon
                roi_x_closed = np.append(roi_x_crop, roi_x_crop[0])
                roi_y_closed = np.append(roi_y_crop, roi_y_crop[0])
                # In matplotlib, x is horizontal and y is vertical
                inset.plot(roi_x_closed, roi_y_closed, 
                          color='white', linewidth=1.2, alpha=0.9)
                roi_count += 1

        print(f"[INFO] Drew {roi_count} ROI contours for {cell_type}")

        inset.set_xlim(0, x2 - x1)
        inset.set_ylim(y2 - y1, 0)  # Flip y-axis to show the image upright
        inset.axis('off')
        
        # Place panel titles above the images
        ax.text(x_pos + CELL_WIDTH / 2, y_pos + CELL_HEIGHT + 0.01, display_name,
                transform=ax.transAxes, fontsize=9, ha='center', va='bottom')


def plot_roc_curves(predictions_df: pd.DataFrame, ax):
    """Plot ROC curves for all cell types."""
    cell_types = predictions_df["cell_type"].unique()
    roc_data = []
    for ct in cell_types:
        ct_data = predictions_df[predictions_df["cell_type"] == ct]
        y_true = ct_data["y_true"].values
        y_score = ct_data["y_score"].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_data.append({"cell_type": ct, "fpr": fpr, "tpr": tpr, "auc": roc_auc})

    roc_data = sorted(roc_data, key=lambda x: x["auc"], reverse=True)

    for item in roc_data:
        ct = item["cell_type"]
        color = get_celltype_color(ct)
        label = f"{get_celltype_display(ct)} ({item['auc']:.3f})"
        ax.plot(item["fpr"], item["tpr"], color=color, linewidth=1.5,
                label=label, alpha=0.9)
        ax.fill_between(item["fpr"], item["tpr"], alpha=0.1, color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(loc="lower right", fontsize=7, title="Cell Type (AUC)", 
              title_fontsize=8, handlelength=1.5)


# ============================================================================
# Main Figure
# ============================================================================

def create_figure3():
    """Generate Figure 3: Morphology-based Classification Performance."""
    print("\n" + "=" * 60)
    print("Generating Figure 3: Morphology-based Classification")
    print("=" * 60)

    # Load data
    seg_data = load_segmentation_data()
    preds_data = load_predictions()
    metrics_data = load_metrics()

    if seg_data is None or preds_data is None or metrics_data is None:
        print("[ERROR] Missing required data files. Aborting.")
        return

    # Create figure
    fig = plt.figure(figsize=(9, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          left=0.10, right=0.97, top=0.95, bottom=0.05,
                          hspace=0.28, wspace=0.28)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel A: Object Recall
    plot_recall_boxplot(seg_data, "object_recall", "Object Recall (%)", ax_a)
    add_panel_label(ax_a, "A")

    # Panel B: Pixel Recall
    plot_recall_boxplot(seg_data, "pixel_recall", "Pixel Recall (%)", ax_b)
    add_panel_label(ax_b, "B")

    # Panel C: Example cell annotations
    plot_cell_examples(ax_c)
    add_panel_label(ax_c, "C")

    # Panel D: ROC curves
    plot_roc_curves(preds_data, ax_d)
    add_panel_label(ax_d, "D")

    # Save
    pdf_path = PLOTS_DIR / "figure3_morphology_classification.pdf"
    png_path = PLOTS_DIR / "figure3_morphology_classification.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    print(f"\n[SAVED] {pdf_path}")
    print(f"[SAVED] {png_path}")
    print("=" * 60 + "\n")

    plt.close(fig)
    return fig


if __name__ == "__main__":
    create_figure3()
