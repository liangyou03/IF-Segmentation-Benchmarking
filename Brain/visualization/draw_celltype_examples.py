#!/usr/bin/env python3
"""
draw_celltype_examples.py

Generates a 3x4 academic-style figure showing:
- Row A: Raw composite images (DAPI + marker)
- Row B: Unrefined predictions with GT contours overlay
- Row C: Best refined predictions with GT contours overlay

For each cell type (OLIG2, NEUN, IBA1, GFAP), the image + algorithm
combination with the best mAP is selected from ALL available algorithms.
"""

from pathlib import Path
import re
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile as tiff
from skimage.measure import find_contours

# Import from visualization config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CELL_2CH_ALGOS, DATASET_DIR, RESULTS_DIR, PLOTS_DIR,
    CELL_TYPE_GROUPS, ALGORITHM_COLORS
)

# Apply scientific publication style
try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass

# Set clean fonts and publication parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
})

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Algorithm directories mapping
# Refined predictions from config
REFINED_DIRS = CELL_2CH_ALGOS

# Unrefined predictions (cyto_prediction for each algorithm)
UNREFINED_DIRS = {
    "CellposeSAM": Path("/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/cyto_prediction"),
    "StarDist": Path("/ihome/jbwang/liy121/ifimage/02_stardist_benchmark/cyto_prediction"),
    "CellSAM": Path("/ihome/jbwang/liy121/ifimage/03_cellsam_benchmark/cyto_prediction"),
    "MESMER": Path("/ihome/jbwang/liy121/ifimage/04_mesmer_benchmark/cyto_prediction"),
    "Watershed": Path("/ihome/jbwang/liy121/ifimage/06_watershed_benchmark/cyto_prediction"),
    "Omnipose": Path("/ihome/jbwang/liy121/ifimage/07_omnipose_benchmark/cyto_prediction"),
    "LACSS": Path("/ihome/jbwang/liy121/ifimage/011_lacss/cyto_prediction"),
    "SplineDist": Path("/ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/cyto_prediction"),
    "MicroSAM": Path("/ihome/jbwang/liy121/ifimage/012_microsam_benchmark/cyto_prediction"),
    "InstanSeg": Path("/ihome/jbwang/liy121/ifimage/013_ins/cyto_prediction"),
}

# Output paths
OUT_PNG = PLOTS_DIR / "best_per_celltype_all_algos_3x4.png"
OUT_PDF = PLOTS_DIR / "best_per_celltype_all_algos_3x4.pdf"

# Visualization parameters
BOUNDARY_W = 1.0
FILL_ALPHA = 0.90
DPI = 300
FIGURE_WIDTH = 18   # cm for 4 columns
FIGURE_HEIGHT = 13  # cm for 3 rows (increased for text below panels)


# ==============================================================================
# FILE I/O UTILITIES
# ==============================================================================

def load_any_image(p: Path) -> np.ndarray:
    """Load image from .npy or .tif/.tiff file."""
    if p.suffix.lower() == ".npy":
        return np.load(p)
    else:
        return np.squeeze(tiff.imread(str(p)))


def to_gray(x: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] >= 3:
        # RGB to grayscale using ITU-R 601-2 luma transform
        return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)
    elif x.ndim == 3:
        return x.max(axis=-1).astype(np.float32)
    elif x.ndim > 2:
        x = np.squeeze(x)
        if x.ndim > 2:
            return x.max(axis=0).astype(np.float32)
    return x.astype(np.float32)


def normalize_01(x: np.ndarray) -> np.ndarray:
    """Percentile-based normalization to [0, 1]."""
    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    if p99 <= p1:
        mn, mx = float(x.min()), float(x.max())
        if mx <= mn:
            return np.zeros_like(x, dtype=np.float32)
        p1, p99 = mn, mx
    return np.clip((x - p1) / max(p99 - p1, 1e-6), 0, 1).astype(np.float32)


def find_first_match(root: Path, patterns: list) -> Path | None:
    """Find first file matching any pattern."""
    for pat in patterns:
        match = next(root.glob(pat), None)
        if match is not None:
            return match
    return None


def find_raw_channels(base: str, dataset_dir: Path):
    """Find DAPI and marker channel files for a given base name."""
    lead = base.split("_", 1)[0].lower()
    
    # Try to find DAPI
    dapi_patterns = [
        f"{base}.tif", f"{base}.tiff", f"{base}.npy",
        f"{base}_dapi.tif", f"{base}_dapi.tiff", f"{base}_dapi.npy"
    ]
    dapi_path = find_first_match(dataset_dir, dapi_patterns)
    
    # Try to find marker
    marker_patterns = [
        f"{base}_{lead}.tif", f"{base}_{lead}.tiff", f"{base}_{lead}.npy",
        f"{base}_marker.tif", f"{base}_marker.tiff", f"{base}_marker.npy"
    ]
    marker_path = find_first_match(dataset_dir, marker_patterns)
    
    dapi = load_any_image(dapi_path) if dapi_path else None
    marker = load_any_image(marker_path) if marker_path else None
    
    return dapi, marker


def make_composite(dapi, marker):
    """Create RGB composite: DAPI=blue, Marker=red."""
    if dapi is None and marker is None:
        return None
    
    d = to_gray(dapi) if dapi is not None else None
    m = to_gray(marker) if marker is not None else None
    
    if d is not None and m is not None:
        h, w = min(d.shape[0], m.shape[0]), min(d.shape[1], m.shape[1])
        d, m = d[:h, :w], m[:h, :w]
    
    base = d if d is not None else m
    H, W = base.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    
    if d is not None:
        rgb[..., 2] = normalize_01(d)  # Blue channel
    if m is not None:
        rgb[..., 0] = normalize_01(m)  # Red channel
    
    return rgb


def stack_to_label(arr: np.ndarray) -> np.ndarray:
    """Convert multi-channel binary stack to label image."""
    if arr.ndim == 3:
        lab = np.zeros(arr.shape[1:], dtype=np.int32)
        k = 0
        for i in range(arr.shape[0]):
            m = arr[i] > 0
            if m.any():
                k += 1
                lab[m] = k
        return lab
    return arr.astype(np.int32)


def find_ground_truth(base: str, dataset_dir: Path) -> Path | None:
    """Find ground truth mask file."""
    patterns = [
        f"{base}_cellbodies.npy",
    ]
    path = find_first_match(dataset_dir, patterns)
    if path is None:
        # Try recursive search
        path = next(dataset_dir.rglob(f"{base}*cellbodies*.npy"), None)
    return path


def find_prediction(base: str, root: Path) -> Path | None:
    """Find prediction mask file."""
    if root is None or not root.exists():
        return None
    
    patterns = [
        f"{base}.npy",  # Direct match
        f"{base}_pred_cyto.npy",
        f"{base}_pred.npy",
    ]
    path = find_first_match(root, patterns)
    if path is None:
        path = next(root.rglob(f"{base}*pred*.npy"), None)
    return path


# ==============================================================================
# MASK VISUALIZATION
# ==============================================================================

def seed_from_text(s: str) -> int:
    """Generate deterministic seed from string."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def pastel_palette(n: int, seed: int) -> np.ndarray:
    """Generate n pastel colors with given seed."""
    rng = np.random.default_rng(seed)
    hues = rng.random(n)
    S, V = 0.35, 0.95  # Pastel saturation and value
    rgb = []
    for h in hues:
        i = int(h * 6.0) % 6
        f = h * 6.0 - i
        p = V * (1 - S)
        q = V * (1 - S * f)
        t = V * (1 - S * (1 - f))
        if i == 0:
            r, g, b = V, t, p
        elif i == 1:
            r, g, b = q, V, p
        elif i == 2:
            r, g, b = p, V, t
        elif i == 3:
            r, g, b = p, q, V
        elif i == 4:
            r, g, b = t, p, V
        else:
            r, g, b = V, p, q
        rgb.append((r, g, b))
    return np.array(rgb, dtype=np.float32)


def label_viz_on_gray(lab: np.ndarray, *, seed_text: str, fill_alpha: float = FILL_ALPHA) -> np.ndarray:
    """Render label mask as pastel colors on light gray background."""
    H, W = lab.shape[:2]
    # Light gray background (#fafafa = 250/255 ≈ 0.98)
    img = np.full((H, W, 3), 0.98, dtype=np.float32)
    
    ids = [i for i in np.unique(lab) if i != 0]
    if not ids:
        return img
    
    pal = pastel_palette(len(ids), seed_from_text(seed_text))
    id2idx = {i: k for k, i in enumerate(ids)}
    
    for i in ids:
        m = (lab == i)
        if not m.any():
            continue
        c = pal[id2idx[i]]
        img[m] = img[m] * (1 - fill_alpha) + c * fill_alpha
    
    return img


def draw_contours(ax, binary_mask, color="#444444", lw=1.0):
    """Draw contours of binary mask on axes with clean style."""
    for cnt in find_contours(binary_mask.astype(float), 0.5):
        ax.plot(cnt[:, 1], cnt[:, 0], color=color, linewidth=lw,
                solid_capstyle="round", solid_joinstyle="round")


# ==============================================================================
# EVALUATION DATA LOADING
# ==============================================================================

def load_evaluation_data(eval_dir: Path) -> pd.DataFrame:
    """Load and prepare evaluation data."""
    per_img_path = eval_dir / "cell_2ch_per_image.csv"
    if not per_img_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {per_img_path}")
    
    df = pd.read_csv(per_img_path)
    
    # Ensure required columns exist
    if "base" not in df.columns:
        raise ValueError("Missing 'base' column in evaluation results")
    if "algorithm" not in df.columns:
        df["algorithm"] = "unknown"
    
    return df


def get_ap_columns(df: pd.DataFrame) -> list:
    """Get AP@xx.xx columns sorted by threshold."""
    ap_cols = [c for c in df.columns if re.match(r"^AP@\d\.\d{2}$", c)]
    if not ap_cols:
        raise ValueError("No AP@xx.xx columns found in evaluation results")
    return sorted(ap_cols, key=lambda c: float(c.split("@")[1]))


# ==============================================================================
# MAIN PLOTTING FUNCTION
# ==============================================================================

def main():
    print("=" * 70)
    print("Cell Type Example Figure Generator (Publication-Ready)")
    print("=" * 70)
    
    # Create output directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation results
    print(f"\nLoading evaluation results from: {RESULTS_DIR}")
    df = load_evaluation_data(RESULTS_DIR)
    print(f"  Loaded {len(df)} evaluation records")
    
    # Get AP columns and compute score
    ap_cols = get_ap_columns(df)
    df["_score"] = pd.to_numeric(df[ap_cols].mean(axis=1), errors="coerce")
    df = df.dropna(subset=["_score"])
    print(f"  AP columns: {ap_cols[:3]}... (total {len(ap_cols)})")
    
    # Normalize algorithm names for matching
    df["algo_normalized"] = df["algorithm"].str.replace(" ", "", regex=False).str.replace("_", "", regex=False)
    
    # Available algorithms
    available_refined = {k: v for k, v in REFINED_DIRS.items() if v and v.exists()}
    available_unrefined = {k: v for k, v in UNREFINED_DIRS.items() if v and v.exists()}
    
    print(f"\n  Available refined algorithms: {list(available_refined.keys())}")
    print(f"  Available unrefined algorithms: {list(available_unrefined.keys())}")
    
    # Find best image + algorithm for each cell type
    best_images = []
    for cell_type, pattern in CELL_TYPE_GROUPS.items():
        # Filter by cell type
        sub_df = df[df["base"].str.contains(pattern, case=False, regex=True, na=False)]
        if sub_df.empty:
            print(f"  WARNING: No images found for {cell_type}")
            best_images.append((cell_type, None))
            continue
        
        # Find best mAP across all algorithms
        best_row = sub_df.loc[sub_df["_score"].idxmax()]
        best_images.append((cell_type, best_row))
        
        base_name = best_row["base"]
        algo_name = best_row["algorithm"]
        ap50 = best_row.get("AP@0.50", np.nan)
        print(f"  {cell_type}: {base_name} | {algo_name} | Precision@0.5={ap50:.3f}")
    
    # Create figure with GridSpec for different row spacing
    n_cols = len(CELL_TYPE_GROUPS)
    n_rows = 3
    fig_width_inches = FIGURE_WIDTH / 2.54
    fig_height_inches = FIGURE_HEIGHT / 2.54

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
    fig.patch.set_facecolor('white')
    
    # GridSpec with custom height ratios and spacing
    # Row A-B gap small, B-C gap larger for text
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  left=0.10, right=0.98, top=0.92, bottom=0.06,
                  hspace=0.22, wspace=0.02,
                  height_ratios=[1, 1.15, 1.15])  # Extra height for text rows
    
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(n_rows)])
    
    # Store text info for each panel (to add below images)
    panel_texts = {(i, j): "" for i in range(n_rows) for j in range(n_cols)}
    
    # Plot each column (cell type)
    for j, (cell_type, best_row) in enumerate(best_images):
        # Row 0: Raw, Row 1: Unrefined, Row 2: Refined
        ax_raw, ax_unr, ax_ref = axes[0, j], axes[1, j], axes[2, j]
        
        # Set facecolor for each subplot (row 1&2 light gray)
        ax_raw.set_facecolor('white')
        ax_unr.set_facecolor('#fafafa')
        ax_ref.set_facecolor('#fafafa')
        
        if best_row is None:
            for ax in (ax_raw, ax_unr, ax_ref):
                ax.axis('off')
            ax_raw.set_title(f"{cell_type}\n(No images)", fontsize=9, fontweight='bold')
            continue
        
        base = str(best_row["base"])
        algo = str(best_row.get("algorithm", "unknown"))
        algo_norm = algo.replace(" ", "").replace("_", "")
        ap50_ref = float(best_row.get("AP@0.50", np.nan))
        
        # ========== Row 0 (A): Raw composite image ==========
        dapi, marker = find_raw_channels(base, DATASET_DIR)
        comp = make_composite(dapi, marker)
        
        if comp is not None:
            ax_raw.imshow(comp)
            # Format title: cell type only at top
            ax_raw.text(0.5, 1.04, f"{cell_type}", transform=ax_raw.transAxes,
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax_raw.axis('off')
            ax_raw.text(0.5, 0.5, f"{cell_type}\n(No raw)", transform=ax_raw.transAxes,
                       ha='center', va='center', fontsize=9)
        
        ax_raw.axis('off')
        
        # ========== Load ground truth ==========
        gt_path = find_ground_truth(base, DATASET_DIR)
        if gt_path is None or not gt_path.exists():
            for ax in [ax_unr, ax_ref]:
                ax.axis('off')
            continue
        
        gt = stack_to_label(load_any_image(gt_path))
        
        # ========== Find prediction directories for this algorithm ==========
        refined_dir = None
        unrefined_dir = None
        
        # Match algorithm to directories
        for algo_key in available_refined.keys():
            if algo_key.replace("_", "").lower() in algo_norm.lower():
                refined_dir = available_refined[algo_key]
                break
        
        for algo_key in available_unrefined.keys():
            if algo_key.replace("_", "").lower() in algo_norm.lower():
                unrefined_dir = available_unrefined[algo_key]
                break
        
        # ========== Row 1 (B): Unrefined prediction ==========
        pred_unr_path = find_prediction(base, unrefined_dir)
        if pred_unr_path and pred_unr_path.exists():
            pred_unr = stack_to_label(load_any_image(pred_unr_path))
            
            # Align sizes
            h = min(gt.shape[0], pred_unr.shape[0])
            w = min(gt.shape[1], pred_unr.shape[1])
            gt_crop = gt[:h, :w]
            pred_unr_crop = pred_unr[:h, :w]
            
            # Get unrefined metrics from evaluation data
            ap50_unr = np.nan
            
            # Strategy 1: exact match with "unrefined" suffix
            unref_sub_df = df[
                (df["base"] == base) &
                (df["algorithm"].str.lower().str.contains("unrefined", na=False)) &
                (df["algorithm"].str.lower().str.contains(algo.lower().replace("sam", "").replace("dist", "")[:5], na=False))
            ]
            
            # Strategy 2: match algorithm key name
            if unref_sub_df.empty:
                for algo_key in available_unrefined.keys():
                    algo_key_lower = algo_key.lower()
                    unref_sub_df = df[
                        (df["base"] == base) &
                        (df["algorithm"].str.lower().str.contains(algo_key_lower, na=False)) &
                        (df["algorithm"].str.lower().str.contains("unref", na=False))
                    ]
                    if not unref_sub_df.empty:
                        break
            
            if not unref_sub_df.empty:
                ap50_unr = float(unref_sub_df.iloc[0].get("AP@0.50", np.nan))
            
            # Render
            ax_unr.imshow(label_viz_on_gray(pred_unr_crop, seed_text=base + "_unr"))
            draw_contours(ax_unr, gt_crop > 0, color="#444444", lw=BOUNDARY_W)
            
            # Store text for below panel
            text_unr = f"{algo}"
            if not np.isnan(ap50_unr):
                text_unr += f", Precision@0.5={ap50_unr:.3f}"
            panel_texts[(1, j)] = text_unr
        else:
            ax_unr.axis('off')
        
        ax_unr.axis('off')
        
        # ========== Row 2 (C): Refined prediction ==========
        pred_ref_path = find_prediction(base, refined_dir)
        if pred_ref_path and pred_ref_path.exists():
            pred_ref = stack_to_label(load_any_image(pred_ref_path))
            
            # Align sizes
            h = min(gt.shape[0], pred_ref.shape[0])
            w = min(gt.shape[1], pred_ref.shape[1])
            gt_crop = gt[:h, :w]
            pred_ref_crop = pred_ref[:h, :w]
            
            # Render
            ax_ref.imshow(label_viz_on_gray(pred_ref_crop, seed_text=base + "_ref"))
            draw_contours(ax_ref, gt_crop > 0, color="#444444", lw=BOUNDARY_W)
            
            # Store text for below panel
            text_ref = f"{algo}, Precision@0.5={ap50_ref:.3f}"
            panel_texts[(2, j)] = text_ref
        else:
            ax_ref.axis('off')
        
        ax_ref.axis('off')
    
    # Clean up any remaining ticks
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add text below each panel in rows 1 and 2 (B and C)
    # Split into two lines to avoid overlap
    for j in range(n_cols):
        for i in [1, 2]:  # Only rows B and C
            ax = axes[i, j]
            text = panel_texts[(i, j)]
            if text:
                # Split: algorithm name on line 1, metric on line 2
                if ", Precision" in text:
                    parts = text.split(", Precision")
                    line1 = parts[0]
                    line2 = f"Precision{parts[1]}"
                    ax.text(0.5, -0.05, line1, transform=ax.transAxes,
                           ha='center', va='top', fontsize=7, color='#333333')
                    ax.text(0.5, -0.14, line2, transform=ax.transAxes,
                           ha='center', va='top', fontsize=7, color='#333333')
                else:
                    ax.text(0.5, -0.08, text, transform=ax.transAxes,
                           ha='center', va='top', fontsize=7, color='#333333')
    
    # Add row labels (A, B, C) and descriptive labels on the left side
    row_letters = ['A', 'B', 'C']
    row_names = ['Raw', 'Unrefined', 'Refined']
    for i, (letter, name) in enumerate(zip(row_letters, row_names)):
        ax = axes[i, 0]
        # Letter label (bold) - at top
        ax.text(-0.18, 0.9, letter, transform=ax.transAxes,
               ha='center', va='center', fontsize=11, fontweight='bold')
        # Descriptive label
        ax.text(-0.08, 0.5, name, transform=ax.transAxes,
               ha='left', va='center', fontsize=9, rotation=90)
    
    plt.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(OUT_PDF, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"\n" + "=" * 70)
    print(f"Saved:")
    print(f"  PNG: {OUT_PNG}")
    print(f"  PDF: {OUT_PDF}")
    print("=" * 70)
    
    return fig, axes


if __name__ == "__main__":
    fig, axes = main()
    plt.show()