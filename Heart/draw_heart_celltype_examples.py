#!/usr/bin/env python3
"""
draw_heart_celltype_examples.py

Generates example annotations for heart cell types (HORIZONTAL LAYOUT).
Clean version: no labels inside images, only scale bar and labels at bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import find_contours
from pathlib import Path

from _paths import HEART_DATA_ROOT

# ============================================================================
# Configuration
# ============================================================================

PROCESSED_DIR = HEART_DATA_ROOT / "processed"
GT_MASKS_DIR = HEART_DATA_ROOT / "ground_truth_masks"
PLOTS_DIR = HEART_DATA_ROOT / "plots"

# Cell types: (key, display_name, marker_name, color, marker_file)
CELLTYPES = [
    ("Epi", "Epicardial", "ALDH1A2", "#E64B35", "aldh1a2"),
    ("Immune", "Immune", "CD45", "#4DBBD5", "cd45"),
    ("Mural", "Mural", "PDGFRB", "#00A087", "pdgfrb"),
]

# Example samples (region, sample_id) - using LA1 for all
EXAMPLE_SAMPLES = {
    "Epi": ("LA", "LA1"),
    "Immune": ("LA", "LA1"),
    "Mural": ("LA", "LA1"),
}

# Visualization parameters
DPI = 300
CROP_HEIGHT = 450
CROP_WIDTH = 150
CONTOUR_WIDTH = 2.0
CONTOUR_COLOR = "#FFFFFF"

# Scale bar: ADJUST THIS BASED ON YOUR PIXEL SIZE
# Assuming 0.5 µm/pixel (change if different)
PIXEL_SIZE_UM = 0.5  # µm per pixel - CHANGE THIS VALUE!
SCALE_BAR_UM = 25    # Scale bar length in µm

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
})

# ============================================================================
# Helper Functions
# ============================================================================

def load_image(path):
    """Load image from tif/tiff."""
    if path and path.exists():
        return np.array(Image.open(path))
    return None


def normalize_image(img):
    """Normalize image to 0-1 using percentiles."""
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
        rgb[..., 2] = d  # Blue channel
    if m is not None:
        rgb[..., 0] = m  # Red channel
        rgb[..., 1] = m * 0.2

    return rgb


def draw_contours(ax, mask, color=CONTOUR_COLOR, lw=CONTOUR_WIDTH):
    """Draw contours of label mask."""
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        binary_mask = (mask == cell_id).astype(float)
        for cnt in find_contours(binary_mask, 0.5):
            ax.plot(cnt[:, 1], cnt[:, 0], color=color, linewidth=lw,
                    solid_capstyle="round", solid_joinstyle="round")


def find_best_crop(mask, target_height=450, target_width=150):
    """Find the crop region with MAXIMUM cell density."""
    if not np.any(mask > 0):
        return None

    best_count = 0
    best_crop = None
    step_h = max(20, target_height // 4)
    step_w = max(20, target_width // 4)

    for row_start in range(0, mask.shape[0] - target_height + 1, step_h):
        for col_start in range(0, mask.shape[1] - target_width + 1, step_w):
            row_end = row_start + target_height
            col_end = col_start + target_width
            crop_mask = mask[row_start:row_end, col_start:col_end]
            cell_count = len(np.unique(crop_mask)) - 1
            if cell_count > best_count:
                best_count = cell_count
                best_crop = (slice(row_start, row_end), slice(col_start, col_end))

    if best_crop is None:
        best_crop = (slice(0, min(target_height, mask.shape[0])),
                    slice(0, min(target_width, mask.shape[1])))

    return best_crop


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Heart Cell Type Annotation Examples (HORIZONTAL, CLEAN)")
    print("=" * 70)
    print(f"\nNOTE: Using {PIXEL_SIZE_UM} µm/pixel for scale bar.")
    print(f"      Change PIXEL_SIZE_UM in script if this is incorrect!")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create figure: SQUARE (10x10 inches)
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    fig.patch.set_facecolor('white')

    # Transpose Epi
    TRANSPOSE_TYPES = ["Epi"]

    for idx, (cell_type, display_name, marker_name, color, marker_file) in enumerate(CELLTYPES):
        ax = axes[idx]

        region, sample_id = EXAMPLE_SAMPLES[cell_type]

        # Load images
        dapi_path = PROCESSED_DIR / region / f"{sample_id}_dapi.tif"
        marker_path = PROCESSED_DIR / region / f"{sample_id}_{marker_file}.tif"
        mask_path = GT_MASKS_DIR / region / f"{cell_type}-{sample_id}_mask.npy"

        print(f"\n{cell_type}:")
        print(f"  DAPI: {dapi_path.exists()}")
        print(f"  Marker ({marker_file}): {marker_path.exists()}")
        print(f"  Mask: {mask_path.exists()}")

        dapi = load_image(dapi_path)
        marker = load_image(marker_path)
        mask = np.load(mask_path) if mask_path.exists() else None

        if mask is None or not np.any(mask > 0):
            ax.text(0.5, 0.5, f"{display_name}\n(No data)",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        # Transpose for Epi
        if cell_type in TRANSPOSE_TYPES:
            dapi = dapi.T if dapi is not None else None
            marker = marker.T if marker is not None else None
            mask = mask.T
            print(f"  Transposed: {mask.shape}")

        # Find best crop
        best_crop = find_best_crop(mask, CROP_HEIGHT, CROP_WIDTH)

        # Crop images
        dapi_crop = dapi[best_crop] if dapi is not None else None
        marker_crop = marker[best_crop] if marker is not None else None
        mask_crop = mask[best_crop]

        n_cells = len(np.unique(mask_crop)) - 1
        print(f"  Crop shape: {mask_crop.shape}")
        print(f"  Cells in crop: {n_cells}")

        # Create composite
        comp = make_composite(dapi_crop, marker_crop)

        if comp is not None:
            ax.imshow(comp)
            draw_contours(ax, mask_crop, color=CONTOUR_COLOR, lw=CONTOUR_WIDTH)

        ax.axis('off')

    # Add labels at the bottom (outside the images) - plain sans-serif, no color
    cell_labels = [
        "Epicardial (ALDH1A2)",
        "Immune (CD45)",
        "Mural (PDGFRB)",
    ]

    # Add cell type labels at the bottom of the figure - moved up, larger font
    for idx, label in enumerate(cell_labels):
        fig.text(0.165 + idx * 0.33, 0.06, label,
                ha='center', va='bottom', fontsize=16, fontweight='normal',
                color='black')

    # Adjust layout - more room at bottom
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.08,
                        wspace=0.03, hspace=0.02)

    # Save
    out_png = PLOTS_DIR / "heart_celltype_examples_horizontal.png"
    out_pdf = PLOTS_DIR / "heart_celltype_examples_horizontal.pdf"

    fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')

    print(f"\n" + "=" * 70)
    print(f"Saved:")
    print(f"  PNG: {out_png}")
    print(f"  PDF: {out_pdf}")
    print("=" * 70)

    return fig, axes


if __name__ == "__main__":
    fig, axes = main()
    plt.show()
