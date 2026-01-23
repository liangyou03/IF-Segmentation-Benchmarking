#!/usr/bin/env python3
"""
prediction_markeronly_instanseg.py — CELL segmentation with InstanSeg (marker only).
Minimal script: import data utils, run model, save masks.

InstanSeg: https://github.com/instanseg/instanseg
pip install instanseg-torch
"""

from pathlib import Path
import numpy as np
import torch

from instanseg import InstanSeg

# Import data utils (adjust path as needed)
import sys
sys.path.insert(0, "/ihome/jbwang/liy121/ifimage/00_dataset_withoutpecam")
from utils import SampleDataset, ensure_dir

# ---- config ----
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset_withoutpecam")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/013_ins/markeronly")

# InstanSeg model for fluorescence
MODEL_NAME = "fluorescence_nuclei_and_cells"
PIXEL_SIZE = 0.5  # adjust based on your data, or set to None for auto


def get_device() -> str:
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_instanseg_markeronly(model: InstanSeg, img: np.ndarray, pixel_size: float = None) -> np.ndarray:
    """
    Run InstanSeg on a single-channel marker image for cell segmentation.
    
    Args:
        model: InstanSeg model instance
        img: 2D grayscale image (H, W) - marker channel only
        pixel_size: pixel size in microns, None for auto
        
    Returns:
        int32 label mask (H, W)
    """
    # Ensure correct shape: InstanSeg expects (H, W, C)
    if img.ndim == 2:
        img_input = img[:, :, np.newaxis]  # (H, W, 1)
    else:
        img_input = img
    
    # Run inference - use "cells" target for marker-only
    # InstanSeg is channel-invariant, so single channel works
    labeled_output, _ = model.eval_small_image(
        img_input, 
        pixel_size=pixel_size,
        target="cells",
        cleanup_fragments=True,
    )
    
    # labeled_output is Tensor - convert to numpy first
    if hasattr(labeled_output, 'cpu'):
        labeled_output = labeled_output.cpu().numpy()
    
    # Extract cell mask
    if labeled_output.ndim == 3 and labeled_output.shape[0] >= 2:
        mask = labeled_output[1]  # cells channel
    elif labeled_output.ndim == 3:
        mask = labeled_output[0]
    else:
        mask = labeled_output
        
    return mask.astype(np.int32)


def main():
    print("=" * 60)
    print("== Cell prediction with InstanSeg (Marker only) ==")
    print("=" * 60)
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker optional).")

    device = get_device()
    print(f"Device: {device}")
    
    # Load InstanSeg model
    print(f"Loading InstanSeg model: {MODEL_NAME}")
    model = InstanSeg(MODEL_NAME, device=device, verbosity=1)
    print("Model loaded successfully.")

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()  # loads/normalizes images
            
            if s.cell_chan is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (no marker)")
                continue
            
            # Use marker channel only
            marker = s.cell_chan  # (H, W)
            
            # Run InstanSeg
            mask = run_instanseg_markeronly(model, marker, pixel_size=PIXEL_SIZE)
            s.predicted_cell = mask
            
            # Save result
            outp = OUTPUT_DIR / f"{s.base}_pred_marker_only.npy"
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
            
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print("=" * 60)
    print(f"Done: marker_ok={n_ok}, marker_skip={n_skip}, total={len(ds)}")


if __name__ == "__main__":
    main()