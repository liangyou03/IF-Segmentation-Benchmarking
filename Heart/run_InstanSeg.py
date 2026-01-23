# heart/run_InstanSeg.py
"""
InstanSeg inference script for all channels.
Environment: ifimage
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
import torch
from instanseg import InstanSeg

from _paths import HEART_DATA_ROOT

PROCESSED_DIR = HEART_DATA_ROOT / "processed"
OUTPUT_DIR = HEART_DATA_ROOT / "benchmark_results" / "instanseg_predictions"

print("=" * 70)
print("🚀 InstanSeg Segmentation")
print("=" * 70)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = InstanSeg("fluorescence_nuclei_and_cells", device=device)
print(f"Device: {device}")

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="InstanSeg", unit="img"):
    image = tifffile.imread(tif_path)
    
    # InstanSeg segmentation (output shape: (2, H, W) -> nuclei, cells)
    labeled_output, _ = model.eval_small_image(image)
    labels = labeled_output[0][1].cpu().numpy().astype(np.int32)
    labels = np.clip(labels, 0, None)
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
