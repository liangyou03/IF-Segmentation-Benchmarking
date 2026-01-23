# heart/run_mesmer.py
"""
Mesmer inference script for all channels.
Environment: deepcell_retinamask
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from deepcell.applications import Mesmer

from _paths import HEART_DATA_ROOT

PROCESSED_DIR = HEART_DATA_ROOT / "processed"
OUTPUT_DIR = HEART_DATA_ROOT / "benchmark_results" / "mesmer_predictions"

print("=" * 70)
print("🚀 Mesmer Segmentation")
print("=" * 70)

# Load Mesmer model
print("Loading Mesmer model...")
app = Mesmer()

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="Mesmer", unit="img"):
    image = tifffile.imread(tif_path)
    
    # Mesmer expects 4D input [batch, height, width, channels]
    # and two channels [nuclear, cytoplasm], so reuse the same image
    if image.ndim == 2:
        # Create a two-channel image
        image_2ch = np.stack([image, image], axis=-1)  # (H, W, 2)
    image_4d = np.expand_dims(image_2ch, axis=0)      # (1, H, W, 2)
    
    # Mesmer segmentation (returns nuclear and whole-cell masks)
    predictions = app.predict(image_4d, image_mpp=0.5)
    
    # Use the nuclear mask (first channel)
    mask = predictions[0, ..., 0]
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
