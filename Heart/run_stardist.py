# heart/run_stardist.py
"""
StarDist inference script for all channels.
Environment: ifimage_stardist
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from stardist.models import StarDist2D
from csbdeep.utils import normalize

from _paths import HEART_DATA_ROOT

PROCESSED_DIR = HEART_DATA_ROOT / "processed"
OUTPUT_DIR = HEART_DATA_ROOT / "benchmark_results" / "stardist_predictions"

print("=" * 70)
print("🚀 StarDist Segmentation")
print("=" * 70)

# Load StarDist model
print("Loading StarDist model (2D_versatile_fluo)...")
model = StarDist2D.from_pretrained('2D_versatile_fluo')

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="StarDist", unit="img"):
    image = tifffile.imread(tif_path)
    
    # Normalize (recommended by StarDist)
    image_norm = normalize(image, 1, 99.8)
    
    # StarDist inference
    labels, _ = model.predict_instances(
        image_norm,
        prob_thresh=0.5,
        nms_thresh=0.4
    )
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
