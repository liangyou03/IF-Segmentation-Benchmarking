# heart/run_lacss.py
"""
LACSS inference script for all channels.
Environment: lacss
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from lacss.deploy import model_urls
from lacss.deploy.predict import Predictor

from _paths import HEART_DATA_ROOT

PROCESSED_DIR = HEART_DATA_ROOT / "processed"
OUTPUT_DIR = HEART_DATA_ROOT / "benchmark_results" / "lacss_predictions"

print("=" * 70)
print("🚀 LACSS Segmentation")
print("=" * 70)

# Load the LACSS model
print("Loading LACSS model...")
predictor = Predictor(model_urls["default"])

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="LACSS", unit="img"):
    image = tifffile.imread(tif_path)
    
    # Ensure 2D grayscale images
    if image.ndim == 3:
        image = image[..., 0]
    
    # LACSS expects [H, W, 1]
    img_3d = image[..., None]
    
    try:
        # LACSS inference
        out = predictor.predict(img_3d, output_type="label")
        mask = out["pred_label"].astype(np.int32, copy=False)
        
        region = tif_path.parent.name
        filename = tif_path.stem
        output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mask)
        
    except Exception as e:
        print(f"\n  ✗ Failed {tif_path.name}: {e}")
        continue

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
