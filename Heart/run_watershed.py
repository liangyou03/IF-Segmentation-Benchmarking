# heart/run_watershed.py
"""
Watershed inference script for all channels.
Environment: ifimage
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

from _paths import HEART_DATA_ROOT

PROCESSED_DIR = HEART_DATA_ROOT / "processed"
OUTPUT_DIR = HEART_DATA_ROOT / "benchmark_results" / "watershed_predictions"

print("=" * 70)
print("🚀 Watershed Segmentation")
print("=" * 70)

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="Watershed", unit="img"):
    image = tifffile.imread(tif_path)
    
    # Watershed segmentation
    thresh = threshold_otsu(image)
    binary = image > thresh
    distance = ndi.distance_transform_edt(binary)
    local_max = peak_local_max(distance, min_distance=10, labels=binary)
    markers = np.zeros_like(image, dtype=int)
    markers[tuple(local_max.T)] = np.arange(len(local_max)) + 1
    markers = ndi.label(markers)[0]
    labels = watershed(-distance, markers, mask=binary)
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
