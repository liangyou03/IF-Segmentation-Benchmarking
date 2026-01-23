import numpy as np
import os
from collections import defaultdict

data_dir = "/ihome/jbwang/liy121/ifimage/00_dataset_withoutpecam"

files = os.listdir(data_dir)

# === NUCLEI (dapimultimask) ===
print("="*50)
print("NUCLEI (DAPI) ANNOTATIONS:")
print("="*50)

dapi_files = [f for f in files if f.endswith('_dapimultimask.npy')]
total_nuclei = 0

for f in sorted(dapi_files):
    mask = np.load(os.path.join(data_dir, f))
    unique_ids = np.unique(mask)
    n_instances = len(unique_ids[unique_ids > 0])
    total_nuclei += n_instances
    print(f"{f}: {n_instances} nuclei")

print(f"\nNUCLEI TOTAL: {len(dapi_files)} images, {total_nuclei} nuclei instances")

# === WHOLE-CELL (cellbodies) ===
print("\n" + "="*50)
print("WHOLE-CELL ANNOTATIONS:")
print("="*50)

cellbody_files = [f for f in files if f.endswith('_cellbodies.npy')]
marker_counts = defaultdict(lambda: {'files': 0, 'instances': 0})

for f in sorted(cellbody_files):
    marker = f.split('_')[0]
    mask = np.load(os.path.join(data_dir, f))
    unique_ids = np.unique(mask)
    n_instances = len(unique_ids[unique_ids > 0])
    marker_counts[marker]['files'] += 1
    marker_counts[marker]['instances'] += n_instances
    print(f"{f}: {n_instances} instances")

print("\n" + "="*50)
print("SUMMARY:")
print("="*50)

total_files = 0
total_cells = 0

for marker in ['gfap', 'iba1', 'neun', 'olig2']:
    info = marker_counts[marker]
    print(f"{marker.upper()}: {info['files']} images, {info['instances']} instances")
    total_files += info['files']
    total_cells += info['instances']

print("="*50)
print(f"WHOLE-CELL TOTAL: {total_files} images, {total_cells} instances")
print(f"NUCLEI TOTAL: {len(dapi_files)} images, {total_nuclei} instances")
print("="*50)
print(f"\nFor your paper:")
print(f"  - Total nuclei instances: {total_nuclei}")
print(f"  - Total whole-cell instances: {total_cells}")
print(f"  - OLIG2/NeuN/IBA1/GFAP: {marker_counts['olig2']['instances']}/{marker_counts['neun']['instances']}/{marker_counts['iba1']['instances']}/{marker_counts['gfap']['instances']}")