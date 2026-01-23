"""
Inspect NeuN file naming patterns.
"""
from pathlib import Path

from _paths import RAW_DATA_ROOT

base = RAW_DATA_ROOT / 'NeuN' / 'NeuN'

# Collect all -1040.tiff files
all_files = list(base.glob('**/*-1040.tiff'))
print(f"Total files ending with -1040.tiff: {len(all_files)}")

# Check naming patterns
has_c0 = [f for f in all_files if 'c0' in f.name.lower()]
has_c1 = [f for f in all_files if 'c1' in f.name.lower()]
has_dapi = [f for f in all_files if 'dapi' in f.name.lower()]
has_neun = [f for f in all_files if 'neun' in f.name.lower()]

print(f"\nFile naming patterns:")
print(f"  Contains 'c0': {len(has_c0)}")
print(f"  Contains 'c1': {len(has_c1)}")
print(f"  Contains 'dapi': {len(has_dapi)}")
print(f"  Contains 'neun': {len(has_neun)}")

# Display sample filenames
print(f"\n=== Sample filenames (first 5) ===")
for f in all_files[:5]:
    print(f"  {f.name}")

# Check if white region files are present (should be excluded)
has_white = [f for f in all_files if 'white' in f.name.lower()]
print(f"\n  Contains 'white' (should exclude): {len(has_white)}")
