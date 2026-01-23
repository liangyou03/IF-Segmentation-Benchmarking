# heart/check_raw_structure.py
"""
Inspect the structure of the raw dataset.
"""
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config_heart import HeartConfig

def check_all_images():
    config = HeartConfig()
    
    print("=" * 70)
    print("🔍 Checking Raw Data Structure")
    print("=" * 70)
    
    # Load mapping
    mapping_df = pd.read_csv(config.MAPPING_FILE)
    
    # Enumerate all unique images
    unique_images = mapping_df['image_absolute_path'].unique()
    
    print(f"\n📊 Checking {len(unique_images)} images...\n")
    
    for i, image_path in enumerate(unique_images[:5], 1):  # Check only first 5
        image_path = Path(image_path)
        print(f"{i}. {image_path.name}")
        
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            print(f"   PIL mode: {img.mode}")
            print(f"   PIL size: {img.size} (width × height)")
            print(f"   Array shape: {img_array.shape}")
            print(f"   Array dtype: {img_array.dtype}")
            print(f"   Value range: [{img_array.min()}, {img_array.max()}]")
            
            # Report additional bands/layers if available
            if hasattr(img, 'n_frames'):
                print(f"   N frames: {img.n_frames}")
            
            # Attempt to read all TIF pages
            if image_path.suffix.lower() in ['.tif', '.tiff']:
                try:
                    from tifffile import TiffFile
                    with TiffFile(image_path) as tif:
                        print(f"   TIF pages: {len(tif.pages)}")
                        print(f"   TIF series: {len(tif.series)}")
                        
                        # Inspect the first series
                        if len(tif.series) > 0:
                            series = tif.series[0]
                            print(f"   Series shape: {series.shape}")
                            print(f"   Series dtype: {series.dtype}")
                            
                            # Load the full array
                            full_data = series.asarray()
                            print(f"   Full data shape: {full_data.shape}")
                            
                except ImportError:
                    print("   (tifffile not available, install with: pip install tifffile)")
                except Exception as e:
                    print(f"   TIF read error: {e}")
            
            print()
            
        except Exception as e:
            print(f"   ✗ Error: {e}\n")
    
    print("\n" + "=" * 70)
    print("💡 Analysis:")
    print("=" * 70)
    
    # Inspect raw directory structure
    print(f"\n📂 Raw directory structure:")
    raw_dir = config.RAW_DIR
    
    for region in config.REGIONS:
        region_dir = raw_dir / region
        if region_dir.exists():
            tif_files = list(region_dir.glob('*.tif')) + list(region_dir.glob('*.tiff'))
            print(f"\n  {region}/")
            print(f"    TIF files: {len(tif_files)}")
            
            if tif_files:
                sample_file = tif_files[0]
                print(f"    Sample: {sample_file.name}")
                
                # Check file size
                file_size_mb = sample_file.stat().st_size / (1024 * 1024)
                print(f"    Size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    check_all_images()
