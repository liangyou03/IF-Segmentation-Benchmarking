from pathlib import Path

from _paths import HEART_DATA_ROOT


def check_missing_files(base_dir):
    """Check data completeness for all regions."""
    base_path = Path(base_dir)
    regions = ['LA', 'RA', 'LV', 'RV', 'SEP']
    cell_types = ['Epi', 'Immune', 'Mural']
    
    missing_files = []
    incomplete_areas = []
    
    print("=" * 60)
    print("🔍 Checking data completeness")
    print("=" * 60)
    
    for region in regions:
        region_dir = base_path / region
        if not region_dir.exists():
            print(f"\n❌ {region}: Directory not found")
            continue
        
        # Get all TIF files
        tif_files = sorted(region_dir.glob('*.tif'))
        print(f"\n📍 {region}: {len(tif_files)} areas")
        
        for tif_file in tif_files:
            area_name = tif_file.stem
            expected_zips = []
            found_zips = []
            missing_zips = []
            
            for cell_type in cell_types:
                zip_file = region_dir / f"{cell_type}-{area_name}.zip"
                expected_zips.append(f"{cell_type}-{area_name}.zip")
                
                if zip_file.exists():
                    found_zips.append(cell_type)
                else:
                    missing_zips.append(cell_type)
                    missing_files.append({
                        'region': region,
                        'area': area_name,
                        'cell_type': cell_type,
                        'expected_file': str(zip_file)
                    })
            
            if missing_zips:
                status = f"⚠️  {area_name}: Missing {', '.join(missing_zips)}"
                incomplete_areas.append(f"{region}/{area_name}")
            else:
                status = f"✅ {area_name}: Complete (Epi, Immune, Mural)"
            
            print(f"  {status}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    if missing_files:
        print(f"\n⚠️  Found {len(missing_files)} missing annotation files:")
        for mf in missing_files:
            print(f"  • {mf['region']}/{mf['area']} - {mf['cell_type']}")
    else:
        print("\n✅ All annotation files are present!")
    
    if incomplete_areas:
        print(f"\n⚠️  {len(incomplete_areas)} areas with incomplete annotations:")
        for area in incomplete_areas:
            print(f"  • {area}")
    
    return missing_files, incomplete_areas

if __name__ == "__main__":
    base_dir = HEART_DATA_ROOT / "raw"
    missing, incomplete = check_missing_files(base_dir)
