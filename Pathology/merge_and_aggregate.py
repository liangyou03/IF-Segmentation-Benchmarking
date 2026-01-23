"""
Merge the original 40 NeuN participants with 27 supplemental donors for 67 total.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from _paths import SEGMENTATION_REFINED_ROOT, FULLDATA_ROOT

seg_refined = SEGMENTATION_REFINED_ROOT
clinical_file = FULLDATA_ROOT / 'ROSMAP_clinical_n69.csv'

print("="*60)
print("Merging: Keep existing 40 + Add supplemented 27 NeuN")
print("="*60)

# 1. Load existing file_mapping_refined.csv
df_existing = pd.read_csv(seg_refined / 'file_mapping_refined.csv')
print(f"\n1. Existing refined mapping: {len(df_existing)} rows")

# 2. Load supplemented NeuN
df_neun_supp = pd.read_csv(seg_refined / 'file_mapping_neun_supplement_refined.csv')
print(f"2. Supplemented NeuN: {len(df_neun_supp)} rows")

# 3. Inspect NeuN participants
existing_neun = df_existing[df_existing['marker'] == 'NeuN'].copy()
print(f"\n3. Existing NeuN data:")
print(f"   Rows: {len(existing_neun)}")
print(f"   Participants: {existing_neun['participant_id'].nunique()}")

supp_neun_pids = set(df_neun_supp['participant_id'].astype(str))
existing_neun_pids = set(existing_neun['participant_id'].astype(str))

print(f"\n4. Supplemented NeuN participants: {len(supp_neun_pids)}")
print(f"   Participants: {sorted(list(supp_neun_pids))[:5]}...")

# Check for overlap
overlap = existing_neun_pids & supp_neun_pids
print(f"\n5. Overlap check:")
print(f"   Participants in both: {len(overlap)}")
if overlap:
    print(f"   Overlapping IDs: {sorted(list(overlap))[:5]}...")
    print("   ⚠️  Will keep existing data for overlapping participants")

# 6. Merge strategy:
# - Keep all existing data (including original NeuN)
# - Only add new participants from the supplemented NeuN set
df_neun_new_only = df_neun_supp[~df_neun_supp['participant_id'].isin(existing_neun_pids)]

print(f"\n6. New NeuN participants only: {len(df_neun_new_only)} rows")
print(f"   New participants: {df_neun_new_only['participant_id'].nunique()}")

# 7. Combine
common_cols = ['marker', 'participant_id', 'sample_name', 'out_dir', 
               'n_total_cells', 'n_marker_positive', 'marker_positive_ratio_refined']

df_combined = pd.concat([
    df_existing[common_cols],
    df_neun_new_only[common_cols]
], ignore_index=True)

print(f"\n7. Combined data: {len(df_combined)} rows")

# Check NeuN totals
neun_combined = df_combined[df_combined['marker'] == 'NeuN']
print(f"\n✓ Total NeuN data:")
print(f"   Rows: {len(neun_combined)}")
print(f"   Participants: {neun_combined['participant_id'].nunique()}")
print(f"   Expected: 67 (40 existing + 27 new)")

# Save
df_combined.to_csv(seg_refined / 'file_mapping_refined_complete.csv', index=False)
print(f"\n✓ Saved: file_mapping_refined_complete.csv")

# ============================================================
# Aggregate
# ============================================================
print("\n" + "="*60)
print("Aggregating at donor level...")
print("="*60)

df_clinical = pd.read_csv(clinical_file)
df_clinical['projid'] = df_clinical['projid'].astype(str)

df_combined['projid'] = df_combined['participant_id'].astype(str)

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']
agg_results = []

for marker in markers:
    df_marker = df_combined[df_combined['marker'] == marker].copy()
    
    if len(df_marker) == 0:
        continue
    
    grouped = df_marker.groupby('projid').agg({
        'sample_name': 'count',
        'n_total_cells': 'sum',
        'n_marker_positive': 'sum',
        'marker_positive_ratio_refined': 'mean'
    }).reset_index()
    
    grouped.columns = [
        'projid',
        f'{marker}_n_samples',
        f'{marker}_n_total_cells_sum',
        f'{marker}_n_marker_positive_sum',
        f'{marker}_marker_positive_ratio_refined_mean'
    ]
    
    grouped[f'{marker}_ratio_from_sum'] = (
        grouped[f'{marker}_n_marker_positive_sum'] / 
        grouped[f'{marker}_n_total_cells_sum']
    )
    
    agg_results.append(grouped)
    
    print(f"\n{marker}:")
    print(f"  Participants: {len(grouped)}")
    print(f"  Avg samples: {grouped[f'{marker}_n_samples'].mean():.1f}")
    print(f"  Avg ratio: {grouped[f'{marker}_ratio_from_sum'].mean():.3f}")

# Merge
df_donor = agg_results[0]
for df in agg_results[1:]:
    df_donor = df_donor.merge(df, on='projid', how='outer')

df_final = df_clinical.merge(df_donor, on='projid', how='left')

# Save
output_file = seg_refined / 'donor_level_aggregation_refined.csv'
df_final.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"✓ FINAL: {output_file}")
print(f"  Total donors: {len(df_final)}")
print("="*60)

# Summary
print("\nFinal summary:")
for marker in markers:
    col = f'{marker}_n_samples'
    if col in df_final.columns:
        n = df_final[col].notna().sum()
        avg_ratio = df_final[f'{marker}_ratio_from_sum'].mean()
        print(f"  {marker}: {n} donors (expected: 69 for GFAP/iba1/Olig2/PECAM, 67 for NeuN)")
        print(f"           avg ratio={avg_ratio:.3f}")

# Verify NeuN reaches the expected 67 donors
neun_count = df_final['NeuN_n_samples'].notna().sum()
if neun_count < 67:
    print(f"\n⚠️  NeuN only has {neun_count} donors, expected 67!")
    print("   Checking which donors are missing...")
    
    all_neun_pids = set(neun_combined['participant_id'].astype(str))
    clinical_pids = set(df_clinical['projid'].astype(str))
    
    # Participants that exist in clinical data but lack NeuN measurements
    missing_in_neun = clinical_pids - all_neun_pids
    print(f"   Donors in clinical but no NeuN data: {len(missing_in_neun)}")
else:
    print(f"\n✓ NeuN has {neun_count} donors - Perfect!")

print("="*60)
