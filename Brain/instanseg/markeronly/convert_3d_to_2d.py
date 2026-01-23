#!/usr/bin/env python3
"""
convert_3d_to_2d.py — Convert 3D masks to 2D masks.
将InstanSeg输出的3D mask转换为2D mask。

Usage:
    python convert_3d_to_2d.py
"""

from pathlib import Path
import numpy as np

# ---- config ----
# 要转换的目录列表
DIRS_TO_CONVERT = [
    Path("/ihome/jbwang/liy121/ifimage/013_ins/nuclei_prediction"),
    Path("/ihome/jbwang/liy121/ifimage/013_ins/cyto_prediction"),
    Path("/ihome/jbwang/liy121/ifimage/013_ins/markeronly"),
]


def convert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert 3D mask to 2D.
    - If (C, H, W): take channel 0 for nuclei, channel 1 for cells if available
    - If (H, W, C): take last channel
    - If already 2D: return as-is
    """
    if mask.ndim == 2:
        return mask.astype(np.int32)
    
    if mask.ndim == 3:
        # Determine if (C, H, W) or (H, W, C)
        if mask.shape[0] <= 4:  # likely (C, H, W)
            # For cells, prefer channel 1 if exists; otherwise channel 0
            if mask.shape[0] >= 2:
                out = mask[1]  # cells channel
            else:
                out = mask[0]
        else:  # likely (H, W, C)
            out = mask[:, :, -1]
        
        return np.squeeze(out).astype(np.int32)
    
    raise ValueError(f"Unexpected mask shape: {mask.shape}")


def process_directory(dir_path: Path) -> tuple[int, int, int]:
    """
    Process all .npy files in a directory.
    Returns (converted, skipped, failed) counts.
    """
    if not dir_path.exists():
        print(f"[SKIP] Directory not found: {dir_path}")
        return 0, 0, 0
    
    npy_files = list(dir_path.glob("*.npy"))
    if not npy_files:
        print(f"[SKIP] No .npy files in: {dir_path}")
        return 0, 0, 0
    
    converted, skipped, failed = 0, 0, 0
    
    for f in npy_files:
        try:
            mask = np.load(f)
            
            if mask.ndim == 2:
                skipped += 1
                continue
            
            # Convert and save
            mask_2d = convert_mask(mask)
            np.save(f, mask_2d)
            converted += 1
            print(f"  [OK] {f.name}: {mask.shape} -> {mask_2d.shape}")
            
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {f.name}: {e}")
    
    return converted, skipped, failed


def main():
    print("=" * 60)
    print("== Converting 3D masks to 2D ==")
    print("=" * 60)
    
    total_converted, total_skipped, total_failed = 0, 0, 0
    
    for dir_path in DIRS_TO_CONVERT:
        print(f"\nProcessing: {dir_path}")
        c, s, f = process_directory(dir_path)
        total_converted += c
        total_skipped += s
        total_failed += f
        print(f"  -> converted={c}, already_2d={s}, failed={f}")
    
    print("\n" + "=" * 60)
    print(f"Total: converted={total_converted}, already_2d={total_skipped}, failed={total_failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
