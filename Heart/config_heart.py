# heart/config_heart.py
"""Heart dataset configuration shared across scripts."""
from pathlib import Path

from _paths import HEART_DATA_ROOT


class HeartConfig:
    # Path configuration
    RAW_DIR = HEART_DATA_ROOT / 'raw'
    GT_DIR = HEART_DATA_ROOT / 'ground_truth_masks'
    OUTPUT_BASE = HEART_DATA_ROOT / 'benchmark_results'
    
    # Dataset structure
    REGIONS = ['LA', 'RA', 'LV', 'RV', 'SEP']
    CELL_TYPES = ['Epi', 'Immune', 'Mural']
    
    # File paths
    MAPPING_FILE = GT_DIR / 'file_mapping.csv'
    
    @staticmethod
    def get_algo_dir(algo_name):
        """Return the directory for a specific algorithm's outputs."""
        return HeartConfig.OUTPUT_BASE / f"{algo_name}_predictions"
