from pathlib import Path


def get_repo_root() -> Path:
    """Traverse upward until the repository root containing .git is found."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root (missing .git directory)")


REPO_ROOT = get_repo_root()
DATA_ROOT = REPO_ROOT / "Data"
IFIMAGE_ROOT = DATA_ROOT / "ifimage"
FULLDATA_ROOT = IFIMAGE_ROOT / "11_fulldata"
RAW_DATA_ROOT = FULLDATA_ROOT / "raw"
SEGMENTATION_ROOT = FULLDATA_ROOT / "segmentation"
SEGMENTATION_REFINED_ROOT = FULLDATA_ROOT / "segmentation_refined"
SEGMENTATION_WATERSHED_ROOT = FULLDATA_ROOT / "segmentation_watershed"
SEGMENTATION_WATERSHED_REFINED_ROOT = FULLDATA_ROOT / "segmentation_watershed_refined"
