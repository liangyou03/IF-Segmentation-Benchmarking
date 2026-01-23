#!/bin/bash
# Comprehensive benchmark script for 10 segmentation algorithms
# Measures environment size, package count, and runtime

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEART_DIR="$SCRIPT_DIR"
OUTPUT_CSV="$HEART_DIR/benchmark_results.csv"

# Activate conda
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create CSV header
echo "algorithm,conda_env,env_size_gb,package_count,runtime_seconds,notes" > "$OUTPUT_CSV"

# Array of algorithms: (name env_name)
declare -A ALGORITHMS
ALGORITHMS["InstanSeg"]="instanseg-env"
ALGORITHMS["CellposeSAM"]="cellpose"
ALGORITHMS["Omnipose"]="omnipose"
ALGORITHMS["MicroSAM"]="microsam-cuda"
ALGORITHMS["StarDist"]="stardist"
ALGORITHMS["MESMER"]="deepcell_retinamask"
ALGORITHMS["LACSS"]="lacss"
ALGORITHMS["SplineDist"]="ifimage_splinedist"
ALGORITHMS["CellSAM"]="ifimage_cellsam"
ALGORITHMS["Watershed"]="ifimage_evaluation"

echo "========================================"
echo "ALGORITHM BENCHMARKING"
echo "========================================"

for algo in "${!ALGORITHMS[@]}"; do
    env="${ALGORITHMS[$algo]}"
    echo ""
    echo "----------------------------------------"
    echo "Testing: $algo"
    echo "Environment: $env"
    echo "----------------------------------------"

    # Get environment size
    env_path="$CONDA_BASE/envs/$env"
    if [ -d "$env_path" ]; then
        size_kb=$(du -s "$env_path" 2>/dev/null | awk '{print $1}')
        size_gb=$(echo "scale=2; $size_kb / 1024 / 1024" | bc)
        echo "  Size: ${size_gb} GB"
    else
        size_gb="N/A"
        echo "  Size: N/A (env not found)"
    fi

    # Get package count
    pkg_count=$(conda list -n "$env" 2>/dev/null | wc -l)
    # Subtract header
    pkg_count=$((pkg_count - 1))
    echo "  Packages: $pkg_count"

    # Measure runtime (simple import test)
    echo "  Testing import..."

    case "$algo" in
        "InstanSeg")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from instanseg import InstanSeg; import torch; device='cuda' if torch.cuda.is_available() else 'cpu'; m=InstanSeg('fluorescence_nuclei_and_cells', device=device); print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "CellposeSAM")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from cellpose import models; m=models.CellposeModel(gpu=True); print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "Omnipose")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from cellpose_omni import models; m=models.CellposeModel(gpu=True, model_type='cyto2_omni'); print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "MicroSAM")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from segment_anything import sam_model_registry; from micro_sam import util; print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "StarDist")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from stardist.models import StarDist2D; m=StarDist2D.from_pretrained('2D_versatile_fluo'); print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "MESMER")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from deepcell.applications import Mesmer; print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "LACSS")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); import lacss; print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "SplineDist")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from splinedist.models import SplineDist2D; print(f'{time.time()-start:.3f}')" 2>/dev/null || echo "N/A")
            ;;
        "CellSAM")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from segment_anything import sam_model_registry; print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
        "Watershed")
            runtime=$(conda run -n "$env" python3 -c "import time; start=time.time(); from skimage.filters import threshold_otsu; from skimage.segmentation import watershed; print(f'{time.time()-start:.3f}')" 2>/dev/null)
            ;;
    esac

    # Clean up runtime output (extract just the number)
    if echo "$runtime" | grep -qE "^[0-9]+\.[0-9]+$"; then
        echo "  Import time: ${runtime}s"
        notes="OK"
    else
        runtime="N/A"
        echo "  Import time: N/A"
        notes="Import failed or timed out"
    fi

    # Append to CSV
    echo "$algo,$env,$size_gb,$pkg_count,$runtime,$notes" >> "$OUTPUT_CSV"
done

echo ""
echo "========================================"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
cat "$OUTPUT_CSV"
