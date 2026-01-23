#!/bin/bash
# Environment Size Check Script - Uses mamba for faster operations

set -euo pipefail

OUTPUT_DIR="/ix/jbwang/liangyou/ifimage/algorithm_environment_analysis"
OUTPUT_FILE="$OUTPUT_DIR/environment_sizes.txt"

echo "========================================"
echo "Conda Environment Size Analysis"
echo "========================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Initialize conda/mamba
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
fi

# Check for mamba
if ! command -v mamba &> /dev/null; then
  echo "Error: mamba not available"
  exit 1
fi

# Find environments directory
CONDA_ENVS_DIR="$HOME/miniconda3/envs"
[ ! -d "$CONDA_ENVS_DIR" ] && CONDA_ENVS_DIR="$HOME/micromamba/envs"

if [ ! -d "$CONDA_ENVS_DIR" ]; then
  echo "Error: Cannot find conda environments directory"
  exit 1
fi

echo "Environments directory: $CONDA_ENVS_DIR"
echo ""

# Initialize output
echo "Analysis run at: $(date)" > "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Environments to check
ENVIRONMENTS=(
  "instanseg-env" "stardist" "ifimage_cellsam"
  "deepcell_retinamask" "ifimage_splinedist" "cellpose" "lacss" "omnipose"
  "microsam-cuda" "ifimage_evaluation"
)

# Check each environment
echo "Environment Sizes:" | tee -a "$OUTPUT_FILE"
echo "--------------------" | tee -a "$OUTPUT_FILE"

for env_name in "${ENVIRONMENTS[@]}"; do
  env_path="$CONDA_ENVS_DIR/$env_name"
  
  if [ -d "$env_path" ]; then
    size_human=$(du -sh "$env_path" 2>/dev/null | cut -f1)
    package_list=$(mamba run -n "$env_name" mamba list 2>/dev/null || echo "")
    n_packages=$(echo "$package_list" | tail -n +3 | wc -l)
    
    printf "%-25s %10s %15d packages\n" "$env_name" "$size_human" "$n_packages" | tee -a "$OUTPUT_FILE"
  else
    printf "%-25s %10s\n" "$env_name" "NOT FOUND" | tee -a "$OUTPUT_FILE"
  fi
done

echo "" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
total_size=$(du -sh "$CONDA_ENVS_DIR" 2>/dev/null | cut -f1)
echo "Total: $total_size" | tee -a "$OUTPUT_FILE"

# Package details
echo "" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "Package Details:" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

for env_name in "${ENVIRONMENTS[@]}"; do
  [ -d "$CONDA_ENVS_DIR/$env_name" ] || continue
  echo "" >> "$OUTPUT_FILE"
  echo "=== $env_name ===" >> "$OUTPUT_FILE"
  mamba run -n "$env_name" mamba list 2>/dev/null >> "$OUTPUT_FILE" || true
done

echo ""
echo "✓ Results saved to: $OUTPUT_FILE"