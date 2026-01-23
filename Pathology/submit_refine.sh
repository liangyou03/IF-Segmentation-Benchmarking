#!/bin/bash
#SBATCH --job-name=refine_seg
#SBATCH --output=refine_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-08:00:00
#SBATCH --mem=64G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
python refine_all.py
