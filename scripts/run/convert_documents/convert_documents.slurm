#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/users/hlwn057u2/data/logs/slurm/convert_%j.log
#SBATCH --error=/cluster/users/hlwn057u2/data/logs/slurm/convert_%j.err

# Exit immediately on error
set -e

# Load and activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate docrag-cpu

# Navigate to working directory
cd /cluster/users/hlwn057u2/data/projects/docrag/

INPUT_DIR="data/tatdqa/archive/pdfs/"
OUTPUT_DIR="data/tatdqa/documents/"

# Run python script
python scripts/run/convert_documents/convert_documents.py \
       --input-dir "$INPUT_DIR" \
       --output-dir "$OUTPUT_DIR" \
       --num-workers "$SLURM_CPUS_PER_TASK" \
       --dpi 300 \
       --quality 95
