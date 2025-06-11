#!/bin/bash
#SBATCH --job-name=unify
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/users/hlwn057u2/data/logs/slurm/unify_%j.log
#SBATCH --error=/cluster/users/hlwn057u2/data/logs/slurm/unify_%j.err

# Exit immediately on error
set -e

# Load and activate environment
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate docrag-cpu

# Navigate to working directory
cd /cluster/users/hlwn057u2/data/projects/docrag/

# Run python script
: "${DATASET_NAME:=tatdqa}"
: "${DATASET_ROOT:=data/${DATASET_NAME}}"
: "${TEST_MODE:=false}"
: "${SKIP_PROBLEMATIC:=false}"

echo "[$(date)] Starting unification of '$DATASET_NAME' (test=$TEST_MODE) (skip_problematic=$SKIP_PROBLEMATIC) at root '$DATASET_ROOT'"

python scripts/run/unify/unify.py \
       --dataset "$DATASET_NAME" \
       --root-dir "$DATASET_ROOT" \
       $([ "$TEST_MODE" = "true" ] && echo "--test") \
       $([ "$SKIP_PROBLEMATIC" = "true" ] && echo "--skip-problematic")

echo "[$(date)] Finished unification of '$DATASET_NAME'"
