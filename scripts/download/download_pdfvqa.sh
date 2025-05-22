#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/users/hlwn057u2/data/logs/slurm/download_%j.log
#SBATCH --error=/cluster/users/hlwn057u2/data/logs/slurm/download_%j.err

# Exit immediately on error
set -e

# Load and activate conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "/cluster/users/hlwn057u2/.conda/envs/dev-env"

# Navigate to working directory
cd /cluster/users/hlwn057u2/data/projects/docrag/

# Run python script
python /cluster/users/hlwn057u2/data/scripts/hf_download.py gigant/pdfvqa data/pdfvqa/raw
