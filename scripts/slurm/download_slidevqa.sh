#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/users/hlwn057u2/data/logs/slurm/download_%j.log
#SBATCH --error=/cluster/users/hlwn057u2/data/logs/slurm/download_%j.err

# Exit immediately on error
set -e

# Load and activate conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "/cluster/users/hlwn057u2/.conda/envs/dev-env"

# Navigate to working directory
cd /cluster/users/hlwn057u2/data/projects/docrag/data/slidevqa/raw

# Download image archive
gdown 'https://drive.google.com/uc?id=11bsX48cPpzCfPBnYJgSesvT7rWc84LpH'

# Extract archive
tar -xvzf images.tar.gz
