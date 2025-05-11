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
