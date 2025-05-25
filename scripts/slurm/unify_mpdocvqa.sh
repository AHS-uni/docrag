#!/bin/bash
#SBATCH --job-name=unify
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/users/hlwn057u2/data/logs/slurm/unify_%j.log
#SBATCH --error=/cluster/users/hlwn057u2/data/logs/slurm/unify_%j.err

# Exit immediately on error
set -e

# Load and activate conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "/cluster/users/hlwn057u2/.conda/envs/dev-env"

# Navigate to working directory
cd /cluster/users/hlwn057u2/data/projects/docrag

poetry install --no-interaction --no-ansi

# Run python script

export $(grep -v '^#' .env | xargs)

poetry run python - <<'PYCODE'
import os
from pathlib import Path
from docrag.unify.mpdocvqa import MPDocVQAUnifier
from docrag.datasets.utils import load_corpus_dataset, load_qa_dataset, push_dataset_to_hub

TOKEN = os.environ["TOKEN"]

ROOT = Path("data/mpdocvqa")  # adjust if needed

# 1) Unify raw MPDocVQA
unifier = MPDocVQAUnifier(name="MPDocVQA", data_dir=ROOT)
unifier.unify()

# 2) Push corpus
corpus_ds = load_corpus_dataset(ROOT, cast_image=True, streaming=False)
push_dataset_to_hub(corpus_ds, repo_id="AHS-uni/mpdocvqa-corpus", token=TOKEN, commit_message="Upload MPDocVQA corpus.")

# 3) Push QA splits
qa_ds = load_qa_dataset(ROOT, include_images=False, streaming=False)
push_dataset_to_hub(qa_ds, repo_id="AHS-uni/mpdocvqa-qa", token=TOKEN, commit_message="Upload MPDocVQA QA.")
PYCODE
