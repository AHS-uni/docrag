#!/bin/bash
#SBATCH --job-name=unify
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
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
cd /cluster/users/hlwn057u2/data/projects/docrag/

poetry install --no-interaction --no-ansi

# Run python script

# -- Load HF token
export $(grep -v '^#' .env | xargs)

poetry run python - <<'PYCODE'
import os
from pathlib import Path
from docrag.unify.dude import DUDEUnifier
from docrag.datasets.utils import load_corpus_dataset, load_qa_dataset, push_dataset_to_hub

TOKEN = os.environ["TOKEN"]
ROOT = Path("data/dude")

# 1) Unify raw DUDE
unifier = DUDEUnifier(name="DUDE", data_dir=ROOT, remove_insane=True)
unifier.unify()

# 2) Load & push corpus
corpus_ds = load_corpus_dataset(ROOT, cast_image=True, streaming=False)
push_dataset_to_hub(corpus_ds, repo_id="AHS-uni/dude-corpus", token=TOKEN, commit_message="Upload DUDE corpus.")

# 3) Load & push QA splits
splits = {
    "train": "unified_qas/2023-03-23_DUDE_gt_test_PUBLIC_train.jsonl",
    "val": "unified_qas/2023-03-23_DUDE_gt_test_PUBLIC_val.jsonl",
    "test": "unified_qas/2023-03-23_DUDE_gt_test_PUBLIC_test.jsonl"
}
qa_ds = load_qa_dataset(ROOT, splits=splits, include_images=False, streaming=False)
push_dataset_to_hub(qa_ds, repo_id="AHS-uni/dude-qa", token=TOKEN, commit_message="Upload DUDE QA.")
PYCODE
