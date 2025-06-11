#!/bin/bash
#SBATCH --job-name=push_to_hub
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/users/hlwn057u2/data/logs/slurm/push_to_hub_%j.log
#SBATCH --error=/cluster/users/hlwn057u2/data/logs/slurm/push_to_hub_%j.err

# Exit immediately on error
set -e

# Load and activate environment
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate docrag-cpu

# Navigate to working directory
cd /cluster/users/hlwn057u2/data/projects/docrag/

# Run python script
if [ -f .env ]; then
   set -a
   source .env
   set +a
fi

: "${ROOT_DIR:=data/${DATASET_NAME:-tatdqa}}"
: "${REPO_ID:?Need REPO_ID (e.g. user/tatdqa_qa)}"
: "${TARGET:=qa}"
: "${MESSAGE:="Upload via Slurm"}"
: "${PRIVATE:=false}"

SPLIT_FLAGS=()
if [ "$TARGET" = "qa" ]; then
  if [ -n "${SPLITS:-}" ]; then
    # Allow semicolon-delimited list:  train=...,val=...,test=...
    IFS=';' read -ra PAIRS <<< "$SPLITS"
    for p in "${PAIRS[@]}"; do
      [[ -z "$p" ]] && continue
      SPLIT_FLAGS+=(--split "$p")
    done
  else
    # Fallback to individual vars: SPLIT_TRAIN, SPLIT_VAL, ...
    for nm in train val test dev; do
      var="SPLIT_${nm^^}"
      if [ -n "${!var:-}" ]; then
        SPLIT_FLAGS+=(--split "${nm}=${!var}")
      fi
    done
  fi
  if (( ${#SPLIT_FLAGS[@]} == 0 )); then
    echo "ERROR: no splits provided (SPLITS or SPLIT_*)." >&2
    exit 1
  fi
fi

echo "[$(date)] Pushing '$TARGET' from '$ROOT_DIR' to HF repo '$REPO_ID' (private=$PRIVATE)"

python scripts/run/push_to_hub/push_to_hub.py \
       --root-dir "$ROOT_DIR" \
       --repo-id "$REPO_ID" \
       --target "$TARGET" \
       --message "$MESSAGE" \
       $([ "$PRIVATE" = "true" ] && echo "--private") \
       "${SPLIT_FLAGS[@]}"

echo "[$(date)] Push complete."
