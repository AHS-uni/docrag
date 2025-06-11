#!/usr/bin/env python3
"""
Upload a unified DocRAG dataset (corpus or QA splits) to the Hugging Face Hub.

Examples
--------
# Push corpus only
python push_to_hub.py -r /data/tatdqa -n user/tatdqa_corpus -t corpus -m "corpus upload"

# Push QA with explicit splits
python push_to_hub.py -r /data/tatdqa -n user/tatdqa_qa -t qa \
  -s train=unified_qas/train.jsonl \
  -s val=unified_qas/val.jsonl \
  -m "initial QA upload" --private
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from docrag.datasets import load_corpus_dataset, load_qa_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Push a unified DocRAG dataset to the HF Hub."
    )

    p.add_argument(
        "-r", "--root-dir", required=True, type=Path, help="Dataset root directory."
    )
    p.add_argument(
        "-n", "--repo-id", required=True, help='Hub repo (e.g. "user/my_dataset").'
    )
    p.add_argument(
        "-t",
        "--target",
        required=True,
        choices=["corpus", "qa"],
        help="Which portion to upload.",
    )
    p.add_argument(
        "-m", "--message", default="Upload via push_to_hub.py", help="Commit message."
    )
    p.add_argument(
        "--private", action="store_true", help="Create / use a private repo."
    )
    p.add_argument(
        "--token", default=os.getenv("HF_TOKEN"), help="HF user token (env HF_TOKEN)."
    )
    p.add_argument(
        "--max-shard-size", default=None, help='e.g. "500MB" forwarded to push_to_hub.'
    )
    p.add_argument(
        "-s",
        "--split",
        action="append",
        metavar="NAME=FILE",
        help="Explicit QA split mapping. Repeatable. Required when --target qa.",
    )
    p.add_argument(
        "--streaming", action="store_true", help="Load dataset in streaming mode."
    )
    return p.parse_args()



def main() -> None:
    args = parse_args()
    root = args.root_dir.expanduser().resolve()

    ds: Dataset | DatasetDict

    if args.target == "corpus":
        ds = load_corpus_dataset(root)
    else:
        if not args.split:
            sys.exit(
                "Error: --target qa requires at least one --split NAME=FILE argument."
            )

        split_map: dict[str, str] = {}
        for item in args.split:
            if "=" not in item:
                sys.exit(f"--split expects NAME=FILE, got '{item}'")
            name, rel = item.split("=", 1)
            split_map[name.strip()] = rel.strip()

        ds = load_qa_dataset(root, splits=split_map)


    api = HfApi()
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=args.token,
        )
    except Exception as e:
        sys.exit(f"Hub error: {e}")


    commit = ds.push_to_hub(
        repo_id=args.repo_id,
        token=args.token,
        commit_message=args.message,
        max_shard_size=args.max_shard_size,
        private=args.private,
    )
    print(f"Uploaded → {commit.repo_url}\n   commit: {commit}")


if __name__ == "__main__":
    main()
