#!/usr/bin/env python3
"""
sample_dataset.py — create a smaller sample of a DocVQA‐style (or similar)
dataset by selecting N documents and filtering Q&A JSON/JSONL files,
with optional regex-based ID extraction.
"""

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample N documents from a dataset and filter its QAS files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Root of the original dataset (must contain 'documents/' and 'raw_qas/').",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Root of the sampled dataset to create.",
    )
    parser.add_argument(
        "-n",
        "--num-docs",
        type=int,
        required=True,
        help="How many document folders to pick at random.",
    )
    parser.add_argument(
        "-k",
        "--filter-key",
        type=str,
        default="doc_id",
        help="Name of the JSON field in each QA entry to filter on.",
    )
    parser.add_argument(
        "--id-regex",
        type=str,
        default=None,
        help=(
            "Optional regex to extract the actual document ID from the filter-key value. "
            "The first capture group will be used. E.g. 'images/([^_]+)_\\d+\\.jpg'"
        ),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def extract_entries(obj):
    if isinstance(obj, list):
        return obj, None
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"], "data"
        for k, v in obj.items():
            if isinstance(v, list) and all(isinstance(it, dict) for it in v):
                return v, k
    raise ValueError("Could not locate a list of entries in JSON root")


def get_entry_id(entry: dict, key: str, id_re: re.Pattern | None) -> str | None:
    """
    Return the document ID for this entry by either:
      - If id_re is None: entry[key] (must be a string)
      - Else: apply id_re to entry[key], return group(1) on match.
    """
    raw = entry.get(key)
    if not isinstance(raw, str):
        return None
    if id_re is None:
        return raw
    m = id_re.search(raw)
    return m.group(1) if m else None


def filter_qas(
    src_path: Path,
    dst_path: Path,
    selected_ids: set[str],
    key: str,
    id_re: re.Pattern | None,
) -> int:
    data = json.loads(src_path.read_text(encoding="utf-8"))
    entries, container_key = extract_entries(data)

    filtered = []
    for e in entries:
        docid = get_entry_id(e, key, id_re)
        if docid is not None and docid in selected_ids:
            filtered.append(e)

    out = filtered if container_key is None else (data | {container_key: filtered})
    dst_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(filtered)


def filter_jsonl(
    src_path: Path,
    dst_path: Path,
    selected_ids: set[str],
    key: str,
    id_re: re.Pattern | None,
) -> int:
    count = 0
    with (
        src_path.open(encoding="utf-8") as src,
        dst_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            obj = json.loads(line)
            docid = get_entry_id(obj, key, id_re)
            if docid is not None and docid in selected_ids:
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Compile regex if provided
    id_re = re.compile(args.id_regex) if args.id_regex else None

    docs_in = args.input_dir / "documents"
    qas_in = args.input_dir / "raw_qas"
    docs_out = args.output_dir / "documents"
    qas_out = args.output_dir / "raw_qas"

    # Create output dirs
    for d in (docs_out, qas_out):
        d.mkdir(parents=True, exist_ok=True)

    # Sample document folders
    all_docs = [p.name for p in docs_in.iterdir() if p.is_dir()]
    if args.num_docs > len(all_docs):
        sys.exit(
            f"Requested {args.num_docs} docs but only found {len(all_docs)} available."
        )
    selected = set(random.sample(all_docs, args.num_docs))
    print(f"Sampling {len(selected)} documents: {sorted(selected)}")

    # Copy the sampled documents
    for doc_id in selected:
        shutil.copytree(docs_in / doc_id, docs_out / doc_id)

    # Filter Q&A files
    total = 0
    for pattern in ("*.json", "*.jsonl"):
        for src in qas_in.glob(pattern):
            dst = qas_out / src.name
            if src.suffix == ".jsonl":
                cnt = filter_jsonl(src, dst, selected, args.filter_key, id_re)
            else:
                cnt = filter_qas(src, dst, selected, args.filter_key, id_re)

            print(f"  → {src.name}: wrote {cnt} entries")
            total += cnt

    print(f"Done! Sampled dataset written to {args.output_dir}")
    print(f"Total Q&A entries across all files: {total}")


if __name__ == "__main__":
    main()
