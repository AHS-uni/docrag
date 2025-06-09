#!/usr/bin/env python3
"""
sample_dataset.py — create a smaller sample of a DocVQA‐style (or similar)
dataset by selecting N documents and filtering Q&A JSON/JSONL files,
with optional regex-based ID extraction.  Supports both nested "documents/"
directories and flat "raw_documents/" with file stems as IDs.
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
        help="Root of the original dataset (must contain 'documents/' or 'raw_documents/' and 'raw_qas/').",
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
        help="How many documents/files to pick at random.",
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
    """
    Locate a list of entries in a JSON object or array.

    Returns:
      entries: list of dicts
      container_key: if entries were under obj[container_key], else None (root array)
    """
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
    Return the document ID for this entry by drilling into nested keys.
    E.g. key="doc.uid" will fetch entry["doc"]["uid"].
    Then, if `id_re` is provided, apply it to the string.
    """
    # 1) Drill into nested dicts
    parts = key.split(".")
    raw = entry
    for p in parts:
        if not isinstance(raw, dict):
            return None
        raw = raw.get(p)
        # 2) Must end up with a string
    if not isinstance(raw, str):
        return None
    # 3) Optionally regex‐extract
    if id_re:
        m = id_re.search(raw)
        return m.group(1) if m else None
    return raw


def filter_qas(
    src_path: Path,
    dst_path: Path,
    selected_ids: set[str],
    key: str,
    id_re: re.Pattern | None,
) -> int:
    data = json.loads(src_path.read_text(encoding="utf-8"))
    entries, container_key = extract_entries(data)

    filtered = [
        e for e in entries if (docid := get_entry_id(e, key, id_re)) in selected_ids
    ]

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
            if (docid := get_entry_id(obj, key, id_re)) in selected_ids:
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Compile regex if provided
    id_re = re.compile(args.id_regex) if args.id_regex else None

    # Determine whether using flat raw_documents or nested documents
    raw_docs_dir = args.input_dir / "raw_documents"
    nested_docs_dir = args.input_dir / "documents"

    if raw_docs_dir.exists():
        docs_in = raw_docs_dir
        is_flat = True
        docs_out = args.output_dir / "raw_documents"
    elif nested_docs_dir.exists():
        docs_in = nested_docs_dir
        is_flat = False
        docs_out = args.output_dir / "documents"
    else:
        sys.exit(
            "ERROR: No 'documents/' or 'raw_documents/' directory found in input path."
        )

    # Prepare output docs directory
    docs_out.mkdir(parents=True, exist_ok=True)

    # Collect and sample
    if is_flat:
        all_items = [p for p in docs_in.iterdir() if p.is_file()]
        if args.num_docs > len(all_items):
            sys.exit(
                f"Requested {args.num_docs} docs but only found {len(all_items)} files."
            )
        selected_items = random.sample(all_items, args.num_docs)
        selected_ids = {p.stem for p in selected_items}
        print(f"Sampling {len(selected_items)} files: {sorted(selected_ids)}")
        for src in selected_items:
            shutil.copy2(src, docs_out / src.name)
    else:
        all_dirs = [p for p in docs_in.iterdir() if p.is_dir()]
        if args.num_docs > len(all_dirs):
            sys.exit(
                f"Requested {args.num_docs} docs but only found {len(all_dirs)} directories."
            )
        selected_dirs = random.sample(all_dirs, args.num_docs)
        selected_ids = {p.name for p in selected_dirs}
        print(f"Sampling {len(selected_dirs)} directories: {sorted(selected_ids)}")
        for src in selected_dirs:
            shutil.copytree(src, docs_out / src.name)

    # Filter Q&A
    qas_in = args.input_dir / "raw_qas"
    qas_out = args.output_dir / "raw_qas"
    qas_out.mkdir(parents=True, exist_ok=True)

    total = 0
    for pattern in ("*.json", "*.jsonl"):  # process both JSON and JSONL
        for src in qas_in.glob(pattern):
            dst = qas_out / src.name
            cnt = (
                filter_jsonl(src, dst, selected_ids, args.filter_key, id_re)
                if src.suffix == ".jsonl"
                else filter_qas(src, dst, selected_ids, args.filter_key, id_re)
            )
            print(f"  → {src.name}: wrote {cnt} entries")
            total += cnt

    print(f"Done! Sampled dataset written to {args.output_dir}")
    print(f"Total Q&A entries across all files: {total}")


if __name__ == "__main__":
    main()
