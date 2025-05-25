#!/usr/bin/env python3
"""
Split a JSON QA file into multiple files based on a given attribute.
Each output file preserves the original top-level metadata, adds a `split` field, and contains only entries matching that split.
"""
import json
import argparse
import os
import sys


def split_qa_file(input_path: str, split_field: str, output_dir: str) -> None:
    # Load the original dataset
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Extract entries
    entries = dataset.get("data")
    if entries is None:
        print("Error: No 'data' array found in the input file.", file=sys.stderr)
        sys.exit(1)

    # Group entries by split value
    groups: dict[str, list] = {}
    for entry in entries:
        key = entry.get(split_field)
        if key is None:
            key = "unknown"
        groups.setdefault(key, []).append(entry)

    # Prepare metadata (everything except 'data')
    metadata = {k: v for k, v in dataset.items() if k != "data"}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write one file per split
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    for split_value, split_entries in groups.items():
        out = metadata.copy()
        out["split"] = split_value
        out["data"] = split_entries

        filename = f"{base_name}_{split_value}.json"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as of:
            json.dump(out, of, indent=2, ensure_ascii=False)
        print(f"Wrote {len(split_entries)} entries to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a QA JSON file by a given entry attribute into multiple files."
    )
    parser.add_argument("input", help="Path to the original JSON file")
    parser.add_argument(
        "--field",
        default="data_split",
        help="Attribute name on each entry to split by (default: data_split)",
    )
    parser.add_argument(
        "--outdir", default="splits", help="Directory to write split files into"
    )
    args = parser.parse_args()

    split_qa_file(args.input, args.field, args.outdir)


if __name__ == "__main__":
    main()
