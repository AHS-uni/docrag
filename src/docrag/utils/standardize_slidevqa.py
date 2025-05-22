#!/usr/bin/env python3
"""
standardize_slidevqa.py

Rename SlideVQA images from:

  <dataset_root>/raw/images/<doc_id>/slide_<n>_1024.jpg

to:

  <dataset_root>/raw/images/<doc_id>/<page_idx:03d>.jpg

where page_idx = n - start_index (default start_index=1).

Usage:
    ./standardize_slidevqa.py /path/to/raw/images --start-index 1
"""

import re
import argparse
from pathlib import Path

SLIDE_PATTERN = re.compile(r"^slide_(\d+)_\d+\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rename SlideVQA slide_N_1024.jpg → zero-padded page images."
    )
    parser.add_argument(
        "images_root",
        type=Path,
        help="Root folder containing subdirs per document, each with slide_*.jpg files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="The slide numbering offset (1 if slides start at 1, 0 if already zero-based).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.images_root
    offset = args.start_index

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir():
            continue
        print(f"Processing document: {doc_dir.name}")
        for img in sorted(doc_dir.iterdir()):
            m = SLIDE_PATTERN.match(img.name)
            if not m:
                print(f"  [SKIP] {img.name}")
                continue
            n_str, ext = m.groups()
            n = int(n_str)
            page_idx = n - offset
            if page_idx < 0:
                print(f"  [ERROR] Negative page index for {img.name}, skipping.")
                continue
            new_name = f"{page_idx:03d}.{ext.lower()}"
            target = doc_dir / new_name
            if target.exists():
                print(
                    f"  [WARN] target exists {target.name}, skipping rename of {img.name}"
                )
                continue
            img.rename(target)
            print(f"  {img.name} → {new_name}")


if __name__ == "__main__":
    main()
