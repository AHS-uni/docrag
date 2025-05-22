#!/usr/bin/env python3
"""
standardize_documents.py

Moves and renames raw page images into a standardized nested structure:
    documents/<doc_id>/<page_idx:03d>.<ext>

Supports both:
  - multi-page datasets with filenames like <doc_id>_p<page_no>.<ext> or <doc_id>_<page_no>.<ext>
  - single-page datasets with filenames like <doc_id>.<ext>

Detects and logs:
  - duplicate pages per document (duplicates.txt)
  - missing pages per document (missing.txt)
  - invalid filenames (invalid.txt)
"""

import re
import argparse
from pathlib import Path
from shutil import move
from collections import defaultdict

# Allow optional 'p' before page number: jmng0065_p1.jpg or jmng0065_1.jpg
FILENAME_PATTERN = re.compile(
    r"^(?P<doc_id>.+?)_(?:p)?(?P<page>\d+)\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Standardize raw page images into documents/<doc_id>/<page_idx:03d>.<ext>"
    )
    parser.add_argument(
        "images_dir", type=Path, help="Directory containing raw image files."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory where 'documents/' will be created.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Treat each image as a standalone document (<doc_id>.<ext> → doc_id/000.<ext>).",
    )
    return parser.parse_args()


def find_files(images_dir: Path, single: bool = False):
    """
    Scan images_dir and classify files.

    Args:
        images_dir: Directory to scan.
        single:     If True, expects filenames <doc_id>.<ext>.

    Returns:
        valid:   List of tuples (doc_id, raw_page, ext, Path).
        invalid: List of Paths that didn't match expected patterns.
    """
    valid, invalid = [], []
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        if single:
            ext = path.suffix.lstrip(".").lower()
            if ext in ("jpg", "jpeg", "png"):
                valid.append((path.stem, 0, ext, path))
            else:
                invalid.append(path)
        else:
            m = FILENAME_PATTERN.match(path.name)
            if m:
                gd = m.groupdict()
                valid.append((gd["doc_id"], int(gd["page"]), gd["ext"].lower(), path))
            else:
                invalid.append(path)
    return valid, invalid


def detect_indexing(pages: list[int]) -> int:
    """
    Determine if page indices are zero- or one-based.

    Args:
        pages: List of raw page numbers.

    Returns:
        1 if one-based (min >=1), else 0.
    """
    return 1 if pages and min(pages) >= 1 else 0


def main():
    """Main processing function."""
    args = parse_args()
    images_dir = args.images_dir
    output_root = args.output_dir / "documents"
    output_root.mkdir(parents=True, exist_ok=True)

    # Gather files
    valid, invalid_files = find_files(images_dir, single=args.single)
    with open("invalid.txt", "w") as f_inv:
        for p in invalid_files:
            f_inv.write(f"{p}\n")

    # Group by document id
    docs = defaultdict(list)
    for doc_id, raw_page, ext, path in valid:
        docs[doc_id].append((raw_page, ext, path))

    duplicates = []
    missing = []

    # Process each document
    for doc_id, entries in docs.items():
        raw_pages = [raw for raw, _, _ in entries]
        offset = 0 if args.single else detect_indexing(raw_pages)
        normalized = [(raw - offset, ext, path) for raw, ext, path in entries]

        # Detect duplicates
        page_map = defaultdict(list)
        for pg, ext, path in normalized:
            page_map[pg].append((ext, path))
            dup_entries = {
                pg: paths for pg, paths in page_map.items() if len(paths) > 1
            }
        with open("duplicates.txt", "a") as f_dup:
            for pg, paths in dup_entries.items():
                line = f"{doc_id},{pg}," + ",".join(str(p) for _, p in paths) + "\n"
                f_dup.write(line)
                duplicates.append((doc_id, pg, paths))

        # Detect missing pages (skip for single-page)
        all_pages = sorted(page_map.keys())
        if not args.single and all_pages:
            expected = set(range(min(all_pages), max(all_pages) + 1))
            missing_pages = sorted(expected - set(all_pages))
            if missing_pages:
                with open("missing.txt", "a") as f_miss:
                    f_miss.write(f"{doc_id},{missing_pages}\n")
                    missing.append((doc_id, missing_pages))

        # Move valid pages
        target_dir = output_root / doc_id
        target_dir.mkdir(exist_ok=True)
        skip_pages = set(dup_entries.keys())
        if not args.single:
            skip_pages |= set(missing_pages) if all_pages else set()
        for pg, ext, path in normalized:
            if pg in skip_pages:
                continue
            new_name = f"{pg:03d}.{ext}"
            move(str(path), str(target_dir / new_name))

    # Summary
    print(f"Documents processed: {len(docs)}")
    print(f"Invalid files:       {len(invalid_files)} (invalid.txt)")
    print(f"Duplicates found:    {len(duplicates)} (duplicates.txt)")
    print(f"Missing pages:       {len(missing)} (missing.txt)")


if __name__ == "__main__":
    main()
