#!/usr/bin/env python3
"""
Run a Unifier for a specified dataset, using the registry to locate the correct subclass.

Usage examples:
  python unify.py -d tatdqa -r /path/to/tatdqa_dataset
  python unify.py --dataset mmlongbenchdoc --root-dir /data/mmld --test

This script relies on each dataset’s unifier module registering itself via
the `@register_unifier("<dataset>")` decorator. Once all modules are imported,
you can look up the appropriate Unifier subclass by name.
"""

import argparse
from pathlib import Path

from docrag.unification import get_unifier


def main():
    """Parse arguments, find the Unifier via registry, and invoke `unify()`."""
    parser = argparse.ArgumentParser(
        description="Run a dataset Unifier via registry lookup."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset slug (e.g. 'tatdqa', 'dude', 'mmlongbenchdoc').",
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "-s",
        "--skip-problematic",
        action="store_true",
        help="Skip problematic documents and questions.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Run in test mode (warnings instead of errors).",
    )

    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    UnifierClass = get_unifier(dataset_name)

    unifier = UnifierClass(
        name=dataset_name,
        dataset_root=args.root_dir,
        test_mode=args.test,
        skip_problematic=args.skip_problematic,
    )
    unified_path = unifier.unify()
    print(f"Unified QA files written to: {unified_path}")


if __name__ == "__main__":
    main()
