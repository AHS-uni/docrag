#!/usr/bin/env python3
"""
flatten_tatdqa.py

Convert a “document‐major” TAT-DQA JSON into a “question‐major” JSON.
Each top‐level document entry (with its "doc" and "questions" array)
will be expanded into one output record per question, copying the doc fields
into each question‐level record.

Usage:
    python flatten_tatdqa.py --input path/to/original.json --output path/to/flattened.json
"""

import json
import argparse
from pathlib import Path


def flatten_document_major_to_question_major(input_path: Path, output_path: Path):
    """
    Reads a JSON file containing a top-level array of objects like:
      {
        "doc": { "uid": ..., "page": ..., "source": ... },
        "questions": [
          { "uid": ..., "order": ..., "question": ..., (plus answer, answer_type, etc.) },
          ...
        ]
      }
    and writes out a new JSON array where each item is one question object,
    with document fields merged in under "doc_…".
    """
    data = json.loads(input_path.read_text(encoding="utf-8"))
    flattened = []

    for doc_entry in data:
        doc = doc_entry.get("doc", {})
        questions = doc_entry.get("questions", [])

        # Required doc fields—will be copied into each question record:
        doc_uid = doc.get("uid")
        doc_page = doc.get("page")
        doc_source = doc.get("source")

        for q in questions:
            # Build a new dict for each question:
            question_record = {
                # Copy over all doc‐level info under top‐level keys:
                "doc_uid": doc_uid,
                "doc_page": doc_page,
                "doc_source": doc_source,
                # Copy every question‐level field as‐is:
                "question_uid": q.get("uid"),
                "order": q.get("order"),
                "question": q.get("question"),
            }

            # The following keys exist in train/dev/test_gold but not in pure test:
            #   answer, derivation, answer_type, scale, req_comparison, facts, block_mapping
            # We copy them if present; otherwise they remain absent.
            if "answer" in q:
                question_record["answer"] = q["answer"]
            if "derivation" in q:
                question_record["derivation"] = q["derivation"]
            if "answer_type" in q:
                question_record["answer_type"] = q["answer_type"]
            if "scale" in q:
                question_record["scale"] = q["scale"]
            if "req_comparison" in q:
                question_record["req_comparison"] = q["req_comparison"]
            if "facts" in q:
                question_record["facts"] = q["facts"]
            if "block_mapping" in q:
                question_record["block_mapping"] = q["block_mapping"]

            flattened.append(question_record)

    # Write out as a single JSON array
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(flattened)} question‐level records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten TAT-DQA from document‐major to question‐major format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the original TAT-DQA JSON (document‐major).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path where the flattened question‐major JSON should be written.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file {args.input} does not exist.")
        return

    flatten_document_major_to_question_major(args.input, args.output)


if __name__ == "__main__":
    main()
