import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from pdf2image import convert_from_path


def pdf_to_jpegs(pdf_path: Path, output_root: Path, dpi: int = 300, quality: int = 95):
    """Convert a single PDF into individual JPEG images.

    Each page of the PDF at `pdf_path` will be rendered at `dpi` and saved as a JPEG
    with the specified `quality`. Output files are placed under
    `output_root/<pdf_stem>/<page_index>.jpg`, where `page_index` is zero-indexed
    with three-digit padding (e.g., 000, 001, ...).

    Args:
        pdf_path (Path): Path to the input PDF file.
        output_root (Path): Directory under which to create a subfolder named after
            the PDF stem and store page images.
        dpi (int, optional): Dots per inch for rendering. Defaults to 300.
        quality (int, optional): JPEG quality (0–100). Defaults to 95.
    """
    doc_id = pdf_path.stem

    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        print(f"[ERROR] Failed to convert {pdf_path}: {e}")
        return

    if not pages:
        print(f"[WARNING] No pages found in {pdf_path}")
        return

    output_dir = output_root / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, page_image in enumerate(pages):
        filename = f"{idx:03d}.jpg"
        save_path = output_dir / filename
        page_image.save(save_path, format="JPEG", quality=quality)


def main():
    """Convert all PDFs in an input directory to JPEGs in parallel.

    Parses command-line arguments for the input directory containing PDFs, the
    output directory for storing JPEGs, and the number of worker processes. Each PDF
    is dispatched to a worker that invokes `pdf_to_jpegs`. The script prints a
    completion message once all files have been processed.
    """
    parser = argparse.ArgumentParser(
        description="Convert PDFs to 300dpi JPEGs organized under subfolders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory under which to store JPEG subfolders.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes to use.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for rendering PDF pages.",
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality (0–100) for saved images."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(args.input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {args.input_dir}. Exiting.")
        return

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for pdf_file in pdf_files:
            executor.submit(
                pdf_to_jpegs, pdf_file, args.output_dir, args.dpi, args.quality
            )

    print("All PDFs have been processed.")


if __name__ == "__main__":
    main()
