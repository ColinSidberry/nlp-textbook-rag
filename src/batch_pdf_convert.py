#!/usr/bin/env python3
"""
Batch PDF to Text Converter
Processes multiple PDFs in a directory with progress tracking and error handling
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime

# Import the PDF processing functions
from pdf_to_text import process_single_pdf


def find_pdf_files(pdf_dir: str) -> List[Path]:
    """
    Find all PDF files in the specified directory.

    Args:
        pdf_dir: Directory containing PDF files

    Returns:
        Sorted list of PDF file paths
    """
    pdf_dir = Path(pdf_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    # Find all PDF files
    pdf_files = []
    seen_files = set()  # Track files we've already added

    # First, add files with .pdf extension
    for item in pdf_dir.iterdir():
        if item.is_file() and item.suffix.lower() == '.pdf':
            pdf_files.append(item)
            seen_files.add(item)

    # Then check all other files to see if they're PDFs (by header)
    # This handles files like "16.Text-to-Speech" which are PDFs without .pdf extension
    for item in pdf_dir.iterdir():
        if item.is_file() and item not in seen_files:
            # Check if it's a PDF by reading file header
            try:
                with open(item, 'rb') as f:
                    header = f.read(4)
                    if header == b'%PDF':
                        pdf_files.append(item)
                        seen_files.add(item)
            except:
                pass

    # Sort files by name
    pdf_files.sort(key=lambda p: p.name)

    return pdf_files


def batch_convert_pdfs(
    pdf_dir: str = "/Users/colinsidberry/Downloads/Convert to txt/",
    output_dir: str = "/Users/colinsidberry/nlp-textbook-rag/data/extracted_text/",
    skip_existing: bool = True,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Convert all PDFs in a directory to cleaned text files.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save cleaned .txt files
        skip_existing: Skip PDFs that have already been converted
        verbose: Print progress messages

    Returns:
        Dictionary with conversion summary statistics
    """
    # Initialize tracking variables
    start_time = time.time()
    results = {
        'total_pdfs': 0,
        'successful': 0,
        'skipped': 0,
        'failed': 0,
        'failed_files': [],
        'stats': []
    }

    # Find all PDF files
    if verbose:
        print("=" * 70)
        print("BATCH PDF TO TEXT CONVERTER")
        print("=" * 70)
        print(f"\nScanning directory: {pdf_dir}")

    pdf_files = find_pdf_files(pdf_dir)
    results['total_pdfs'] = len(pdf_files)

    if verbose:
        print(f"Found {len(pdf_files)} PDF files")
        print(f"Output directory: {output_dir}")
        print()

    if len(pdf_files) == 0:
        if verbose:
            print("No PDF files found!")
        return results

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each PDF
    if verbose:
        print("=" * 70)
        print("PROCESSING PDFs")
        print("=" * 70)
        print()

    for i, pdf_file in enumerate(pdf_files, 1):
        # Check if output file already exists
        output_filename = pdf_file.stem + ".txt"
        output_file = output_path / output_filename

        if skip_existing and output_file.exists():
            if verbose:
                print(f"[{i}/{len(pdf_files)}] Skipping {pdf_file.name} (already converted)")
            results['skipped'] += 1
            continue

        # Process the PDF
        if verbose:
            print(f"\n[{i}/{len(pdf_files)}] {pdf_file.name}")
            print("-" * 70)

        try:
            success, output, stats = process_single_pdf(
                str(pdf_file),
                output_dir,
                verbose=verbose
            )

            if success:
                results['successful'] += 1
                results['stats'].append({
                    'filename': pdf_file.name,
                    'output': output,
                    'stats': stats
                })
            else:
                results['failed'] += 1
                results['failed_files'].append(pdf_file.name)

        except Exception as e:
            if verbose:
                print(f"  ERROR: {str(e)}")
            results['failed'] += 1
            results['failed_files'].append(pdf_file.name)

    # Calculate summary statistics
    elapsed_time = time.time() - start_time

    # Print summary
    if verbose:
        print()
        print("=" * 70)
        print("CONVERSION SUMMARY")
        print("=" * 70)
        print(f"\nTotal PDFs found:      {results['total_pdfs']}")
        print(f"Successfully converted: {results['successful']}")
        print(f"Skipped (existing):    {results['skipped']}")
        print(f"Failed:                {results['failed']}")
        print(f"\nTotal time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

        if results['failed'] > 0:
            print("\nFailed files:")
            for filename in results['failed_files']:
                print(f"  - {filename}")

        # Cleaning statistics
        if results['stats']:
            total_lines_removed = sum(s['stats']['lines_removed'] for s in results['stats'])
            total_chars_removed = sum(s['stats']['chars_removed'] for s in results['stats'])
            avg_reduction = sum(s['stats']['reduction_percent'] for s in results['stats']) / len(results['stats'])

            print("\nCleaning statistics:")
            print(f"  Total lines removed:  {total_lines_removed:,}")
            print(f"  Total chars removed:  {total_chars_removed:,}")
            print(f"  Average reduction:    {avg_reduction:.1f}%")

        print()
        print("=" * 70)
        print("COMPLETE!")
        print("=" * 70)
        print(f"Output directory: {output_dir}")

    return results


def list_pdfs(pdf_dir: str):
    """
    List all PDFs in a directory (utility function).

    Args:
        pdf_dir: Directory to scan
    """
    print(f"Scanning: {pdf_dir}\n")

    pdf_files = find_pdf_files(pdf_dir)

    print(f"Found {len(pdf_files)} PDF files:")
    print()

    for i, pdf_file in enumerate(pdf_files, 1):
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        print(f"{i:2}. {pdf_file.name:<60} ({size_mb:.2f} MB)")

    total_size = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
    print()
    print(f"Total size: {total_size:.2f} MB")


def main():
    """
    Main function for batch conversion.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch convert PDFs to cleaned text files for RAG system'
    )
    parser.add_argument(
        '--pdf-dir',
        default='/Users/colinsidberry/Downloads/Convert to txt/',
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--output-dir',
        default='/Users/colinsidberry/nlp-textbook-rag/data/extracted_text/',
        help='Directory to save cleaned .txt files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-conversion even if output files exist'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List PDF files without converting'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        list_pdfs(args.pdf_dir)
        return

    # Convert mode
    skip_existing = not args.force
    verbose = not args.quiet

    results = batch_convert_pdfs(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        skip_existing=skip_existing,
        verbose=verbose
    )

    # Exit with error code if any conversions failed
    if results['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
