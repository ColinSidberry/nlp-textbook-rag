#!/usr/bin/env python3
"""
PDF to Text Converter with Advanced Cleaning
Extracts text from PDFs and applies cleaning operations for RAG system
"""

import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional


def extract_text_from_pdf(pdf_path: str, use_layout: bool = True) -> Tuple[bool, str, str]:
    """
    Extract text from PDF using pdftotext.

    Args:
        pdf_path: Path to the PDF file
        use_layout: Whether to preserve layout (default: True)

    Returns:
        Tuple of (success: bool, text: str, error_message: str)
    """
    try:
        # Build pdftotext command
        cmd = ["pdftotext"]

        if use_layout:
            cmd.append("-layout")  # Preserve text layout

        cmd.append("-nopgbrk")  # Remove page break markers
        cmd.append(pdf_path)
        cmd.append("-")  # Output to stdout

        # Run pdftotext
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return True, result.stdout, ""

    except subprocess.CalledProcessError as e:
        error_msg = f"pdftotext failed: {e.stderr}"
        return False, "", error_msg
    except FileNotFoundError:
        error_msg = "pdftotext not found. Please install poppler-utils: brew install poppler"
        return False, "", error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return False, "", error_msg


def remove_headers_footers(text: str) -> str:
    """
    Remove common header and footer patterns from textbook pages.

    Args:
        text: Raw text content

    Returns:
        Cleaned text with headers/footers removed
    """
    lines = text.split('\n')
    cleaned_lines = []

    # Patterns to remove
    header_footer_patterns = [
        r'^Speech and Language Processing',
        r'^Copyright.*',
        r'^Draft of.*',
        r'^\d+\s*$',  # Page numbers (standalone)
        r'^Chapter \d+\s*$',  # Standalone "Chapter X"
        r'^\d+\.\d+\s+\w+',  # Section numbers at start (e.g., "2.3 Tokenization")
    ]

    for line in lines:
        stripped = line.strip()

        # Check if line matches any header/footer pattern
        is_header_footer = False
        for pattern in header_footer_patterns:
            if re.match(pattern, stripped):
                is_header_footer = True
                break

        # Keep line if it doesn't match any pattern
        if not is_header_footer:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def fix_hyphenation(text: str) -> str:
    """
    Fix hyphenation at line breaks.
    Merges words split across lines (e.g., "computa-\ntion" → "computation").

    Args:
        text: Text with potential hyphenation issues

    Returns:
        Text with hyphenation fixed
    """
    # Pattern: word ending with hyphen + newline + lowercase word
    # Replace with merged word (no hyphen, no newline)
    pattern = r'(\w+)-\s*\n\s*(\w+)'

    def merge_hyphenated(match):
        """Merge hyphenated words"""
        first_part = match.group(1)
        second_part = match.group(2)

        # Only merge if second part starts with lowercase
        if second_part[0].islower():
            return first_part + second_part
        else:
            # Keep hyphen and line break if second part is capitalized
            # (might be a new sentence)
            return f"{first_part}-\n{second_part}"

    return re.sub(pattern, merge_hyphenated, text)


def remove_tables_figures(text: str) -> str:
    """
    Remove or replace table and figure markers.

    Args:
        text: Text with potential tables/figures

    Returns:
        Text with tables/figures handled
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip figure captions
        if re.match(r'^Figure \d+[\.:]\d*', stripped):
            # Option: Skip completely
            continue
            # Option: Replace with placeholder
            # cleaned_lines.append(f"[{stripped}]")

        # Skip table markers
        elif re.match(r'^Table \d+[\.:]\d*', stripped):
            continue

        # Skip lines that are mostly special characters (likely table borders)
        elif len(stripped) > 0 and sum(c in '|+-_=*' for c in stripped) / len(stripped) > 0.5:
            continue

        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Text with irregular whitespace

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space (but preserve newlines)
    lines = text.split('\n')
    normalized_lines = []

    for line in lines:
        # Replace multiple spaces with single space
        normalized_line = re.sub(r' +', ' ', line)
        # Remove trailing whitespace
        normalized_line = normalized_line.rstrip()
        normalized_lines.append(normalized_line)

    # Join lines back together
    text = '\n'.join(normalized_lines)

    # Normalize line breaks: max 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def clean_text(raw_text: str) -> str:
    """
    Apply all cleaning operations to extracted text.

    Cleaning pipeline:
    1. Remove headers/footers
    2. Fix hyphenation
    3. Remove tables/figures
    4. Normalize whitespace

    Args:
        raw_text: Raw text extracted from PDF

    Returns:
        Cleaned text ready for indexing
    """
    # Apply cleaning operations in order
    text = raw_text

    # Step 1: Remove headers and footers
    text = remove_headers_footers(text)

    # Step 2: Fix hyphenation at line breaks
    text = fix_hyphenation(text)

    # Step 3: Remove tables and figures
    text = remove_tables_figures(text)

    # Step 4: Normalize whitespace
    text = normalize_whitespace(text)

    return text


def get_cleaning_stats(raw_text: str, cleaned_text: str) -> dict:
    """
    Get statistics about the cleaning process.

    Args:
        raw_text: Original text before cleaning
        cleaned_text: Text after cleaning

    Returns:
        Dictionary with cleaning statistics
    """
    raw_lines = raw_text.split('\n')
    cleaned_lines = cleaned_text.split('\n')

    raw_chars = len(raw_text)
    cleaned_chars = len(cleaned_text)

    return {
        'raw_lines': len(raw_lines),
        'cleaned_lines': len(cleaned_lines),
        'lines_removed': len(raw_lines) - len(cleaned_lines),
        'raw_chars': raw_chars,
        'cleaned_chars': cleaned_chars,
        'chars_removed': raw_chars - cleaned_chars,
        'reduction_percent': round((raw_chars - cleaned_chars) / raw_chars * 100, 2) if raw_chars > 0 else 0
    }


def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    verbose: bool = True
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Process a single PDF file: extract and clean text.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the cleaned .txt file
        verbose: Whether to print progress messages

    Returns:
        Tuple of (success: bool, output_path: Optional[str], stats: Optional[dict])
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    output_filename = pdf_path.stem + ".txt"
    output_path = output_dir / output_filename

    if verbose:
        print(f"Processing: {pdf_path.name}")

    # Step 1: Extract text from PDF
    success, raw_text, error = extract_text_from_pdf(str(pdf_path))

    if not success:
        if verbose:
            print(f"  ERROR: {error}")
        return False, None, None

    if verbose:
        print(f"  Extracted {len(raw_text)} characters")

    # Step 2: Clean text
    cleaned_text = clean_text(raw_text)

    # Get cleaning statistics
    stats = get_cleaning_stats(raw_text, cleaned_text)

    if verbose:
        print(f"  Cleaned: {stats['lines_removed']} lines removed, "
              f"{stats['chars_removed']} chars removed ({stats['reduction_percent']}%)")

    # Step 3: Save cleaned text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    if verbose:
        print(f"  Saved to: {output_path}")

    return True, str(output_path), stats


def main():
    """
    Main function for testing individual PDF conversion.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_to_text.py <pdf_file> [output_dir]")
        print("\nExample: python pdf_to_text.py sample.pdf ./output/")
        sys.exit(1)

    pdf_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"

    print("=" * 70)
    print("PDF TO TEXT CONVERTER")
    print("=" * 70)
    print()

    success, output_path, stats = process_single_pdf(pdf_file, output_dir, verbose=True)

    if success:
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Output: {output_path}")
        print()
        print("Statistics:")
        print(f"  Raw lines:     {stats['raw_lines']}")
        print(f"  Cleaned lines: {stats['cleaned_lines']}")
        print(f"  Lines removed: {stats['lines_removed']}")
        print(f"  Reduction:     {stats['reduction_percent']}%")
    else:
        print()
        print("FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
