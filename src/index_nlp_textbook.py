#!/usr/bin/env python3
"""
NLP Textbook Parser for RAG System
Parses textbook chapters from .txt files and outputs normalized JSON format.
"""

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List


def find_textbook_files(directory: str) -> List[Path]:
    """
    Find all .txt files in the specified directory.

    Args:
        directory: Path to the directory containing textbook files

    Returns:
        List of Path objects for .txt files
    """
    directory_path = Path(directory)
    txt_files = list(directory_path.glob("*.txt"))

    # Sort files for consistent processing order
    txt_files.sort()

    print(f"Found {len(txt_files)} .txt files in {directory}")
    for file in txt_files:
        print(f"  - {file.name}")

    return txt_files


def extract_chapter_title(content: str, filename: str) -> str:
    """
    Extract chapter title from the file content.

    The chapter structure can vary:
    - CHAPTER → title (possibly multi-line) → number
    - CHAPTER → number → title

    Args:
        content: Full text content of the file
        filename: Name of the file (used as fallback)

    Returns:
        Extracted chapter title
    """
    lines = content.split('\n')

    # Find the line with "CHAPTER"
    chapter_index = -1
    for i, line in enumerate(lines):
        if line.strip() == "CHAPTER":
            chapter_index = i
            break

    if chapter_index == -1:
        # Fallback: use filename without extension
        return filename.replace('.txt', '').replace('-', ' ').replace('_', ' ').title()

    # Collect consecutive non-empty lines after "CHAPTER" to handle multi-line titles
    title_parts = []
    found_title_start = False

    for i in range(chapter_index + 1, min(chapter_index + 15, len(lines))):
        line = lines[i].strip()

        # Skip empty lines before title starts
        if not line:
            if found_title_start:
                # Empty line after title started means title is complete
                break
            continue

        # Skip metadata lines
        if 'Copyright' in line or 'Speech and Language' in line or 'Draft of' in line:
            continue

        # If it's a single digit or number, it's likely the chapter number
        if line.isdigit() and len(line) <= 2:
            if found_title_start:
                # Chapter number after title means title is complete
                break
            # Chapter number before title, keep looking
            continue

        # Stop if we hit an epigraph (quote starting with quote marks or containing "...")
        # Check for both straight quotes and curly quotes (Unicode)
        quote_chars = ['"', "'", '\u201c', '\u201d', '\u2018', '\u2019', '\u8343']  # includes 荃
        if any(line.startswith(char) for char in quote_chars) or '...' in line:
            break

        # Check if this line looks like prose (too long or lowercase start)
        # Title lines are typically capitalized and not super long
        if found_title_start and (len(line) > 60 or (line[0].islower() if line else False)):
            break

        # This is part of the title
        title_parts.append(line)
        found_title_start = True

        # Limit title to reasonable length (max 3 lines or ~15 words)
        total_words = sum(len(part.split()) for part in title_parts)
        if total_words >= 15:
            break

        # If we've collected title parts and hit another empty or short line, stop
        if len(title_parts) >= 1 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line or (next_line.isdigit() and len(next_line) <= 2):
                break

    # Join title parts with space
    if title_parts:
        return ' '.join(title_parts)

    # Fallback to filename if no title found
    return filename.replace('.txt', '').replace('-', ' ').replace('_', ' ').title()


def generate_id(content: str) -> str:
    """
    Generate a unique hash ID for the document.

    Args:
        content: Document content to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def parse_textbook_file(file_path: Path) -> Dict:
    """
    Parse a single textbook file and extract metadata.

    Args:
        file_path: Path to the .txt file

    Returns:
        Dictionary with parsed data matching the schema
    """
    # Read file content with UTF-8 encoding
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    # Extract chapter title
    chapter_title = extract_chapter_title(content, file_path.stem)

    # Get file size in KB
    file_size_bytes = file_path.stat().st_size
    file_size_kb = round(file_size_bytes / 1024, 2)

    # Generate unique ID
    doc_id = generate_id(content)

    # Build the normalized document
    document = {
        "id": doc_id,
        "source": "nlp_textbook",
        "chapter": chapter_title,
        "filename": file_path.name,
        "content": content,
        "file_path": str(file_path.absolute()),
        "file_size_kb": file_size_kb
    }

    print(f"Parsed: {file_path.name} → '{chapter_title}' ({file_size_kb} KB)")

    return document


def main():
    """
    Main execution function.
    Discovers textbook files, parses them, and saves to JSON.
    """
    # Configuration
    textbook_directory = "/Users/colinsidberry/Downloads/NLP_Textbook"
    output_file = "/Users/colinsidberry/nlp-textbook-rag/data/normalized/nlp_textbook.json"

    print("=" * 60)
    print("NLP Textbook Parser - Milestone 1")
    print("=" * 60)
    print()

    # Step 1: Find all .txt files
    print("Step 1: Discovering textbook files...")
    txt_files = find_textbook_files(textbook_directory)
    print()

    # Step 2: Parse each file
    print("Step 2: Parsing textbook files...")
    documents = []

    for file_path in txt_files:
        # Skip duplicate files (neural_networks.txt vs neural-networks.txt)
        if file_path.name == "neural_networks.txt":
            print(f"Skipping duplicate: {file_path.name}")
            continue

        try:
            document = parse_textbook_file(file_path)
            documents.append(document)
        except Exception as e:
            print(f"ERROR parsing {file_path.name}: {e}")

    print()

    # Step 3: Save to JSON
    print("Step 3: Saving to JSON...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(documents)} documents to {output_file}")
    print()

    # Step 4: Validation summary
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    total_size_kb = sum(doc['file_size_kb'] for doc in documents)
    print(f"Total documents: {len(documents)}")
    print(f"Total size: {total_size_kb:.2f} KB")
    print()

    print("Documents:")
    for doc in documents:
        print(f"  [{doc['id']}] {doc['chapter']} ({doc['file_size_kb']} KB)")
    print()

    # Verify all required fields
    required_fields = ['id', 'source', 'chapter', 'filename', 'content', 'file_path', 'file_size_kb']
    all_valid = True
    for doc in documents:
        for field in required_fields:
            if field not in doc:
                print(f"ERROR: Missing field '{field}' in document {doc.get('filename', 'unknown')}")
                all_valid = False

    if all_valid:
        print("✓ All documents have required fields")
    print()

    print("=" * 60)
    print("Milestone 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
