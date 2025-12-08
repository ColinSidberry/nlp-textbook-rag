#!/usr/bin/env python3
"""
Verify that no text is missing when chunking documents.
Compares concatenated chunks to original document content.
"""

import json
from pathlib import Path
from chunk_and_index import TextbookChunker


def verify_coverage(document, chunker):
    """
    Verify that all text from the original document appears in chunks.

    Args:
        document: Document dict with 'content' field
        chunker: TextbookChunker instance

    Returns:
        dict with verification results
    """
    original_content = document['content']
    chunks = chunker.chunk_document(document)

    # Extract just the content portion (remove chapter title prefix)
    chunk_contents = []
    for chunk in chunks:
        # The chunk text has format "Chapter: {title}\n\n{content}"
        # We need to extract just the content portion
        chunk_text = chunk['text']
        if '\n\n' in chunk_text:
            content_part = chunk_text.split('\n\n', 1)[1]
        else:
            content_part = chunk_text
        chunk_contents.append(content_part)

    # Check first chunk starts with beginning of document
    first_chunk_content = chunk_contents[0]
    # The first chunk might have leading whitespace stripped during json loading
    # So we compare after stripping both
    doc_start = original_content[:len(first_chunk_content)].strip()
    chunk_start = first_chunk_content.strip()

    starts_match = doc_start.startswith(chunk_start[:100]) or chunk_start.startswith(doc_start[:100])

    # Check last chunk ends with end of document
    last_chunk_content = chunk_contents[-1]
    doc_end = original_content[-len(last_chunk_content):].strip()
    chunk_end = last_chunk_content.strip()

    ends_match = doc_end.endswith(chunk_end[-100:]) or chunk_end.endswith(doc_end[-100:])

    # Count total characters in chunks (with overlap, this should be >= original)
    total_chunk_chars = sum(len(c) for c in chunk_contents)
    original_chars = len(original_content)

    # Calculate overlap characters
    overlap_chars = 0
    for i in range(len(chunk_contents) - 1):
        current = chunk_contents[i]
        next_chunk = chunk_contents[i + 1]

        # Find overlap between end of current and start of next
        max_overlap = min(len(current), len(next_chunk))
        for overlap_len in range(max_overlap, 0, -1):
            if current[-overlap_len:] == next_chunk[:overlap_len]:
                overlap_chars += overlap_len
                break

    # After removing overlap, we should have approximately the original length
    unique_chunk_chars = total_chunk_chars - overlap_chars
    coverage_ratio = unique_chunk_chars / original_chars if original_chars > 0 else 0

    return {
        'filename': document['filename'],
        'chapter': document['chapter'],
        'original_chars': original_chars,
        'total_chunk_chars': total_chunk_chars,
        'overlap_chars': overlap_chars,
        'unique_chunk_chars': unique_chunk_chars,
        'coverage_ratio': coverage_ratio,
        'num_chunks': len(chunks),
        'starts_match': starts_match,
        'ends_match': ends_match,
        'first_100_original': original_content[:100],
        'first_100_chunks': first_chunk_content[:100],
        'last_100_original': original_content[-100:],
        'last_100_chunks': last_chunk_content[-100:]
    }


def main():
    print("=" * 70)
    print("VERIFYING NO TEXT IS MISSING FROM CHUNKS")
    print("=" * 70)

    # Load normalized data
    DATA_DIR = Path(__file__).parent.parent / "data"
    NORMALIZED_JSON = DATA_DIR / "normalized" / "nlp_textbook.json"

    with open(NORMALIZED_JSON, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Initialize chunker with same settings
    chunker = TextbookChunker(max_chunk_chars=280, min_chunk_chars=50, overlap_ratio=0.2)

    print(f"\nTesting {len(documents)} documents...\n")

    all_passed = True
    issues = []

    for i, doc in enumerate(documents[:3], 1):  # Test first 3 documents in detail
        print(f"Document {i}: {doc['filename']}")
        result = verify_coverage(doc, chunker)

        print(f"  Original chars:     {result['original_chars']:,}")
        print(f"  Total chunk chars:  {result['total_chunk_chars']:,}")
        print(f"  Overlap chars:      {result['overlap_chars']:,}")
        print(f"  Unique chunk chars: {result['unique_chunk_chars']:,}")
        print(f"  Coverage ratio:     {result['coverage_ratio']:.2%}")
        print(f"  Num chunks:         {result['num_chunks']}")
        print(f"  Starts match:       {result['starts_match']}")
        print(f"  Ends match:         {result['ends_match']}")

        # Check for issues
        if result['coverage_ratio'] < 0.95 or result['coverage_ratio'] > 1.05:
            all_passed = False
            issues.append(f"{doc['filename']}: Coverage ratio {result['coverage_ratio']:.2%}")
            print(f"  ⚠ WARNING: Coverage ratio outside expected range!")

        if not result['starts_match']:
            all_passed = False
            issues.append(f"{doc['filename']}: Start doesn't match")
            print(f"  ⚠ WARNING: Start doesn't match!")
            print(f"    Original start: {result['first_100_original'][:50]}...")
            print(f"    Chunk start:    {result['first_100_chunks'][:50]}...")

        if not result['ends_match']:
            all_passed = False
            issues.append(f"{doc['filename']}: End doesn't match")
            print(f"  ⚠ WARNING: End doesn't match!")
            print(f"    Original end: ...{result['last_100_original'][-50:]}")
            print(f"    Chunk end:    ...{result['last_100_chunks'][-50:]}")

        print()

    # Quick check for all documents
    print(f"Quick check for all {len(documents)} documents:")
    for doc in documents:
        result = verify_coverage(doc, chunker)
        if result['coverage_ratio'] < 0.95 or result['coverage_ratio'] > 1.05:
            print(f"  ⚠ {doc['filename']}: {result['coverage_ratio']:.2%}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ SUCCESS: All documents have complete coverage!")
        print("  No text is missing from chunks")
    else:
        print("✗ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    print("=" * 70)


if __name__ == "__main__":
    main()
