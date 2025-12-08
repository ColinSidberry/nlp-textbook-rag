#!/usr/bin/env python3
"""
Test script to verify chunk overlap ratio is correct.
Samples consecutive chunk pairs and calculates actual overlap percentage.
"""

import chromadb
import random
from typing import List, Tuple
from pathlib import Path


def strip_chapter_prefix(text: str) -> str:
    """
    Remove the 'Chapter: ...' prefix that's prepended to each chunk.
    Returns just the content portion.
    """
    if '\n\n' in text:
        return text.split('\n\n', 1)[1]
    return text


def get_overlap_ratio(text1: str, text2: str) -> float:
    """
    Calculate overlap ratio between two text chunks.
    Returns the ratio of overlapping characters to the first chunk's length.
    """
    if not text1 or not text2:
        return 0.0

    # Strip chapter prefixes to compare actual content
    content1 = strip_chapter_prefix(text1)
    content2 = strip_chapter_prefix(text2)

    # Find the longest common suffix of content1 and prefix of content2
    # We need to search for the maximum overlap, not just check sequentially
    max_possible = min(len(content1), len(content2))

    # Binary search or direct check from max down to find longest match
    for overlap_len in range(max_possible, 0, -1):
        if content1[-overlap_len:] == content2[:overlap_len]:
            return overlap_len / len(content1) if len(content1) > 0 else 0.0

    return 0.0


def verify_overlap_ratio(
    chroma_path: str = "data/chroma",
    expected_ratio: float = 0.2,
    tolerance: float = 0.05,
    sample_size: int = 100
) -> dict:
    """
    Verify that consecutive chunks have the expected overlap ratio.

    Args:
        chroma_path: Path to ChromaDB directory
        expected_ratio: Expected overlap ratio (default 0.2 = 20%)
        tolerance: Acceptable deviation (default 0.05 = ±5%)
        sample_size: Number of chunk pairs to sample

    Returns:
        Dictionary with test results and statistics
    """
    print("=" * 70)
    print("OVERLAP VERIFICATION TEST")
    print("=" * 70)
    print(f"Expected overlap ratio: {expected_ratio:.1%} ± {tolerance:.1%}")
    print(f"Sample size: {sample_size} consecutive chunk pairs\n")

    # Load ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("nlp_textbook")
    total_chunks = collection.count()

    print(f"Total chunks in database: {total_chunks:,}\n")

    # Get all chunks with their IDs (we'll need to sort them)
    all_results = collection.get(
        limit=total_chunks,
        include=['documents', 'metadatas']
    )

    # Group chunks by document
    doc_groups = {}
    for i, meta in enumerate(all_results['metadatas']):
        doc_id = meta.get('document_id', meta.get('id', 'unknown'))
        if doc_id not in doc_groups:
            doc_groups[doc_id] = []
        doc_groups[doc_id].append({
            'index': i,
            'chunk_index': meta.get('chunk_index', 0),
            'text': all_results['documents'][i]
        })

    # Sort chunks within each document by chunk_index
    for doc_id in doc_groups:
        doc_groups[doc_id].sort(key=lambda x: x['chunk_index'])

    # Collect consecutive chunk pairs from each document
    all_pairs = []
    for doc_id, chunks in doc_groups.items():
        for i in range(len(chunks) - 1):
            all_pairs.append({
                'doc_id': doc_id,
                'chunk1': chunks[i]['text'],
                'chunk2': chunks[i + 1]['text'],
                'index1': chunks[i]['chunk_index'],
                'index2': chunks[i + 1]['chunk_index']
            })

    print(f"Total consecutive chunk pairs available: {len(all_pairs)}\n")

    # Sample pairs
    sample_pairs = random.sample(all_pairs, min(sample_size, len(all_pairs)))

    # Calculate overlap for each pair
    overlap_ratios = []
    for pair in sample_pairs:
        ratio = get_overlap_ratio(pair['chunk1'], pair['chunk2'])
        overlap_ratios.append(ratio)

    # Calculate statistics
    avg_ratio = sum(overlap_ratios) / len(overlap_ratios)
    min_ratio = min(overlap_ratios)
    max_ratio = max(overlap_ratios)

    # Count how many are within tolerance
    within_tolerance = sum(
        1 for r in overlap_ratios
        if abs(r - expected_ratio) <= tolerance
    )

    # Determine pass/fail
    passed = abs(avg_ratio - expected_ratio) <= tolerance

    # Print results
    print("RESULTS:")
    print("-" * 70)
    print(f"Average overlap ratio:    {avg_ratio:.1%}")
    print(f"Minimum overlap ratio:    {min_ratio:.1%}")
    print(f"Maximum overlap ratio:    {max_ratio:.1%}")
    print(f"Chunks within tolerance:  {within_tolerance}/{len(overlap_ratios)} ({within_tolerance/len(overlap_ratios):.1%})")
    print()

    if passed:
        print(f"✅ PASS: Average overlap ({avg_ratio:.1%}) is within tolerance of expected ({expected_ratio:.1%})")
    else:
        print(f"❌ FAIL: Average overlap ({avg_ratio:.1%}) is outside tolerance of expected ({expected_ratio:.1%})")

    print("=" * 70)

    # Return detailed results
    return {
        'passed': passed,
        'average_ratio': avg_ratio,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio,
        'expected_ratio': expected_ratio,
        'tolerance': tolerance,
        'sample_size': len(overlap_ratios),
        'within_tolerance': within_tolerance,
        'total_chunks': total_chunks,
        'total_pairs': len(all_pairs)
    }


if __name__ == "__main__":
    import sys

    # Allow custom parameters from command line
    chroma_path = sys.argv[1] if len(sys.argv) > 1 else "data/chroma"
    expected_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

    results = verify_overlap_ratio(
        chroma_path=chroma_path,
        expected_ratio=expected_ratio,
        tolerance=0.05,
        sample_size=100
    )

    # Exit with appropriate code
    sys.exit(0 if results['passed'] else 1)
