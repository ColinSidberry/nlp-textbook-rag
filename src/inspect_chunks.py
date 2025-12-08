#!/usr/bin/env python3
"""
Inspect specific chunks from ChromaDB to verify overlap.
"""

import chromadb
from pathlib import Path


def inspect_chunk_pair(collection, chunk_id1, chunk_id2):
    """Show detailed view of two chunks and their overlap."""

    # Get the chunks
    result1 = collection.get(ids=[f"chunk_{chunk_id1}"], include=["documents", "metadatas"])
    result2 = collection.get(ids=[f"chunk_{chunk_id2}"], include=["documents", "metadatas"])

    if not result1['documents'] or not result2['documents']:
        print("One or both chunks not found!")
        return

    chunk1_full = result1['documents'][0]
    chunk2_full = result2['documents'][0]
    meta1 = result1['metadatas'][0]
    meta2 = result2['metadatas'][0]

    print(f"\n{'='*70}")
    print(f"CHUNK {chunk_id1}")
    print(f"{'='*70}")
    print(f"Document: {meta1['filename']}")
    print(f"Chapter: {meta1['chapter']}")
    print(f"Chunk index in doc: {meta1['chunk_index']}")
    print(f"\nFull text length: {len(chunk1_full)} chars")
    print(f"\nFull text:\n{chunk1_full}")

    print(f"\n{'='*70}")
    print(f"CHUNK {chunk_id2}")
    print(f"{'='*70}")
    print(f"Document: {meta2['filename']}")
    print(f"Chapter: {meta2['chapter']}")
    print(f"Chunk index in doc: {meta2['chunk_index']}")
    print(f"\nFull text length: {len(chunk2_full)} chars")
    print(f"\nFull text:\n{chunk2_full}")

    # Check if they're from the same document and consecutive
    same_doc = meta1['document_id'] == meta2['document_id']
    consecutive = meta1['chunk_index'] + 1 == meta2['chunk_index']

    print(f"\n{'='*70}")
    print(f"OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"Same document: {same_doc}")
    print(f"Consecutive chunks: {consecutive}")

    if not same_doc:
        print("\n⚠ These chunks are from DIFFERENT documents - no overlap expected!")
        return

    if not consecutive:
        print("\n⚠ These chunks are NOT consecutive within the document - no overlap expected!")
        return

    # Extract content portions (remove chapter prefix)
    def get_content(full_text):
        if '\n\n' in full_text:
            return full_text.split('\n\n', 1)[1]
        return full_text

    content1 = get_content(chunk1_full)
    content2 = get_content(chunk2_full)

    print(f"\nContent 1 length (without chapter prefix): {len(content1)} chars")
    print(f"Content 2 length (without chapter prefix): {len(content2)} chars")

    # Find overlap between end of content1 and start of content2
    max_overlap = min(len(content1), len(content2))
    overlap_found = 0

    for overlap_len in range(max_overlap, 0, -1):
        if content1[-overlap_len:] == content2[:overlap_len]:
            overlap_found = overlap_len
            break

    print(f"\n{'='*70}")
    print(f"OVERLAP DETECTED: {overlap_found} characters")
    print(f"{'='*70}")

    if overlap_found > 0:
        overlap_text = content1[-overlap_found:]
        print(f"\nOverlapping text ({overlap_found} chars):")
        print(f"[{overlap_text}]")

        overlap_ratio = overlap_found / len(content1) if len(content1) > 0 else 0
        print(f"\nOverlap ratio: {overlap_ratio:.1%}")

        expected_overlap = 56  # 20% of 280
        print(f"Expected overlap: {expected_overlap} chars (20%)")

        if abs(overlap_found - expected_overlap) <= 5:
            print("✓ PASS: Overlap matches expected value")
        else:
            print(f"✗ FAIL: Overlap doesn't match (expected ~{expected_overlap})")
    else:
        print("\n✗ FAIL: NO OVERLAP FOUND!")
        print("\nEnd of chunk 1 content (last 100 chars):")
        print(f"[{content1[-100:]}]")
        print("\nStart of chunk 2 content (first 100 chars):")
        print(f"[{content2[:100]}]")


def main():
    # Connect to ChromaDB
    DATA_DIR = Path(__file__).parent.parent / "data"
    CHROMA_DIR = DATA_DIR / "chroma"

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("nlp_textbook")

    print("="*70)
    print("CHUNK OVERLAP INSPECTOR")
    print("="*70)

    # Inspect chunks 2 and 3 (the ones the user mentioned)
    print("\nInspecting chunks 2 and 3...")
    inspect_chunk_pair(collection, 2, 3)

    # Also check chunks 0 and 1 for comparison
    print("\n\n" + "="*70)
    print("For comparison, also checking chunks 0 and 1...")
    inspect_chunk_pair(collection, 0, 1)


if __name__ == "__main__":
    main()
