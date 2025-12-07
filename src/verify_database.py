#!/usr/bin/env python3
"""
Database Verification Script
Proves ChromaDB has 10k+ entries without needing to commit the database files
"""

from pathlib import Path
import chromadb

def verify_database():
    """Verify ChromaDB entry count"""

    # Path to ChromaDB
    chroma_path = Path(__file__).parent.parent / "data" / "chroma"

    if not chroma_path.exists():
        print("❌ ChromaDB not found at:", chroma_path)
        print("Run src/chunk_and_index.py to create the database")
        return

    # Connect to ChromaDB
    print(f"Connecting to ChromaDB at: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Get collection
    try:
        collection = client.get_collection(name="nlp_textbook")
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        return

    # Count entries
    count = collection.count()

    # Display results
    print("\n" + "="*60)
    print("DATABASE VERIFICATION RESULTS")
    print("="*60)
    print(f"Collection name: nlp_textbook")
    print(f"Total entries:   {count:,}")
    print(f"Requirement:     10,000 entries minimum")
    print(f"Status:          {'✅ PASS' if count >= 10000 else '❌ FAIL'}")
    print(f"Exceeds by:      {count - 10000:,} entries ({((count - 10000) / 10000 * 100):.1f}%)")
    print("="*60)

    # Get sample entry to show structure
    results = collection.get(limit=1, include=["documents", "metadatas"])

    if results['documents']:
        print("\nSample entry:")
        print(f"  Document: {results['documents'][0][:100]}...")
        print(f"  Metadata: {results['metadatas'][0]}")

    print("\n✅ Verification complete!")
    return count

if __name__ == "__main__":
    verify_database()
