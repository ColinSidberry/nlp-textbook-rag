#!/usr/bin/env python3
"""
Auto-indexing module for ChromaDB
Checks if ChromaDB exists and rebuilds it if missing
"""

import os
import sys
from pathlib import Path
import json
import chromadb
from chromadb.config import Settings


def check_chroma_exists(chroma_path: str = None) -> bool:
    """
    Check if ChromaDB collection exists and is valid

    Args:
        chroma_path: Path to ChromaDB storage (default: ../data/chroma)

    Returns:
        True if ChromaDB exists and is valid, False otherwise
    """
    if chroma_path is None:
        chroma_path = str(Path(__file__).parent.parent / "data" / "chroma")

    # Check if directory exists
    if not os.path.exists(chroma_path):
        return False

    # Check if it's a valid ChromaDB
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name="nlp_textbook")
        count = collection.count()

        # Verify it has a reasonable number of chunks
        if count < 1000:  # Should have 11k+ chunks
            print(f"Warning: ChromaDB exists but only has {count} chunks (expected 11k+)")
            return False

        return True
    except Exception as e:
        print(f"ChromaDB validation failed: {e}")
        return False


def rebuild_index(chroma_path: str = None, normalized_data_path: str = None) -> bool:
    """
    Rebuild ChromaDB index from normalized data

    Args:
        chroma_path: Path to ChromaDB storage
        normalized_data_path: Path to normalized JSON data

    Returns:
        True if successful, False otherwise
    """
    try:
        # Import here to avoid circular dependencies
        from chunk_and_index import TextbookChunker, RAGIndexer

        # Set default paths
        if chroma_path is None:
            chroma_path = str(Path(__file__).parent.parent / "data" / "chroma")

        if normalized_data_path is None:
            normalized_data_path = str(Path(__file__).parent.parent / "data" / "normalized" / "nlp_textbook.json")

        print("=" * 70)
        print("REBUILDING CHROMADB INDEX")
        print("=" * 70)

        # Load normalized data
        print(f"\n1. Loading data from: {normalized_data_path}")
        if not os.path.exists(normalized_data_path):
            print(f"ERROR: Normalized data not found at {normalized_data_path}")
            return False

        with open(normalized_data_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"✓ Loaded {len(documents)} documents")

        # Initialize chunker (same settings as original indexing)
        print("\n2. Initializing chunker...")
        chunker = TextbookChunker(max_chunk_chars=100, min_chunk_chars=30, overlap_ratio=0.5)
        print("✓ Chunker initialized")

        # Chunk all documents
        print("\n3. Chunking documents...")
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        print(f"✓ Created {len(all_chunks)} chunks")

        # Initialize RAG indexer
        print("\n4. Initializing RAG indexer...")
        indexer = RAGIndexer(chroma_persist_dir=chroma_path)
        print("✓ RAG indexer initialized")

        # Index chunks
        print("\n5. Indexing chunks to ChromaDB...")
        indexer.index_chunks(all_chunks, batch_size=100)

        # Verify
        stats = indexer.get_collection_stats()
        print(f"\n✓ Indexing complete!")
        print(f"  Total chunks indexed: {stats['total_chunks']}")
        print(f"  Persist directory: {stats['persist_directory']}")

        print("\n" + "=" * 70)
        print("✓ CHROMADB REBUILT SUCCESSFULLY!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\nERROR: Failed to rebuild index: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def ensure_chroma_ready(chroma_path: str = None, force_rebuild: bool = False) -> bool:
    """
    Ensure ChromaDB is ready (rebuild if necessary)

    Args:
        chroma_path: Path to ChromaDB storage
        force_rebuild: Force rebuild even if ChromaDB exists

    Returns:
        True if ChromaDB is ready, False otherwise
    """
    if chroma_path is None:
        chroma_path = str(Path(__file__).parent.parent / "data" / "chroma")

    # Check if rebuild is needed
    if force_rebuild:
        print("Force rebuild requested...")
        return rebuild_index(chroma_path)

    # Check if ChromaDB exists
    if check_chroma_exists(chroma_path):
        print(f"✓ ChromaDB ready at: {chroma_path}")
        return True

    # ChromaDB missing or invalid - rebuild it
    print(f"ChromaDB not found or invalid at: {chroma_path}")
    print("Starting automatic rebuild...")
    return rebuild_index(chroma_path)


if __name__ == "__main__":
    # Can be run standalone to rebuild index
    import sys
    force = "--force" in sys.argv

    success = ensure_chroma_ready(force_rebuild=force)
    sys.exit(0 if success else 1)
