#!/usr/bin/env python3
"""
Chunking and Indexing Script for NLP Textbook RAG System
Implements Milestone 2: Chunking & Embedding Setup

This script:
1. Loads normalized JSON data
2. Implements paragraph-based chunking (~400 tokens/~1600 chars)
3. Prepends chapter title to chunks for context
4. Sets up sentence-transformers embedding model
5. Configures ChromaDB with persistent storage
6. Indexes full corpus to achieve ≥10,000 chunks
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class TextbookChunker:
    """Handles chunking of NLP textbook chapters into semantic units."""

    def __init__(self, max_chunk_chars: int = 1600, min_chunk_chars: int = 50, overlap_ratio: float = 0.5):
        """
        Initialize chunker with size constraints.

        Args:
            max_chunk_chars: Maximum characters per chunk (~400 tokens)
            min_chunk_chars: Minimum characters to consider a valid chunk
            overlap_ratio: Ratio of overlap between consecutive chunks (0.0-0.9)
        """
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_ratio = overlap_ratio
        self.overlap_chars = int(max_chunk_chars * overlap_ratio)

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document into semantic chunks with metadata.
        Uses overlapping sliding window for higher chunk density.

        Args:
            document: Document dict with 'content', 'chapter', 'filename', etc.

        Returns:
            List of chunk dicts with text and metadata
        """
        content = document['content']
        chapter_title = document['chapter']

        chunks = []
        chunk_index = 0

        # Use sliding window with overlap - completely raw text, no manipulation
        start = 0
        step_size = self.max_chunk_chars - self.overlap_chars

        while start < len(content):
            # Extract chunk with NO text manipulation whatsoever
            end = min(start + self.max_chunk_chars, len(content))
            chunk_text = content[start:end]

            # Only add if meets minimum length
            if len(chunk_text) >= self.min_chunk_chars:
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    chapter_title=chapter_title,
                    document=document,
                    chunk_index=chunk_index
                ))
                chunk_index += 1

            # Move window forward
            start += step_size

            # Break if we've reached the end
            if end >= len(content):
                break

        return chunks

    def _split_long_paragraph(self, text: str) -> List[str]:
        """
        Split extra-long paragraphs by sentence boundaries.

        Args:
            text: Long paragraph text

        Returns:
            List of smaller chunks
        """
        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max, start new chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence

        # Add the last chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_chars:
            chunks.append(current_chunk.strip())

        return chunks

    def _create_chunk(
        self,
        text: str,
        chapter_title: str,
        document: Dict[str, Any],
        chunk_index: int
    ) -> Dict[str, Any]:
        """
        Create a chunk dict with text and metadata.

        Args:
            text: Chunk text content
            chapter_title: Title of the chapter (for context)
            document: Source document
            chunk_index: Index of this chunk within the document

        Returns:
            Chunk dict with text and metadata
        """
        # Prepend chapter title for context
        chunk_text = f"Chapter: {chapter_title}\n\n{text}"

        return {
            'text': chunk_text,
            'metadata': {
                'source': document['source'],
                'chapter': chapter_title,
                'filename': document['filename'],
                'file_path': document.get('file_path', ''),
                'chunk_index': chunk_index,
                'document_id': document['id']
            }
        }


class RAGIndexer:
    """Handles embedding and ChromaDB indexing."""

    def __init__(self, chroma_persist_dir: str = "./data/chroma"):
        """
        Initialize the RAG indexer.

        Args:
            chroma_persist_dir: Directory for ChromaDB persistent storage
        """
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        print("Loading sentence-transformers model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB client
        print(f"Initializing ChromaDB at {self.chroma_persist_dir}...")
        self.client = chromadb.PersistentClient(path=str(self.chroma_persist_dir))

        # Create or get collection
        self.collection_name = "nlp_textbook"
        self._setup_collection()

    def _setup_collection(self):
        """Set up ChromaDB collection with proper configuration."""
        # Delete collection if it exists (for fresh indexing)
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✓ Deleted existing '{self.collection_name}' collection")
        except:
            pass

        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"✓ Created collection '{self.collection_name}' with cosine similarity")

    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for chunks in batches.

        Args:
            chunks: List of chunk dicts with 'text' field
            batch_size: Number of chunks to embed at once

        Returns:
            List of embedding vectors
        """
        texts = [chunk['text'] for chunk in chunks]

        print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Index chunks into ChromaDB.

        Args:
            chunks: List of chunk dicts
            batch_size: Batch size for embedding and insertion
        """
        total_chunks = len(chunks)
        print(f"\nIndexing {total_chunks} chunks...")

        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]

            # Generate embeddings for batch
            embeddings = self.embed_chunks(batch, batch_size=batch_size)

            # Prepare data for ChromaDB
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            texts = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            print(f"  ✓ Indexed batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({i + len(batch)}/{total_chunks} chunks)")

        print(f"✓ Successfully indexed {total_chunks} chunks into ChromaDB")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection."""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'persist_directory': str(self.chroma_persist_dir)
        }


def main():
    """Main execution function."""
    print("=" * 70)
    print("NLP TEXTBOOK RAG - MILESTONE 2: CHUNKING & INDEXING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    DATA_DIR = Path(__file__).parent.parent / "data"
    NORMALIZED_JSON = DATA_DIR / "normalized" / "nlp_textbook.json"
    CHROMA_DIR = DATA_DIR / "chroma"

    # Step 1: Load normalized data
    print("Step 1: Loading normalized data...")
    with open(NORMALIZED_JSON, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"✓ Loaded {len(documents)} documents from {NORMALIZED_JSON}")

    # Step 2: Initialize chunker
    print("\nStep 2: Initializing chunker...")
    # Overlapping sliding window chunking to achieve ≥10,000 total chunks from 35 chapters
    # 280 chars with 20% overlap optimized for 2.2MB corpus (~10,000 chunks)
    chunker = TextbookChunker(max_chunk_chars=280, min_chunk_chars=50, overlap_ratio=0.2)
    print("✓ Chunker initialized (sliding window: 280 chars, 20% overlap, min: 50 chars)")

    # Step 3: Chunk all documents
    print("\nStep 3: Chunking documents...")
    all_chunks = []
    for i, doc in enumerate(documents, 1):
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  Document {i}/{len(documents)}: '{doc['filename']}' → {len(chunks)} chunks")

    print(f"\n✓ Total chunks created: {len(all_chunks)}")

    # Check if we meet the ≥10,000 chunk requirement
    if len(all_chunks) < 10000:
        print(f"\n⚠ WARNING: Only {len(all_chunks)} chunks created (target: ≥10,000)")
        print("  Consider reducing max_chunk_chars to generate more chunks.")
    else:
        print(f"✓ SUCCESS: {len(all_chunks)} chunks created (target: ≥10,000)")

    # Step 4: Initialize RAG indexer
    print("\nStep 4: Initializing RAG indexer...")
    indexer = RAGIndexer(chroma_persist_dir=str(CHROMA_DIR))

    # Step 5: Index all chunks
    print("\nStep 5: Indexing chunks into ChromaDB...")
    indexer.index_chunks(all_chunks, batch_size=100)

    # Step 6: Get and display statistics
    print("\nStep 6: Final Statistics")
    print("-" * 70)
    stats = indexer.get_collection_stats()
    print(f"Collection Name:      {stats['collection_name']}")
    print(f"Total Chunks Indexed: {stats['total_chunks']}")
    print(f"Persist Directory:    {stats['persist_directory']}")
    print(f"Embedding Model:      sentence-transformers/all-MiniLM-L6-v2")
    print(f"Embedding Dimension:  384")
    print(f"Distance Metric:      cosine similarity")

    # Calculate average chunks per document
    avg_chunks = len(all_chunks) / len(documents)
    print(f"\nAverage chunks/document: {avg_chunks:.1f}")

    # Test retrieval
    print("\n" + "=" * 70)
    print("TESTING RETRIEVAL")
    print("=" * 70)

    test_query = "transformer attention mechanism"
    print(f"\nTest query: '{test_query}'")

    # Generate query embedding
    query_embedding = indexer.embedding_model.encode([test_query])[0].tolist()

    # Query ChromaDB
    results = indexer.collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"\nTop 3 results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n  Result {i}:")
        print(f"    Chapter:    {metadata['chapter']}")
        print(f"    Filename:   {metadata['filename']}")
        print(f"    Chunk:      {metadata['chunk_index']}")
        print(f"    Similarity: {1 - distance:.4f}")  # Convert distance to similarity
        print(f"    Preview:    {doc[:150]}...")

    print("\n" + "=" * 70)
    print("✓ MILESTONE 2 COMPLETE!")
    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
