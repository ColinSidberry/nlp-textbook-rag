#!/usr/bin/env python3
"""
Debug script to investigate why "what is RAG?" gets declined
"""

from rag_pipeline import NLPTextbookRAG

def debug_query(question):
    """Debug a specific query to see retrieval details"""
    print("="*70)
    print(f"DEBUGGING QUERY: {question}")
    print("="*70)

    rag = NLPTextbookRAG(temperature=0.3, max_tokens=500, top_k=5)

    # Get retrieval results
    documents, metadatas = rag.retrieve(question, k=10)

    print(f"\nRetrieved {len(documents)} chunks:")
    print("-"*70)

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        print(f"\n[{i}] Similarity: {meta['similarity']:.4f}")
        print(f"    Chapter: {meta['chapter']}")
        print(f"    Filename: {meta['filename']}")
        print(f"    Preview: {doc[:150]}...")

    # Calculate average similarity
    if metadatas:
        avg_sim = sum(m['similarity'] for m in metadatas) / len(metadatas)
        print(f"\n{'='*70}")
        print(f"Average Similarity: {avg_sim:.4f}")
        print(f"Threshold: 0.5")
        print(f"Would be declined: {avg_sim < 0.5}")
        print("="*70)

    # Now try the full query
    print("\n" + "="*70)
    print("FULL QUERY RESULT:")
    print("="*70)
    result = rag.query(question, verbose=True)
    print(f"\nAnswer: {result['answer']}")

if __name__ == "__main__":
    # Test the problematic query
    debug_query("What is RAG?")

    # Also test variations
    print("\n\n")
    debug_query("What is retrieval-augmented generation?")

    print("\n\n")
    debug_query("Explain RAG in NLP")
