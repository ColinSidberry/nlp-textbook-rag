#!/usr/bin/env python3
"""
Semantic Similarity Test Runner
Milestone 5 - Task 1C

Tests vocabulary mismatch handling by running pairs of semantically similar queries
with different vocabulary through the RAG system.
"""

import json
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from rag_pipeline import NLPTextbookRAG


def run_semantic_tests():
    """Run semantic similarity test queries"""
    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY TEST - Task 1C")
    print("Testing vocabulary mismatch handling with embeddings")
    print("=" * 70)

    # Define test query pairs
    query_pairs = [
        {
            "pair_id": 1,
            "concept": "Word Embeddings",
            "query_a": {
                "type": "Exact Match",
                "text": "What are word embeddings?"
            },
            "query_b": {
                "type": "Vocabulary Mismatch",
                "text": "How do learned parameters encode relationships between linguistic tokens?"
            }
        },
        {
            "pair_id": 2,
            "concept": "Attention Mechanism",
            "query_a": {
                "type": "Exact Match",
                "text": "What is the attention mechanism?"
            },
            "query_b": {
                "type": "Vocabulary Mismatch",
                "text": "What mechanism allows neural models to weigh importance of different input elements?"
            }
        },
        {
            "pair_id": 3,
            "concept": "Backpropagation",
            "query_a": {
                "type": "Exact Match",
                "text": "How does backpropagation work in neural networks?"
            },
            "query_b": {
                "type": "Vocabulary Mismatch",
                "text": "How do neural networks propagate errors backwards through layers to update weights?"
            }
        },
        {
            "pair_id": 4,
            "concept": "Transformer Architecture",
            "query_a": {
                "type": "Exact Match",
                "text": "How do transformers work?"
            },
            "query_b": {
                "type": "Vocabulary Mismatch",
                "text": "What neural architecture processes sequences in parallel using self-attention?"
            }
        },
        {
            "pair_id": 5,
            "concept": "N-gram Language Models",
            "query_a": {
                "type": "Exact Match",
                "text": "What are n-grams?"
            },
            "query_b": {
                "type": "Vocabulary Mismatch",
                "text": "What are contiguous sequences of words used for statistical language modeling?"
            }
        }
    ]

    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    rag = NLPTextbookRAG(temperature=0.7, max_tokens=500, top_k=5)
    print("✓ RAG pipeline initialized\n")

    # Run tests
    results = []

    for pair in query_pairs:
        print(f"\n{'=' * 70}")
        print(f"PAIR {pair['pair_id']}: {pair['concept']}")
        print(f"{'=' * 70}")

        # Run Query A (exact match)
        print(f"\n[Query A - {pair['query_a']['type']}]")
        print(f"Question: \"{pair['query_a']['text']}\"")
        print("Processing...")

        start_time = time.time()
        result_a = rag.query(pair['query_a']['text'], verbose=False)
        query_time_a = time.time() - start_time

        result_a['query_time'] = query_time_a
        result_a['pair_id'] = pair['pair_id']
        result_a['concept'] = pair['concept']
        result_a['query_type'] = pair['query_a']['type']

        # Calculate average similarity
        similarities_a = [chunk['similarity'] for chunk in result_a['retrieved_chunks']]
        result_a['avg_similarity'] = sum(similarities_a) / len(similarities_a) if similarities_a else 0
        result_a['max_similarity'] = max(similarities_a) if similarities_a else 0
        result_a['min_similarity'] = min(similarities_a) if similarities_a else 0

        print(f"✓ Completed in {query_time_a:.2f}s")
        print(f"  Top chapter: {result_a['retrieved_chunks'][0]['chapter']}")
        print(f"  Top similarity: {result_a['max_similarity']:.3f}")
        print(f"  Avg similarity: {result_a['avg_similarity']:.3f}")

        results.append(result_a)

        # Run Query B (vocabulary mismatch)
        print(f"\n[Query B - {pair['query_b']['type']}]")
        print(f"Question: \"{pair['query_b']['text']}\"")
        print("Processing...")

        start_time = time.time()
        result_b = rag.query(pair['query_b']['text'], verbose=False)
        query_time_b = time.time() - start_time

        result_b['query_time'] = query_time_b
        result_b['pair_id'] = pair['pair_id']
        result_b['concept'] = pair['concept']
        result_b['query_type'] = pair['query_b']['type']

        # Calculate average similarity
        similarities_b = [chunk['similarity'] for chunk in result_b['retrieved_chunks']]
        result_b['avg_similarity'] = sum(similarities_b) / len(similarities_b) if similarities_b else 0
        result_b['max_similarity'] = max(similarities_b) if similarities_b else 0
        result_b['min_similarity'] = min(similarities_b) if similarities_b else 0

        print(f"✓ Completed in {query_time_b:.2f}s")
        print(f"  Top chapter: {result_b['retrieved_chunks'][0]['chapter']}")
        print(f"  Top similarity: {result_b['max_similarity']:.3f}")
        print(f"  Avg similarity: {result_b['avg_similarity']:.3f}")

        results.append(result_b)

        # Compare results
        print(f"\n[Pair Comparison]")
        chapters_a = set(chunk['chapter'] for chunk in result_a['retrieved_chunks'])
        chapters_b = set(chunk['chapter'] for chunk in result_b['retrieved_chunks'])
        overlap = chapters_a & chapters_b

        print(f"  Chapters retrieved by A: {chapters_a}")
        print(f"  Chapters retrieved by B: {chapters_b}")
        print(f"  Overlap: {len(overlap)}/{max(len(chapters_a), len(chapters_b))} chapters")
        print(f"  Similarity score difference: {abs(result_a['max_similarity'] - result_b['max_similarity']):.3f}")

        if len(overlap) > 0:
            print(f"  ✓ PASS: Embeddings successfully matched different vocabulary to same concept")
        else:
            print(f"  ⚠ WARN: No chapter overlap - vocabulary mismatch challenge")

    # Save results
    output_file = Path(__file__).parent / "semantic_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Total queries processed: {len(results)}")
    print(f"✓ Test pairs evaluated: {len(query_pairs)}")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    results = run_semantic_tests()
