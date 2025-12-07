#!/usr/bin/env python3
"""
Milestone 5 Comprehensive Testing Script

Tests all 20 diverse queries and collects:
- Generated answers
- Retrieved chunks with similarity scores
- Citations
- Query timing
- Retrieval relevance ratings
- Answer quality ratings
"""

import sys
import time
from pathlib import Path
from rag_pipeline import NLPTextbookRAG
import json


def load_test_queries(filepath: str) -> list:
    """Load test queries from file"""
    queries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract query text (remove [Type] prefix)
                if ']' in line:
                    query_type, query_text = line.split(']', 1)
                    query_type = query_type.strip('[').strip()
                    query_text = query_text.strip()
                    queries.append({
                        'type': query_type,
                        'text': query_text
                    })
    return queries


def run_comprehensive_test(rag: NLPTextbookRAG, query_data: dict, query_num: int, total: int):
    """Run a single query and collect comprehensive metrics"""
    print(f"\n{'='*70}")
    print(f"QUERY {query_num}/{total}: [{query_data['type']}]")
    print(f"{'='*70}")
    print(f"Question: {query_data['text']}")

    # Time the query
    start_time = time.time()
    result = rag.query(query_data['text'], verbose=False)
    end_time = time.time()
    query_time = end_time - start_time

    # Add timing to result
    result['query_time'] = query_time
    result['query_type'] = query_data['type']

    # Calculate average similarity
    similarities = [chunk['similarity'] for chunk in result['retrieved_chunks']]
    result['avg_similarity'] = sum(similarities) / len(similarities) if similarities else 0.0
    result['max_similarity'] = max(similarities) if similarities else 0.0
    result['min_similarity'] = min(similarities) if similarities else 0.0

    # Print summary
    print(f"\n✓ Query completed in {query_time:.2f} seconds")
    print(f"  Answer length: {len(result['answer'])} characters")
    print(f"  Citations: {len(result['citations'])} sources")
    print(f"  Chunks retrieved: {len(result['retrieved_chunks'])}")
    print(f"  Avg similarity: {result['avg_similarity']:.3f}")
    print(f"  Max similarity: {result['max_similarity']:.3f}")
    print(f"  Top chapter: {result['retrieved_chunks'][0]['chapter'] if result['retrieved_chunks'] else 'N/A'}")

    return result


def rate_retrieval_relevance(result: dict) -> int:
    """
    Rate retrieval relevance on 1-5 scale based on similarity scores
    5: Excellent (avg similarity > 0.80)
    4: Good (avg similarity 0.70-0.80)
    3: Fair (avg similarity 0.60-0.70)
    2: Poor (avg similarity 0.50-0.60)
    1: Very poor (avg similarity < 0.50)
    """
    avg_sim = result['avg_similarity']
    if avg_sim >= 0.80:
        return 5
    elif avg_sim >= 0.70:
        return 4
    elif avg_sim >= 0.60:
        return 3
    elif avg_sim >= 0.50:
        return 2
    else:
        return 1


def rate_answer_quality(result: dict) -> int:
    """
    Rate answer quality on 1-5 scale based on multiple factors
    5: Excellent (>1000 chars, coherent, has citations)
    4: Good (700-1000 chars, coherent, has citations)
    3: Fair (400-700 chars, somewhat coherent)
    2: Poor (<400 chars or lacks citations)
    1: Very poor (very short or no answer)
    """
    answer_len = len(result['answer'])
    has_citations = len(result['citations']) > 0

    if answer_len > 1000 and has_citations:
        return 5
    elif answer_len >= 700 and has_citations:
        return 4
    elif answer_len >= 400:
        return 3
    elif answer_len >= 200:
        return 2
    else:
        return 1


def save_results_json(results: list, filepath: str):
    """Save results to JSON file for later analysis"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filepath}")


def print_summary_statistics(results: list):
    """Print comprehensive summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Timing stats
    query_times = [r['query_time'] for r in results]
    avg_time = sum(query_times) / len(query_times)
    min_time = min(query_times)
    max_time = max(query_times)

    # Similarity stats
    avg_similarities = [r['avg_similarity'] for r in results]
    overall_avg_sim = sum(avg_similarities) / len(avg_similarities)

    # Citation stats
    total_citations = sum(len(r['citations']) for r in results)
    avg_citations = total_citations / len(results)

    # Ratings
    relevance_ratings = [rate_retrieval_relevance(r) for r in results]
    quality_ratings = [rate_answer_quality(r) for r in results]
    avg_relevance = sum(relevance_ratings) / len(relevance_ratings)
    avg_quality = sum(quality_ratings) / len(quality_ratings)

    # Chapters coverage
    all_chapters = set()
    for r in results:
        for chunk in r['retrieved_chunks']:
            all_chapters.add(chunk['chapter'])

    print(f"\n📊 QUERY PERFORMANCE:")
    print(f"  Total queries tested: {len(results)}")
    print(f"  Average query time: {avg_time:.2f}s")
    print(f"  Min query time: {min_time:.2f}s")
    print(f"  Max query time: {max_time:.2f}s")

    print(f"\n📚 RETRIEVAL QUALITY:")
    print(f"  Average similarity score: {overall_avg_sim:.3f}")
    print(f"  Average relevance rating: {avg_relevance:.2f}/5")
    print(f"  Unique chapters referenced: {len(all_chapters)}/7")
    print(f"  Chapters: {', '.join(sorted(all_chapters))}")

    print(f"\n✍️  ANSWER QUALITY:")
    print(f"  Average answer quality rating: {avg_quality:.2f}/5")
    print(f"  Average citations per query: {avg_citations:.1f}")
    print(f"  Total citations generated: {total_citations}")

    # Query type breakdown
    print(f"\n📋 QUERY TYPE BREAKDOWN:")
    query_types = {}
    for r in results:
        qtype = r['query_type']
        if qtype not in query_types:
            query_types[qtype] = []
        query_types[qtype].append(r)

    for qtype, type_results in sorted(query_types.items()):
        avg_type_sim = sum(r['avg_similarity'] for r in type_results) / len(type_results)
        print(f"  {qtype}: {len(type_results)} queries, avg similarity: {avg_type_sim:.3f}")

    print("\n" + "="*70)


def identify_best_demo_queries(results: list, top_n: int = 5):
    """Identify best queries for demo based on multiple criteria"""
    print("\n" + "="*70)
    print(f"TOP {top_n} DEMO QUERY CANDIDATES")
    print("="*70)

    # Score each query
    scored_queries = []
    for r in results:
        # Scoring criteria:
        # - High similarity (weight: 3)
        # - Good answer length (weight: 2)
        # - Fast query time (weight: 1)
        # - Has citations (weight: 2)

        sim_score = r['avg_similarity'] * 3
        length_score = min(len(r['answer']) / 1000, 2.0) * 2  # Normalize to max 2.0
        time_score = max(0, (30 - r['query_time']) / 30) * 1  # Faster = better
        citation_score = min(len(r['citations']) / 2, 2.0) * 2  # Normalize to max 2.0

        total_score = sim_score + length_score + time_score + citation_score

        scored_queries.append({
            'result': r,
            'score': total_score,
            'relevance': rate_retrieval_relevance(r),
            'quality': rate_answer_quality(r)
        })

    # Sort by score
    scored_queries.sort(key=lambda x: x['score'], reverse=True)

    # Print top N
    for i, sq in enumerate(scored_queries[:top_n], 1):
        r = sq['result']
        print(f"\n#{i} (Score: {sq['score']:.2f})")
        print(f"  Query: {r['question']}")
        print(f"  Type: [{r['query_type']}]")
        print(f"  Similarity: {r['avg_similarity']:.3f}")
        print(f"  Time: {r['query_time']:.2f}s")
        print(f"  Citations: {len(r['citations'])}")
        print(f"  Top chapter: {r['retrieved_chunks'][0]['chapter']}")
        print(f"  Relevance: {sq['relevance']}/5, Quality: {sq['quality']}/5")

    print("\n" + "="*70)
    return [sq['result'] for sq in scored_queries[:top_n]]


def main():
    """Run Milestone 5 comprehensive testing"""
    print("\n" + "="*70)
    print("MILESTONE 5 - COMPREHENSIVE QUERY TESTING")
    print("="*70)

    # Load test queries
    queries_file = Path(__file__).parent.parent / "test_queries.txt"
    print(f"\nLoading test queries from: {queries_file}")

    try:
        test_queries = load_test_queries(str(queries_file))
        print(f"✓ Loaded {len(test_queries)} test queries")
    except Exception as e:
        print(f"✗ ERROR: Failed to load test queries: {e}")
        sys.exit(1)

    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    try:
        rag = NLPTextbookRAG(
            temperature=0.7,
            max_tokens=500,
            top_k=5
        )
        print("✓ RAG pipeline initialized")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize RAG pipeline: {e}")
        sys.exit(1)

    # Run all queries
    print(f"\n{'='*70}")
    print(f"RUNNING {len(test_queries)} QUERIES")
    print(f"{'='*70}")

    results = []
    for i, query_data in enumerate(test_queries, 1):
        try:
            result = run_comprehensive_test(rag, query_data, i, len(test_queries))

            # Add ratings
            result['relevance_rating'] = rate_retrieval_relevance(result)
            result['quality_rating'] = rate_answer_quality(result)

            results.append(result)
        except Exception as e:
            print(f"\n✗ ERROR on query {i}: {e}")
            continue

    # Save results
    results_file = Path(__file__).parent.parent / "test_results.json"
    save_results_json(results, str(results_file))

    # Print summary statistics
    print_summary_statistics(results)

    # Identify best demo queries
    best_demos = identify_best_demo_queries(results, top_n=5)

    # Final message
    print("\n" + "="*70)
    print("✓ MILESTONE 5 TESTING COMPLETE")
    print("="*70)
    print(f"\n  Queries tested: {len(results)}/{len(test_queries)}")
    print(f"  Results saved to: {results_file}")
    print(f"  Next step: Create test_results.md from results")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
