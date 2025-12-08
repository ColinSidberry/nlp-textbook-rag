#!/usr/bin/env python3
"""
Edge Case Testing for RAG System
Tests unusual inputs and boundary conditions
"""

from rag_pipeline import NLPTextbookRAG
import json
from datetime import datetime
from pathlib import Path


class EdgeCaseTester:
    """Test edge cases for RAG robustness"""

    def __init__(self, rag: NLPTextbookRAG):
        self.rag = rag
        self.results = []

    def test_case(self, name, query, expected_behavior, should_answer=True):
        """Run a single test case"""
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print(f"Query: '{query}'")
        print(f"Expected: {expected_behavior}")
        print('-'*70)

        try:
            result = self.rag.query(query, verbose=False)
            answer = result['answer']

            print(f"Answer: {answer[:200]}...")

            # Basic checks
            has_answer = len(answer) > 20
            is_declined = any(phrase in answer.lower() for phrase in [
                'outside the scope',
                'cannot answer',
                'don\'t have enough'
            ])

            passed = has_answer and (should_answer == (not is_declined))

            self.results.append({
                'name': name,
                'query': query,
                'expected_behavior': expected_behavior,
                'should_answer': should_answer,
                'answer': answer,
                'is_declined': is_declined,
                'passed': passed
            })

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}")

            return passed

        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            self.results.append({
                'name': name,
                'query': query,
                'error': str(e),
                'passed': False
            })
            return False

    def run_all_tests(self):
        """Run comprehensive edge case tests"""
        print("\n" + "#"*70)
        print("RAG EDGE CASE TEST SUITE")
        print("#"*70)

        passed = 0
        total = 0

        # 1. Multi-acronym queries
        print("\n" + "="*70)
        print("CATEGORY 1: MULTI-ACRONYM QUERIES")
        print("="*70)

        total += 1
        if self.test_case(
            "Multi-acronym query",
            "How does BERT use transformers?",
            "Should expand both acronyms and answer",
            should_answer=True
        ):
            passed += 1

        total += 1
        if self.test_case(
            "Mixed acronyms",
            "Compare RNN and LSTM",
            "Should expand both and provide comparison",
            should_answer=True
        ):
            passed += 1

        # 2. Very short/vague queries
        print("\n" + "="*70)
        print("CATEGORY 2: SHORT/VAGUE QUERIES")
        print("="*70)

        total += 1
        if self.test_case(
            "Single word query",
            "transformers",
            "Should handle single word (may be ambiguous)",
            should_answer=True
        ):
            passed += 1

        total += 1
        if self.test_case(
            "Two word query",
            "neural networks",
            "Should provide general overview",
            should_answer=True
        ):
            passed += 1

        # 3. Case sensitivity
        print("\n" + "="*70)
        print("CATEGORY 3: CASE SENSITIVITY")
        print("="*70)

        total += 1
        if self.test_case(
            "Lowercase acronym",
            "what is rag?",
            "Should still expand RAG correctly",
            should_answer=True
        ):
            passed += 1

        total += 1
        if self.test_case(
            "Mixed case",
            "What is Bert?",
            "Should recognize BERT",
            should_answer=True
        ):
            passed += 1

        # 4. Typos and misspellings
        print("\n" + "="*70)
        print("CATEGORY 4: TYPOS/MISSPELLINGS")
        print("="*70)

        total += 1
        if self.test_case(
            "Common typo",
            "What are tranformers?",
            "May still retrieve relevant chunks (embedding similarity)",
            should_answer=True  # Might still work due to similarity
        ):
            passed += 1

        total += 1
        if self.test_case(
            "Spelling variant",
            "What is an embeding?",
            "May still work with similarity matching",
            should_answer=True
        ):
            passed += 1

        # 5. Empty/whitespace queries
        print("\n" + "="*70)
        print("CATEGORY 5: EMPTY/INVALID QUERIES")
        print("="*70)

        total += 1
        if self.test_case(
            "Empty string",
            "",
            "Should handle gracefully",
            should_answer=False
        ):
            passed += 1

        total += 1
        if self.test_case(
            "Only whitespace",
            "   ",
            "Should handle gracefully",
            should_answer=False
        ):
            passed += 1

        # 6. Very long queries
        print("\n" + "="*70)
        print("CATEGORY 6: VERY LONG QUERIES")
        print("="*70)

        total += 1
        if self.test_case(
            "Very long query",
            "I'm writing a research paper about natural language processing and I need to understand how transformer models work, specifically the attention mechanism, and I'm wondering if you could explain in detail how the multi-head attention works and what are the query, key, and value matrices?",
            "Should handle long query and extract main question",
            should_answer=True
        ):
            passed += 1

        # 7. Comparison questions
        print("\n" + "="*70)
        print("CATEGORY 7: COMPARISON QUESTIONS")
        print("="*70)

        total += 1
        if self.test_case(
            "Comparison",
            "What's the difference between word embeddings and contextualized embeddings?",
            "Should provide comparison",
            should_answer=True
        ):
            passed += 1

        # 8. Meta questions about the textbook
        print("\n" + "="*70)
        print("CATEGORY 8: META QUESTIONS")
        print("="*70)

        total += 1
        if self.test_case(
            "About the book",
            "Who wrote this textbook?",
            "May appear in context (Jurafsky & Martin mentioned)",
            should_answer=True  # Authors mentioned in chapter titles
        ):
            passed += 1

        total += 1
        if self.test_case(
            "Chapter listing",
            "What chapters are in this book?",
            "May decline or list based on retrieved context",
            should_answer=False  # Unlikely to have chapter listing
        ):
            passed += 1

        # 9. Implementation/code questions
        print("\n" + "="*70)
        print("CATEGORY 9: IMPLEMENTATION QUESTIONS")
        print("="*70)

        total += 1
        if self.test_case(
            "Code implementation",
            "How do I implement a transformer in Python?",
            "Should focus on concepts, not code (textbook is conceptual)",
            should_answer=True  # May describe architecture
        ):
            passed += 1

        # 10. Temporal/historical questions
        print("\n" + "="*70)
        print("CATEGORY 10: TEMPORAL/HISTORICAL QUESTIONS")
        print("="*70)

        total += 1
        if self.test_case(
            "When question",
            "When was BERT invented?",
            "Only answer if date appears in textbook",
            should_answer=True  # Dates might be in context
        ):
            passed += 1

        # 11. Special characters
        print("\n" + "="*70)
        print("CATEGORY 11: SPECIAL CHARACTERS")
        print("="*70)

        total += 1
        if self.test_case(
            "Angle brackets",
            "What is the <mask> token?",
            "Should handle special chars",
            should_answer=True
        ):
            passed += 1

        # 12. Negative questions
        print("\n" + "="*70)
        print("CATEGORY 12: NEGATIVE QUESTIONS")
        print("="*70)

        total += 1
        if self.test_case(
            "Limitation question",
            "Why can't RNNs handle long sequences well?",
            "Should explain limitations",
            should_answer=True
        ):
            passed += 1

        # 13. Multiple questions in one
        print("\n" + "="*70)
        print("CATEGORY 13: MULTIPLE QUESTIONS")
        print("="*70)

        total += 1
        if self.test_case(
            "Multiple questions",
            "What are transformers and how do they differ from RNNs?",
            "Should answer both parts",
            should_answer=True
        ):
            passed += 1

        # 14. Ambiguous queries
        print("\n" + "="*70)
        print("CATEGORY 14: AMBIGUOUS QUERIES")
        print("="*70)

        total += 1
        if self.test_case(
            "Ambiguous term",
            "What is attention?",
            "Should provide NLP context (attention mechanism)",
            should_answer=True
        ):
            passed += 1

        # Print summary
        print("\n" + "="*70)
        print("EDGE CASE TEST SUMMARY")
        print("="*70)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed} ✓")
        print(f"Failed:       {total - passed} ✗")
        print(f"Pass Rate:    {passed / total * 100:.1f}%")

        # Identify failing categories
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)

        failures = [r for r in self.results if not r.get('passed', False)]
        if failures:
            print("\nFailing test cases that need attention:")
            for f in failures:
                print(f"\n- {f['name']}")
                print(f"  Query: {f['query']}")
                if 'error' in f:
                    print(f"  Error: {f['error']}")
        else:
            print("\n✓ All edge cases handled well!")

        # Save results
        output_dir = Path(__file__).parent.parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"edge_case_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': total - passed,
                    'pass_rate': passed / total
                },
                'results': self.results
            }, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")

        return passed, total


def main():
    """Run edge case tests"""
    print("Initializing RAG pipeline...")
    rag = NLPTextbookRAG(
        temperature=0.3,
        max_tokens=500,
        top_k=5
    )

    tester = EdgeCaseTester(rag)
    passed, total = tester.run_all_tests()

    if passed < total:
        print(f"\n⚠ {total - passed} edge cases need attention")
        exit(1)
    else:
        print("\n✓ All edge cases handled!")
        exit(0)


if __name__ == "__main__":
    main()
