#!/usr/bin/env python3
"""
Comprehensive RAG Quality Testing Suite
Tests for hallucination, out-of-scope handling, and other RAG flaws
"""

from rag_pipeline import NLPTextbookRAG
import json
from datetime import datetime
from pathlib import Path


class RAGTester:
    """Test suite for RAG quality and correctness"""

    def __init__(self, rag: NLPTextbookRAG):
        self.rag = rag
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }

    def test_out_of_scope_questions(self):
        """Test that out-of-scope questions are properly declined"""
        print("\n" + "="*70)
        print("TEST 1: OUT-OF-SCOPE QUESTION HANDLING")
        print("="*70)

        out_of_scope_questions = [
            {
                'question': 'Who is the president of the United States?',
                'category': 'current_events',
                'expected_behavior': 'Decline and suggest NLP topics'
            },
            {
                'question': 'What is the capital of France?',
                'category': 'general_knowledge',
                'expected_behavior': 'Decline and suggest NLP topics'
            },
            {
                'question': 'How do I cook pasta?',
                'category': 'unrelated_topic',
                'expected_behavior': 'Decline and suggest NLP topics'
            },
            {
                'question': 'What is quantum mechanics?',
                'category': 'different_field',
                'expected_behavior': 'Decline and suggest NLP topics'
            }
        ]

        for test_case in out_of_scope_questions:
            question = test_case['question']
            print(f"\nQ: {question}")

            result = self.rag.query(question, verbose=False)
            answer = result['answer'].lower()

            # Check if answer properly declines
            decline_phrases = [
                'outside the scope',
                'not covered in the textbook',
                'outside the textbook',
                'not in the textbook',
                'cannot answer this',
                'this question is outside'
            ]

            properly_declined = any(phrase in answer for phrase in decline_phrases)

            # Check that it doesn't hallucinate
            hallucination_indicators = [
                'george marshall',  # Random person mentioned before
                'according to',     # Making claims
                'the textbook states', # False attribution
                'weizenbaum',      # Unless actually in context
            ]

            has_hallucination = any(indicator in answer for indicator in hallucination_indicators)

            test_result = {
                'test': 'out_of_scope',
                'question': question,
                'category': test_case['category'],
                'answer': result['answer'],
                'properly_declined': properly_declined,
                'has_hallucination': has_hallucination,
                'passed': properly_declined and not has_hallucination
            }

            self.results['tests'].append(test_result)
            self.results['summary']['total'] += 1

            if test_result['passed']:
                self.results['summary']['passed'] += 1
                print(f"✓ PASS: Properly declined")
            else:
                self.results['summary']['failed'] += 1
                print(f"✗ FAIL:")
                if not properly_declined:
                    print(f"  - Did not properly decline out-of-scope question")
                if has_hallucination:
                    print(f"  - Contains hallucination indicators")
                print(f"  Answer: {result['answer'][:200]}...")

    def test_valid_nlp_questions(self):
        """Test that valid NLP questions get good answers"""
        print("\n" + "="*70)
        print("TEST 2: VALID NLP QUESTION HANDLING")
        print("="*70)

        nlp_questions = [
            {
                'question': 'How do transformers use attention mechanisms?',
                'expected_chapters': ['transformer', '8'],
                'keywords': ['attention', 'query', 'key', 'value']
            },
            {
                'question': 'What are word embeddings?',
                'expected_chapters': ['embedding', '4'],
                'keywords': ['vector', 'semantic', 'word']
            },
            {
                'question': 'Explain n-gram language models',
                'expected_chapters': ['n-gram', '3'],
                'keywords': ['probability', 'sequence', 'context']
            },
            {
                'question': 'What is RAG?',
                'expected_chapters': ['retrieval', '11', 'rag'],
                'keywords': ['retrieval', 'augmented', 'generation', 'documents']
            }
        ]

        for test_case in nlp_questions:
            question = test_case['question']
            print(f"\nQ: {question}")

            result = self.rag.query(question, verbose=False)
            answer = result['answer'].lower()
            citations = [c.lower() for c in result['citations']]

            # Check that it didn't decline
            decline_phrases = ['outside the scope', 'cannot answer']
            incorrectly_declined = any(phrase in answer for phrase in decline_phrases)

            # Check for expected keywords
            has_keywords = any(keyword in answer for keyword in test_case['keywords'])

            # Check citations are relevant
            has_relevant_citations = any(
                any(chapter in citation for chapter in test_case['expected_chapters'])
                for citation in citations
            )

            # Check for hallucination of specific facts
            # (We can't easily verify all facts, but we can check for common issues)
            has_specific_numbers = any(char.isdigit() for char in answer)
            if has_specific_numbers:
                # If answer contains numbers, check they're from context
                retrieved_text = ' '.join([chunk['text'].lower() for chunk in result['retrieved_chunks']])
                # This is a simplified check
                hallucination_risk = False  # Would need more sophisticated checking
            else:
                hallucination_risk = False

            test_result = {
                'test': 'valid_nlp_question',
                'question': question,
                'answer': result['answer'][:500],
                'citations': result['citations'],
                'incorrectly_declined': incorrectly_declined,
                'has_keywords': has_keywords,
                'has_relevant_citations': has_relevant_citations,
                'hallucination_risk': hallucination_risk,
                'passed': not incorrectly_declined and has_keywords and has_relevant_citations
            }

            self.results['tests'].append(test_result)
            self.results['summary']['total'] += 1

            if test_result['passed']:
                self.results['summary']['passed'] += 1
                print(f"✓ PASS: Good answer with relevant citations")
            else:
                self.results['summary']['failed'] += 1
                print(f"✗ FAIL:")
                if incorrectly_declined:
                    print(f"  - Incorrectly declined a valid NLP question")
                if not has_keywords:
                    print(f"  - Missing expected keywords: {test_case['keywords']}")
                if not has_relevant_citations:
                    print(f"  - Missing relevant citations from expected chapters")

    def test_context_grounding(self):
        """Test that answers are grounded in retrieved context"""
        print("\n" + "="*70)
        print("TEST 3: CONTEXT GROUNDING")
        print("="*70)

        question = "What is backpropagation in neural networks?"
        print(f"\nQ: {question}")

        result = self.rag.query(question, verbose=False)
        answer = result['answer'].lower()
        retrieved_text = ' '.join([chunk['text'].lower() for chunk in result['retrieved_chunks']])

        # Extract key claims from answer (simplified - would need NLP for real extraction)
        # For now, just check that answer doesn't contain common hallucination patterns
        hallucination_patterns = [
            'in 19',  # Specific years not in context
            'invented by',  # Attribution not in context
            'first introduced in',  # Historical claims
            'won the nobel prize',  # Specific achievements
            'according to recent studies',  # Recency claims
        ]

        found_patterns = [p for p in hallucination_patterns if p in answer]

        # Check if suspicious patterns are actually in the retrieved context
        false_positives = []
        true_hallucinations = []

        for pattern in found_patterns:
            if pattern in retrieved_text:
                false_positives.append(pattern)  # It's in the context, so it's okay
            else:
                true_hallucinations.append(pattern)  # Not in context - hallucination!

        test_result = {
            'test': 'context_grounding',
            'question': question,
            'answer': result['answer'][:500],
            'hallucination_patterns_found': found_patterns,
            'true_hallucinations': true_hallucinations,
            'passed': len(true_hallucinations) == 0
        }

        self.results['tests'].append(test_result)
        self.results['summary']['total'] += 1

        if test_result['passed']:
            self.results['summary']['passed'] += 1
            print(f"✓ PASS: Answer appears grounded in context")
        else:
            self.results['summary']['failed'] += 1
            print(f"✗ FAIL: Potential hallucinations detected:")
            for pattern in true_hallucinations:
                print(f"  - Pattern '{pattern}' found in answer but not in context")

    def test_citation_accuracy(self):
        """Test that citations match the retrieved chunks"""
        print("\n" + "="*70)
        print("TEST 4: CITATION ACCURACY")
        print("="*70)

        question = "How do RNNs handle sequential data?"
        print(f"\nQ: {question}")

        result = self.rag.query(question, verbose=False)
        citations = result['citations']
        chunks = result['retrieved_chunks']

        # Extract chapters from citations
        cited_chapters = set()
        for citation in citations:
            # Extract chapter name from citation string
            if 'Chapter:' in citation:
                chapter_part = citation.split('Chapter:')[1].split(']')[0].strip()
                cited_chapters.add(chapter_part.lower())

        # Extract chapters from retrieved chunks
        retrieved_chapters = set(chunk['chapter'].lower() for chunk in chunks)

        # Check that all cited chapters actually appear in retrieved chunks
        incorrect_citations = cited_chapters - retrieved_chapters

        # Check that we're not missing important sources
        missing_citations = len(retrieved_chapters) - len(cited_chapters)

        test_result = {
            'test': 'citation_accuracy',
            'question': question,
            'citations': citations,
            'cited_chapters': list(cited_chapters),
            'retrieved_chapters': list(retrieved_chapters),
            'incorrect_citations': list(incorrect_citations),
            'missing_important_sources': missing_citations > 2,  # Some duplicates expected
            'passed': len(incorrect_citations) == 0
        }

        self.results['tests'].append(test_result)
        self.results['summary']['total'] += 1

        if test_result['passed']:
            self.results['summary']['passed'] += 1
            print(f"✓ PASS: All citations match retrieved chunks")
        else:
            self.results['summary']['failed'] += 1
            print(f"✗ FAIL: Citation accuracy issues:")
            if incorrect_citations:
                print(f"  - Cited chapters not in retrieved chunks: {incorrect_citations}")

    def test_insufficient_context_handling(self):
        """Test handling when context is insufficient"""
        print("\n" + "="*70)
        print("TEST 5: INSUFFICIENT CONTEXT HANDLING")
        print("="*70)

        # Ask a very specific question that's unlikely to have good context
        question = "What is the exact formula for the third derivative of the loss function in transformers?"
        print(f"\nQ: {question}")

        result = self.rag.query(question, verbose=False)
        answer = result['answer'].lower()

        # Check for appropriate hedging/uncertainty
        uncertainty_phrases = [
            "don't have enough information",
            "insufficient information",
            "context doesn't contain",
            "not enough detail",
            "unable to find",
            "cannot provide the exact"
        ]

        shows_uncertainty = any(phrase in answer for phrase in uncertainty_phrases)

        # Should not make up specific formulas
        makes_up_formula = ('∂' in answer or 'derivative' in answer) and not shows_uncertainty

        test_result = {
            'test': 'insufficient_context',
            'question': question,
            'answer': result['answer'][:500],
            'shows_uncertainty': shows_uncertainty,
            'makes_up_formula': makes_up_formula,
            'passed': shows_uncertainty or not makes_up_formula
        }

        self.results['tests'].append(test_result)
        self.results['summary']['total'] += 1

        if test_result['passed']:
            self.results['summary']['passed'] += 1
            print(f"✓ PASS: Appropriately handles insufficient context")
        else:
            self.results['summary']['failed'] += 1
            print(f"✗ FAIL: May be making up specific details")

    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "#"*70)
        print("RAG QUALITY TEST SUITE")
        print("#"*70)

        self.test_out_of_scope_questions()
        self.test_valid_nlp_questions()
        self.test_context_grounding()
        self.test_citation_accuracy()
        self.test_insufficient_context_handling()

        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests:  {self.results['summary']['total']}")
        print(f"Passed:       {self.results['summary']['passed']} ✓")
        print(f"Failed:       {self.results['summary']['failed']} ✗")
        print(f"Pass Rate:    {self.results['summary']['passed'] / self.results['summary']['total'] * 100:.1f}%")

        # Save detailed results
        output_dir = Path(__file__).parent.parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"rag_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")

        return self.results


def main():
    """Run RAG quality tests"""
    print("Initializing RAG pipeline...")
    rag = NLPTextbookRAG(
        temperature=0.3,  # Lower temperature for more consistent testing
        max_tokens=500,
        top_k=5
    )

    tester = RAGTester(rag)
    results = tester.run_all_tests()

    # Exit with error code if tests failed
    if results['summary']['failed'] > 0:
        print("\n⚠ Some tests failed. Review the results above.")
        exit(1)
    else:
        print("\n✓ All tests passed!")
        exit(0)


if __name__ == "__main__":
    main()
