#!/usr/bin/env python3
"""
RAG Pipeline for NLP Textbook
Milestone 3 Implementation

Features:
- Query embedding and retrieval from ChromaDB
- LangChain integration with Ollama (Mistral model)
- Citation system with chapter and filename references
- End-to-end question-answering pipeline
"""

from pathlib import Path
from typing import List, Dict, Tuple
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


class NLPTextbookRAG:
    """RAG pipeline for querying NLP textbook content"""

    # Common NLP acronyms and their expansions
    ACRONYM_EXPANSIONS = {
        'RAG': 'retrieval-augmented generation retrieval augmented generation',
        'NER': 'named entity recognition',
        'POS': 'part-of-speech part of speech tagging',
        'LLM': 'large language model',
        'LSTM': 'long short-term memory',
        'RNN': 'recurrent neural network',
        'CNN': 'convolutional neural network',
        'GRU': 'gated recurrent unit',
        'NLP': 'natural language processing',
        'CRF': 'conditional random field',
        'HMM': 'hidden markov model',
        'TF-IDF': 'term frequency inverse document frequency',
        'BERT': 'bidirectional encoder representations from transformers',
        'GPT': 'generative pre-trained transformer',
        'IR': 'information retrieval',
    }

    def __init__(
        self,
        chroma_path: str = None,
        collection_name: str = "nlp_textbook",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistral:7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline

        Args:
            chroma_path: Path to ChromaDB storage (default: ../data/chroma)
            collection_name: Name of ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            llm_model: Ollama model for generation
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens in LLM response
            top_k: Number of chunks to retrieve
        """
        # Set up paths
        if chroma_path is None:
            chroma_path = Path(__file__).parent.parent / "data" / "chroma"

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_collection(name=collection_name)

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model, local_files_only=True)

        # Initialize Ollama LLM
        print(f"Initializing Ollama with model: {llm_model}...")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Using Ollama at: {ollama_base_url}")
        self.llm = OllamaLLM(
            model=llm_model,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=ollama_base_url
        )

        # Configuration
        self.top_k = top_k

        # Create prompt template with strong guardrails
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI assistant helping students understand concepts from the NLP textbook "Speech and Language Processing" by Jurafsky & Martin.

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. SCOPE: ONLY answer questions about NLP/linguistics concepts (transformers, embeddings, parsing, language models, etc.)
2. GROUNDING: ONLY use information explicitly stated in the context below - NEVER use external knowledge
3. OUT-OF-SCOPE QUESTIONS: If the question asks about current events, general knowledge, cooking, geography, politics, or ANY non-NLP topic, respond EXACTLY with: "This question is outside the scope of the NLP textbook. Please ask about natural language processing concepts like transformers, embeddings, language models, parsing, etc."
4. INSUFFICIENT CONTEXT: If the context lacks information for a valid NLP question, say: "I don't have enough information in the retrieved context to answer this question. Try rephrasing or asking about a different aspect."
5. NO FABRICATION: Do NOT make up facts, examples, names, dates, or connections not in the context
6. NO EXTERNAL KNOWLEDGE: Even if you know the answer from general knowledge, ONLY use the context
7. CITATION ACCURACY: Only cite chapters/sources that appear in the context excerpts below

Context from textbook:
{context}

Question: {question}

Answer (use ONLY the context above, decline if out-of-scope):"""
        )

        print("RAG pipeline initialized successfully!")

    def expand_query(self, query: str) -> str:
        """
        Expand acronyms and enhance query for better retrieval

        Args:
            query: Original query string

        Returns:
            Expanded query string
        """
        import re

        expanded = query

        # 1. Expand acronyms
        for acronym, expansion in self.ACRONYM_EXPANSIONS.items():
            # Look for the acronym as a whole word (case-insensitive)
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                # Add expansion to query (don't replace, augment)
                expanded = f"{query} {expansion}"
                break  # Only expand the first acronym found

        # 2. Handle comparison questions - extract both terms
        comparison_patterns = [
            r"difference between (\w+) and (\w+)",
            r"compare (\w+) (?:and|vs|versus) (\w+)",
            r"(\w+) vs (\w+)",
            r"(\w+) versus (\w+)"
        ]

        for pattern in comparison_patterns:
            match = re.search(pattern, query.lower())
            if match:
                term1, term2 = match.groups()
                # Augment query with both terms emphasized
                expanded = f"{query} {term1} {term2} comparison"
                break

        return expanded

    def retrieve(self, query: str, k: int = None) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant chunks from ChromaDB

        Args:
            query: User query string
            k: Number of results to retrieve (default: self.top_k)

        Returns:
            Tuple of (documents, metadatas) lists
        """
        if k is None:
            k = self.top_k

        # Expand query for better acronym matching
        expanded_query = self.expand_query(query)

        # Generate query embedding (use expanded query)
        query_embedding = self.embedding_model.encode([expanded_query])[0].tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Handle empty results
        if not results['documents'] or not results['documents'][0]:
            return [], []

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Add similarity scores to metadata
        for i, metadata in enumerate(metadatas):
            metadata['similarity'] = 1 - distances[i]

        return documents, metadatas

    def format_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for LLM

        Args:
            documents: List of retrieved text chunks
            metadatas: List of metadata dicts

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
            chapter = metadata.get('chapter', 'Unknown')
            filename = metadata.get('filename', 'Unknown')
            similarity = metadata.get('similarity', 0.0)

            context_parts.append(
                f"[Excerpt {i} - Chapter: {chapter}, Source: {filename}, Relevance: {similarity:.2f}]\n{doc}\n"
            )

        return "\n".join(context_parts)

    def extract_citations(self, metadatas: List[Dict]) -> List[str]:
        """
        Extract and format citations from retrieved chunks

        Args:
            metadatas: List of metadata dicts

        Returns:
            List of formatted citation strings
        """
        citations = []
        seen = set()  # For deduplication

        for metadata in metadatas:
            chapter = metadata.get('chapter', 'Unknown')
            filename = metadata.get('filename', 'Unknown')

            # Create unique key for deduplication
            citation_key = (chapter, filename)

            if citation_key not in seen:
                citations.append(f"[Chapter: {chapter}] (Source: {filename})")
                seen.add(citation_key)

        return citations

    def query(self, question: str, verbose: bool = False) -> Dict:
        """
        Execute end-to-end RAG query

        Args:
            question: User question
            verbose: If True, print detailed information

        Returns:
            Dict with keys: 'question', 'answer', 'citations', 'retrieved_chunks'
        """
        if verbose:
            print(f"\nProcessing query: '{question}'")
            print("=" * 70)

        # Step 1: Retrieve relevant chunks
        if verbose:
            print(f"\n[1] Retrieving top-{self.top_k} chunks from ChromaDB...")

        documents, metadatas = self.retrieve(question)

        if not documents:
            return {
                'question': question,
                'answer': "I couldn't find relevant information in the textbook to answer this question.",
                'citations': [],
                'retrieved_chunks': []
            }

        if verbose:
            print(f"    Retrieved {len(documents)} chunks")
            for i, metadata in enumerate(metadatas, 1):
                print(f"    - Chunk {i}: {metadata['chapter']} ({metadata['similarity']:.3f} similarity)")

        # Step 2: Format context
        if verbose:
            print(f"\n[2] Formatting context for LLM...")

        context = self.format_context(documents, metadatas)

        # Step 2.5: Check if retrieved chunks are actually relevant
        # If all chunks have low similarity, question is likely out-of-scope
        avg_similarity = sum(m['similarity'] for m in metadatas) / len(metadatas)
        max_similarity = max(m['similarity'] for m in metadatas)

        # Adaptive threshold: Allow if avg > 0.4 OR if (avg > 0.35 AND max > 0.42)
        # This handles typos while blocking truly out-of-scope questions
        is_out_of_scope = avg_similarity < 0.35 or (avg_similarity < 0.4 and max_similarity < 0.42)

        if is_out_of_scope:
            return {
                'question': question,
                'answer': "This question is outside the scope of the NLP textbook. Please ask about natural language processing concepts like transformers, embeddings, language models, parsing, sentiment analysis, etc.",
                'citations': [],
                'retrieved_chunks': []
            }

        # Step 3: Generate response
        if verbose:
            print(f"\n[3] Generating response with Ollama (mistral)...")
            print(f"    Average chunk similarity: {avg_similarity:.3f}")

        # Format prompt with context and question
        formatted_prompt = self.prompt_template.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)

        # Step 4: Extract citations
        if verbose:
            print(f"\n[4] Extracting citations...")

        citations = self.extract_citations(metadatas)

        if verbose:
            print(f"    Found {len(citations)} unique source(s)")

        # Return results
        result = {
            'question': question,
            'answer': response.strip(),
            'citations': citations,
            'retrieved_chunks': [
                {
                    'text': doc,
                    'chapter': meta['chapter'],
                    'filename': meta['filename'],
                    'similarity': meta['similarity']
                }
                for doc, meta in zip(documents, metadatas)
            ]
        }

        if verbose:
            print("\n" + "=" * 70)
            print("ANSWER:")
            print("=" * 70)
            print(result['answer'])
            print("\n" + "=" * 70)
            print("CITATIONS:")
            print("=" * 70)
            for citation in citations:
                print(f"  - {citation}")
            print("=" * 70)

        return result

    def batch_query(self, questions: List[str], verbose: bool = False) -> List[Dict]:
        """
        Process multiple queries

        Args:
            questions: List of question strings
            verbose: If True, print detailed information

        Returns:
            List of result dicts
        """
        results = []
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n{'#' * 70}")
                print(f"QUERY {i}/{len(questions)}")
                print(f"{'#' * 70}")

            result = self.query(question, verbose=verbose)
            results.append(result)

        return results


def main():
    """Demo of RAG pipeline"""
    print("\n" + "=" * 70)
    print("NLP TEXTBOOK RAG PIPELINE - DEMO")
    print("=" * 70)

    # Initialize RAG pipeline
    rag = NLPTextbookRAG(
        temperature=0.7,
        max_tokens=500,
        top_k=5
    )

    # Test query
    test_question = "How do transformers use attention mechanisms?"

    print(f"\nTest Query: '{test_question}'")
    print("=" * 70)

    result = rag.query(test_question, verbose=True)

    print("\n" + "=" * 70)
    print("✓ RAG PIPELINE DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
