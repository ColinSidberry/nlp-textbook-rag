#!/usr/bin/env python3
"""
Streamlit UI for NLP Textbook RAG System
Milestone 4 Implementation

A web interface for querying the NLP textbook using RAG.
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import NLPTextbookRAG


# Page configuration
st.set_page_config(
    page_title="NLP Textbook RAG System",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
        margin-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """
    Initialize RAG pipeline with caching
    This ensures the pipeline is only loaded once and reused across sessions
    """
    return NLPTextbookRAG(
        temperature=0.7,
        max_tokens=500,
        top_k=5
    )


def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate user query

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Please enter a question"

    if len(query.strip()) < 3:
        return False, "Please enter a longer question (at least 3 characters)"

    return True, ""


def display_citations(citations: list[str]):
    """Display citations in a formatted box"""
    if citations:
        st.markdown("### 📚 Citations")
        for citation in citations:
            st.markdown(f'<div class="citation-box">{citation}</div>', unsafe_allow_html=True)
    else:
        st.info("No citations available")


def display_retrieved_chunks(chunks: list[dict]):
    """Display retrieved chunks in an expandable section"""
    if not chunks:
        return

    with st.expander(f"📖 Retrieved Chunks ({len(chunks)})", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}** - Chapter: {chunk['chapter']} | Similarity: {chunk['similarity']:.3f}")
            st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
            st.divider()


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">📚 NLP Textbook RAG System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions about NLP concepts from Jurafsky & Martin\'s textbook</div>',
        unsafe_allow_html=True
    )

    # Initialize RAG pipeline with error handling
    try:
        with st.spinner("Initializing RAG pipeline..."):
            rag = initialize_rag()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.info("Please ensure:\n- ChromaDB data exists at `/data/chroma/`\n- Ollama is running with the Mistral model\n- All dependencies are installed")
        st.stop()

    # Query interface
    st.markdown("---")

    # Text input
    query = st.text_input(
        "Your Question",
        placeholder="e.g., How do transformers use attention mechanisms?",
        help="Enter a question about NLP concepts from the textbook",
        label_visibility="collapsed"
    )

    # Submit button
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    # Process query
    if ask_button:
        # Validate query
        is_valid, error_message = validate_query(query)

        if not is_valid:
            st.warning(error_message)
            st.stop()

        # Execute query with loading spinner
        try:
            start_time = time.time()

            with st.spinner("Searching textbook and generating answer..."):
                result = rag.query(query, verbose=False)

            query_time = time.time() - start_time

            # Display results
            st.markdown("---")

            # Answer section
            st.markdown("### 📝 Answer")

            if result['answer']:
                st.markdown(result['answer'])
            else:
                st.warning("No answer generated. Try rephrasing your question.")

            # Citations section
            st.markdown("---")
            display_citations(result['citations'])

            # Retrieved chunks section
            st.markdown("---")
            display_retrieved_chunks(result['retrieved_chunks'])

            # Query metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Query Time", f"{query_time:.2f}s")
            with col2:
                st.metric("Chunks Used", len(result['retrieved_chunks']))
            with col3:
                st.metric("Sources", len(result['citations']))

        except ConnectionError as e:
            st.error("Cannot connect to Ollama. Please ensure Ollama is running with the Mistral model.")
            st.info("Start Ollama with: `ollama run mistral:7b-instruct`")

        except FileNotFoundError as e:
            st.error("Cannot connect to database. Please check that the data directory exists.")
            st.info(f"Expected ChromaDB location: `/Users/colinsidberry/nlp-textbook-rag/data/chroma/`")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again or rephrase your question.")

    # Footer with helpful information
    st.markdown("---")
    with st.expander("ℹ️ About this system"):
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) system provides answers to questions about NLP concepts
        using content from *Speech and Language Processing* by Jurafsky & Martin.

        **How it works:**
        1. Your question is converted to an embedding vector
        2. The most relevant chunks are retrieved from 7 indexed chapters (11,419 total chunks)
        3. A language model (Mistral 7B) generates an answer based on the retrieved context
        4. Citations show which chapters were used

        **Sample questions to try:**
        - How do transformers use attention mechanisms?
        - Explain backpropagation in neural networks
        - What are word embeddings and how are they created?
        - How do n-gram language models work?
        - Explain logistic regression for text classification

        **Indexed chapters:**
        - RAG (Retrieval-Augmented Generation)
        - Embeddings
        - Logistic Regression
        - N-gram Language Models
        - Neural Networks
        - Transformers
        - Words and Tokens
        """)


if __name__ == "__main__":
    main()
