# NLP Textbook RAG System

A Retrieval-Augmented Generation (RAG) system for semantic search and synthesis across NLP textbook chapters from Jurafsky & Martin's "Speech and Language Processing".

## Project Overview

- **Data Source**: 7 NLP textbook chapters (~565KB)
- **Chapters**: RAG, Embeddings, Transformers, Neural Networks, Logistic Regression, N-grams, Words & Tokens
- **Tech Stack**: Python, ChromaDB, LangChain, Ollama, Streamlit, sentence-transformers
- **Architecture**: Local-first, privacy-preserving

## Performance Metrics

**Query Performance** (tested on MacBook Pro M1 Pro, 32GB RAM):
- **Average Query Time:** 18.99 seconds
- **Median Query Time:** 17.94 seconds
- **Range:** 14.55s - 25.45s

**System Specs:**
- **LLM:** Mistral 7B-Instruct via Ollama
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (384 dims)
- **Indexed Chunks:** 11,419
- **ChromaDB Size:** 97MB

**Performance Breakdown:**
- Retrieval (Embedding + Search): ~1-2s (8-10%)
- Generation (LLM): ~17-23s (90-92%)

**Quality-over-Speed Trade-off:**
The ~19 second average query time reflects a deliberate choice to use Mistral 7B-Instruct for high-quality, comprehensive answers (400-500 words) rather than a smaller, faster model. For an educational use case, this trade-off prioritizes thorough explanations over instant results.

**Optimization Options Available:**
- Smaller model (Llama 3.2:3b): ~10s queries
- GPU acceleration: ~5-8s queries
- Reduced output length: ~12s queries

See [PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md) for detailed analysis, optimization opportunities, and complete benchmark results.

## Project Structure

```
nlp-textbook-rag/
├── src/                    # Source code
│   └── index_nlp_textbook.py    # Textbook parser (Milestone 1)
├── data/
│   ├── normalized/         # Parsed JSON data
│   │   └── nlp_textbook.json    # Normalized textbook data
│   └── chroma/            # Vector database storage
├── tests/                 # Test scripts
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Milestone 1: Data Pipeline Setup ✅

**Status**: Complete

### Completed Tasks
1. ✅ Project structure setup
2. ✅ Virtual environment created
3. ✅ Dependencies specified in requirements.txt
4. ✅ Textbook parser developed
5. ✅ JSON normalization implemented

### Parser Features
- Discovers all .txt files in source directory
- Extracts chapter titles (handles multi-line titles)
- Handles Unicode characters (curly quotes, Chinese characters)
- Generates unique document IDs using SHA-256 hashing
- Preserves full text content and metadata

### JSON Schema
```json
{
  "id": "unique-hash",
  "source": "nlp_textbook",
  "chapter": "extracted-chapter-title",
  "filename": "transformers.txt",
  "content": "full-text-content",
  "file_path": "/absolute/path/to/chapter.txt",
  "file_size_kb": 85.34
}
```

### Validation Results
- ✅ 7 documents successfully parsed
- ✅ Total content size: 564.54 KB
- ✅ All required fields present in each document
- ✅ No encoding errors or parsing failures

### Parsed Chapters
1. Information Retrieval and Retrieval-Augmented Generation (61.27 KB)
2. Embeddings (85.34 KB)
3. Logistic Regression and Text Classification (97.37 KB)
4. N-gram Language Models (74.14 KB)
5. Neural Networks (67.93 KB)
6. Transformers (73.46 KB)
7. Words and Tokens (105.03 KB)

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or navigate to project directory**
   ```bash
   cd /Users/colinsidberry/nlp-textbook-rag
   ```

2. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies** (when ready for Milestone 2+)
   ```bash
   pip install -r requirements.txt
   ```

### Running the Parser

```bash
python3 src/index_nlp_textbook.py
```

This will:
- Discover all .txt files in `/Users/colinsidberry/Downloads/NLP_Textbook`
- Parse chapter titles and content
- Generate normalized JSON output at `data/normalized/nlp_textbook.json`
- Display validation summary

## Next Steps: Milestone 2

- [ ] Implement chunking strategy (paragraph-based, ~400 tokens)
- [ ] Set up sentence-transformers embedding model
- [ ] Initialize ChromaDB with persistent storage
- [ ] Index full corpus (target: 10k+ chunks)

## License

Educational use only. Textbook content copyright © 2025 by Jurafsky & Martin.
