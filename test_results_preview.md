# Milestone 5 Test Results
## NLP Textbook RAG System - Comprehensive Query Testing

**Date:** December 6, 2025  
**Total Queries Tested:** 20  
**Test Duration:** ~6.5 minutes  

---

## EXECUTIVE SUMMARY

Successfully tested the RAG system with 20 diverse queries across 6 query types (Concept, Definition, Comparison, Application, Technical, Edge). The system demonstrated strong performance with:

- **Average query time:** 19.12 seconds
- **Average similarity score:** 0.644 (good relevance)
- **Average relevance rating:** 3.10/5
- **Average quality rating:** 5.00/5 (all answers were comprehensive)
- **Citations:** 100% of queries generated proper citations

All 7 textbook chapters were successfully retrieved across the test set, demonstrating comprehensive coverage.

---

## SUMMARY STATISTICS

### Performance Metrics

| Metric | Value |
|--------|-------|
| Average Query Time | 19.12s |
| Min Query Time | 14.55s |
| Max Query Time | 25.45s |
| Average Similarity Score | 0.644 |
| Average Relevance Rating | 3.10/5 |
| Average Quality Rating | 5.00/5 |
| Total Citations Generated | 27 |
| Average Citations per Query | 1.4 |

### Query Type Breakdown

| Query Type | Count | Avg Similarity | Avg Time |
|------------|-------|----------------|----------|
| Application | 3 | 0.529 | 15.19s |
| Comparison | 3 | 0.705 | 24.16s |
| Concept | 5 | 0.704 | 19.93s |
| Definition | 4 | 0.657 | 17.72s |
| Edge | 2 | 0.425 | 19.48s |
| Technical | 3 | 0.724 | 18.28s |


### Chapter Coverage

**Unique chapters referenced:** 7/7

Chapters retrieved:
- Embeddings
- Information Retrieval and Retrieval-Augmented Generation
- Logistic Regression and Text Classification
- N-gram Language Models
- Neural Networks
- Transformers
- Words and Tokens

---

## TOP 5 DEMO QUERY CANDIDATES

Based on multi-factor scoring (similarity × 3 + answer_length × 2 + speed × 1 + citations × 2):


### Demo Query #1 (Score: 7.66)

**Query:** "What's the difference between n-grams and neural language models?"  
**Type:** Comparison  
**Performance:**
- Similarity: 0.709 (max: 0.729)
- Query time: 22.79s
- Answer length: 1646 characters
- Citations: 2 sources
- Top chapter: N-gram Language Models

**Ratings:** Relevance 4/5, Quality 5/5

**Answer Preview:**
> Based on the provided textbook excerpts, n-grams and neural language models are two different approaches used in Natural Language Processing (NLP).

1. N-gram Language Models: These models are based on sequences of contiguous n items from a given sample of text. For example, bigrams consist of pairs...

**Citations:**
- [Chapter: N-gram Language Models] (Source: n-gram.txt)
- [Chapter: Neural Networks] (Source: neural-networks.txt)

### Demo Query #2 (Score: 7.18)

**Query:** "How do you handle out-of-vocabulary words?"  
**Type:** Edge  
**Performance:**
- Similarity: 0.526 (max: 0.535)
- Query time: 22.65s
- Answer length: 1680 characters
- Citations: 2 sources
- Top chapter: Embeddings

**Ratings:** Relevance 2/5, Quality 5/5

**Answer Preview:**
> In the context provided from Speech and Language Processing by Jurafsky & Martin, while there is no direct mention of how out-of-vocabulary (OOV) words are handled, we can infer some insights based on the discussions about vocabulary size and word embeddings in Chapters "Embeddings" and "Words and T...

**Citations:**
- [Chapter: Embeddings] (Source: embeddings.txt)
- [Chapter: Words and Tokens] (Source: words-and-tokens.txt)

### Demo Query #3 (Score: 7.03)

**Query:** "How do transformers differ from RNNs?"  
**Type:** Comparison  
**Performance:**
- Similarity: 0.635 (max: 0.688)
- Query time: 25.03s
- Answer length: 1981 characters
- Citations: 1 sources
- Top chapter: Transformers

**Ratings:** Relevance 3/5, Quality 5/5

**Answer Preview:**
> Transformers and Recurrent Neural Networks (RNNs) are two different types of neural networks used in Natural Language Processing (NLP). Although they both aim to process sequential data, they have distinct differences in their structure and functioning.

Transformers, as discussed in Chapter: Transf...

**Citations:**
- [Chapter: Transformers] (Source: transformers.txt)

### Demo Query #4 (Score: 6.96)

**Query:** "What is logistic regression?"  
**Type:** Definition  
**Performance:**
- Similarity: 0.648 (max: 0.677)
- Query time: 17.63s
- Answer length: 1302 characters
- Citations: 2 sources
- Top chapter: Logistic Regression and Text Classification

**Ratings:** Relevance 3/5, Quality 5/5

**Answer Preview:**
> Logistic Regression is a statistical method used for classification problems, where the goal is to predict the probability of an event occurring based on given features or independent variables. It is introduced in detail in Chapter 4 of the textbook (Speech and Language Processing by Jurafsky & Mar...

**Citations:**
- [Chapter: Logistic Regression and Text Classification] (Source: logistic-regression.txt)
- [Chapter: Neural Networks] (Source: neural-networks.txt)

### Demo Query #5 (Score: 6.94)

**Query:** "How does self-attention compute relationships between words?"  
**Type:** Technical  
**Performance:**
- Similarity: 0.594 (max: 0.624)
- Query time: 18.86s
- Answer length: 1395 characters
- Citations: 2 sources
- Top chapter: Transformers

**Ratings:** Relevance 2/5, Quality 5/5

**Answer Preview:**
> Self-attention in Transformers computes relationships between words by assigning weights or attention scores to different words within a sentence or sequence. These weights determine how much each word should focus on other words when processing the input data.

The idea behind self-attention is to ...

**Citations:**
- [Chapter: Transformers] (Source: transformers.txt)
- [Chapter: Information Retrieval and Retrieval-Augmented Generation] (Source: RAG.txt)
