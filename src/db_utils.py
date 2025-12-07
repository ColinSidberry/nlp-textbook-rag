#!/usr/bin/env python3
"""
Database utilities for ChromaDB access
Shared between query interface and database viewer
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
import chromadb


@st.cache_resource
def get_chroma_client():
    """
    Get cached ChromaDB client (shared across all sessions)

    Returns:
        ChromaDB PersistentClient instance
    """
    chroma_path = Path(__file__).parent.parent / "data" / "chroma"
    return chromadb.PersistentClient(path=str(chroma_path))


@st.cache_resource
def get_collection():
    """
    Get cached collection reference

    Returns:
        ChromaDB collection for nlp_textbook
    """
    client = get_chroma_client()
    return client.get_collection(name="nlp_textbook")


@st.cache_data(ttl=3600)
def get_collection_count() -> int:
    """
    Get total entry count (cached for 1 hour)

    Returns:
        Total number of entries in the collection
    """
    collection = get_collection()
    return collection.count()


@st.cache_data(ttl=600)
def get_unique_chapters() -> List[str]:
    """
    Get list of unique chapters (cached for 10 minutes)

    Returns:
        Sorted list of unique chapter names
    """
    collection = get_collection()
    # Fetch all entries to extract unique chapters
    # This is acceptable as it only fetches metadata, not documents
    results = collection.get(include=['metadatas'])
    chapters = sorted(set(
        meta.get('chapter', 'Unknown')
        for meta in results['metadatas']
        if meta.get('chapter')
    ))
    return chapters


@st.cache_data(ttl=600)
def get_unique_filenames() -> List[str]:
    """
    Get list of unique filenames (cached for 10 minutes)

    Returns:
        Sorted list of unique filenames
    """
    collection = get_collection()
    results = collection.get(include=['metadatas'])
    filenames = sorted(set(
        meta.get('filename', 'Unknown')
        for meta in results['metadatas']
        if meta.get('filename')
    ))
    return filenames


@st.cache_data(ttl=300)
def get_paginated_entries(
    page_num: int = 1,
    page_size: int = 50,
    filters: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Get paginated entries from ChromaDB

    Uses an efficient ID-based pagination approach:
    1. Get all IDs (cheap operation, no documents)
    2. Apply metadata filters if provided
    3. Slice IDs for current page
    4. Fetch only the documents for this page

    Args:
        page_num: Page number (1-indexed)
        page_size: Number of entries per page
        filters: Optional metadata filters (e.g., {"chapter": "Transformers"})

    Returns:
        Dict with:
            - ids: List of document IDs
            - documents: List of document texts
            - metadatas: List of metadata dicts
            - total_count: Total number of entries (filtered or unfiltered)
            - page: Current page number
            - page_size: Entries per page
            - total_pages: Total number of pages
    """
    collection = get_collection()

    # Get IDs (with optional filtering)
    if filters:
        try:
            results = collection.get(
                where=filters,
                include=[]  # Only IDs, no documents
            )
            all_ids = results['ids']
        except Exception as e:
            # If filter fails, fall back to all IDs
            print(f"Filter failed: {e}, falling back to all IDs")
            results = collection.get(include=[])
            all_ids = results['ids']
    else:
        results = collection.get(include=[])
        all_ids = results['ids']

    # Calculate pagination
    total_count = len(all_ids)

    if total_count == 0:
        return {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'total_count': 0,
            'page': 1,
            'page_size': page_size,
            'total_pages': 0
        }

    total_pages = (total_count + page_size - 1) // page_size

    # Ensure page_num is valid
    page_num = max(1, min(page_num, total_pages))

    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, total_count)
    page_ids = all_ids[start_idx:end_idx]

    # Fetch documents for this page
    if page_ids:
        page_results = collection.get(
            ids=page_ids,
            include=['documents', 'metadatas']
        )
    else:
        page_results = {'ids': [], 'documents': [], 'metadatas': []}

    return {
        'ids': page_results.get('ids', []),
        'documents': page_results.get('documents', []),
        'metadatas': page_results.get('metadatas', []),
        'total_count': total_count,
        'page': page_num,
        'page_size': page_size,
        'total_pages': total_pages
    }


def search_in_documents(
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
    search_text: str
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Filter documents by text search (client-side)

    Performs case-insensitive substring matching within document contents.

    Args:
        documents: List of document texts
        metadatas: List of metadata dicts
        ids: List of document IDs
        search_text: Text to search for (case-insensitive)

    Returns:
        Tuple of (filtered_documents, filtered_metadatas, filtered_ids)
    """
    if not search_text or not search_text.strip():
        return documents, metadatas, ids

    search_lower = search_text.strip().lower()

    filtered_docs = []
    filtered_metas = []
    filtered_ids = []

    for doc, meta, doc_id in zip(documents, metadatas, ids):
        if search_lower in doc.lower():
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_ids.append(doc_id)

    return filtered_docs, filtered_metas, filtered_ids
