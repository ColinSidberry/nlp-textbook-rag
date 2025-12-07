#!/usr/bin/env python3
"""
Database Viewer Page
Browse and search the indexed NLP textbook chunks
"""

import streamlit as st
import json
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db_utils import (
    get_collection_count,
    get_unique_chapters,
    get_paginated_entries,
    search_in_documents
)


# Page configuration
st.set_page_config(
    page_title="Database Viewer - NLP Textbook RAG",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (match app.py styling)
st.markdown("""
    <style>
    .metric-box {
        background-color: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
        margin-right: 1rem;
        font-size: 1.1rem;
    }
    .entry-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
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
    </style>
""", unsafe_allow_html=True)


def main():
    """Main database viewer interface"""

    # Header
    st.markdown('<div class="main-header">📊 Database Viewer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Explore the indexed NLP textbook chunks</div>',
        unsafe_allow_html=True
    )

    # Display total count prominently
    try:
        total_count = get_collection_count()
    except Exception as e:
        st.error(f"❌ Failed to connect to database: {str(e)}")
        st.info(
            "**Troubleshooting:**\n\n"
            "1. Ensure ChromaDB exists at `data/chroma/`\n"
            "2. Run `python src/verify_database.py` to check database status\n"
            "3. Rebuild database if necessary with `python src/chunk_and_index.py`"
        )
        st.stop()

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 1
    if 'selected_chapter' not in st.session_state:
        st.session_state.selected_chapter = "All Chapters"

    # Filtering Section
    st.markdown("### 🔍 Filters")

    col1, col2 = st.columns([3, 1])

    with col1:
        try:
            chapters = get_unique_chapters()
            selected_chapter = st.selectbox(
                "Filter by Chapter",
                options=["All Chapters"] + chapters,
                key="chapter_filter"
            )
        except Exception as e:
            st.error(f"Error loading chapters: {e}")
            selected_chapter = "All Chapters"
            chapters = []

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state.page = 1
            st.session_state.selected_chapter = "All Chapters"
            st.rerun()

    # Text search
    search_text = st.text_input(
        "🔍 Search within documents",
        placeholder="Enter text to search for within document contents...",
        help="Case-insensitive text search within document content"
    )

    st.divider()

    # Build filters
    where_clause = {}
    if selected_chapter != "All Chapters":
        where_clause["chapter"] = selected_chapter

    # Get paginated results
    with st.spinner("Loading entries..."):
        try:
            results = get_paginated_entries(
                page_num=st.session_state.page,
                page_size=50,
                filters=where_clause if where_clause else None
            )
        except Exception as e:
            st.error(f"Error loading entries: {e}")
            st.stop()

    # Apply text search if provided
    search_count = None
    if search_text and search_text.strip():
        filtered_docs, filtered_metas, filtered_ids = search_in_documents(
            results['documents'],
            results['metadatas'],
            results['ids'],
            search_text
        )

        results['documents'] = filtered_docs
        results['metadatas'] = filtered_metas
        results['ids'] = filtered_ids
        search_count = len(filtered_docs)

    # Statistics
    st.markdown("### 📈 Results")

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Total Entries", f"{results['total_count']:,}")
    with stats_col2:
        if search_count is not None:
            st.metric("Search Results", f"{search_count}")
        else:
            st.metric("Showing", f"{len(results['documents'])}")
    with stats_col3:
        st.metric("Current Page", f"{results['page']} / {results['total_pages']}")

    st.divider()

    # Display results
    if not results['documents']:
        st.warning("⚠️ No entries found matching your filters.")
        st.info("💡 Try adjusting your filters or search query.")
    else:
        st.markdown(f"**Displaying {len(results['documents'])} entries on this page**")

        for i, (doc_id, doc, meta) in enumerate(zip(
            results['ids'],
            results['documents'],
            results['metadatas']
        ), 1):
            entry_num = (results['page'] - 1) * results['page_size'] + i

            with st.container():
                st.markdown(f"#### Entry #{entry_num}")

                # Metadata in columns
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                with meta_col1:
                    st.metric("ID", doc_id[:15] + "..." if len(doc_id) > 15 else doc_id)
                with meta_col2:
                    chapter_name = meta.get('chapter', 'N/A')
                    st.metric("Chapter", chapter_name[:25] + "..." if len(chapter_name) > 25 else chapter_name)
                with meta_col3:
                    st.metric("Filename", meta.get('filename', 'N/A'))
                with meta_col4:
                    st.metric("Chunk Index", meta.get('chunk_index', 'N/A'))

                # Content preview
                preview_length = 300
                preview = doc[:preview_length]
                if len(doc) > preview_length:
                    preview += "..."

                st.text_area(
                    "Content Preview",
                    value=preview,
                    height=120,
                    disabled=True,
                    key=f"preview_{doc_id}_{entry_num}",
                    label_visibility="collapsed"
                )

                st.divider()

    # Pagination controls
    if results['total_pages'] > 0:
        st.markdown("### 📑 Navigation")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button(
                "◄ Previous Page",
                disabled=(st.session_state.page <= 1),
                use_container_width=True
            ):
                st.session_state.page -= 1
                st.rerun()

        with col2:
            st.markdown(
                f"<center><strong>Page {results['page']} of {results['total_pages']}</strong></center>",
                unsafe_allow_html=True
            )

            # Jump to page
            page_jump = st.number_input(
                "Jump to page:",
                min_value=1,
                max_value=max(1, results['total_pages']),
                value=st.session_state.page,
                step=1,
                label_visibility="collapsed"
            )
            if page_jump != st.session_state.page:
                st.session_state.page = page_jump
                st.rerun()

        with col3:
            if st.button(
                "Next Page ►",
                disabled=(st.session_state.page >= results['total_pages']),
                use_container_width=True
            ):
                st.session_state.page += 1
                st.rerun()

    # Export functionality
    if results['documents']:
        st.divider()
        st.markdown("### 📥 Export Data")

        export_data = {
            'metadata': {
                'total_database_entries': total_count,
                'filtered_entries': results['total_count'],
                'page': results['page'],
                'page_size': results['page_size'],
                'filters': where_clause,
                'search_query': search_text if search_text else None,
                'export_timestamp': datetime.now().isoformat(),
                'entries_in_export': len(results['documents'])
            },
            'entries': [
                {
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta
                }
                for doc_id, doc, meta in zip(
                    results['ids'],
                    results['documents'],
                    results['metadatas']
                )
            ]
        }

        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

        st.download_button(
            label=f"📥 Download Current Page as JSON ({len(results['documents'])} entries)",
            data=json_str,
            file_name=f"database_export_page_{results['page']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

        if len(results['documents']) > 0 and results['total_count'] <= 500:
            st.info(
                f"💡 Tip: Your current filters match {results['total_count']} total entries. "
                "You can export all filtered results by navigating through all pages."
            )

    # Sidebar with information
    with st.sidebar:
        st.markdown("## 📊 Database Statistics")
        st.info(
            f"""
            **Collection Info:**
            - Total Entries: {total_count:,}
            - Indexed Chapters: 7
            - Embedding Dimension: 384
            - Chunk Strategy: Sliding window (100 chars, 50% overlap)

            **Features:**
            - Browse entries with pagination
            - Filter by chapter
            - Search text within documents
            - Export filtered results to JSON

            **Performance:**
            - Page size: 50 entries
            - Results cached for fast browsing
            """
        )


if __name__ == "__main__":
    main()
