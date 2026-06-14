#!/usr/bin/env python3
"""
One-time offline export of the existing ChromaDB collection to JSONL.

Pulls every chunk's raw 384-dim MiniLM vector (already L2-normalized) plus its
document text and metadata, so the TS ingestion script can upsert them straight
into Supabase pgvector — no re-embedding, identical retrieval to the original app.

Drops the `file_path` metadata field (leaks an absolute local path).

Output: data/export/chunks.jsonl  (one JSON object per line)
  { "id", "document", "embedding": [384 floats], "chapter", "filename",
    "chunk_index", "document_id" }
"""

import json
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "data" / "chroma"
OUT_PATH = ROOT / "data" / "export" / "chunks.jsonl"
COLLECTION = "nlp_textbook"
BATCH = 1000


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(COLLECTION)
    total = col.count()
    print(f"Collection '{COLLECTION}': {total} chunks")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen_dims = set()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for offset in range(0, total, BATCH):
            g = col.get(
                limit=BATCH,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            for cid, doc, emb, meta in zip(
                g["ids"], g["documents"], g["embeddings"], g["metadatas"]
            ):
                vec = [float(x) for x in emb]
                seen_dims.add(len(vec))
                row = {
                    "id": cid,
                    "document": doc,
                    "embedding": vec,
                    "chapter": meta.get("chapter", ""),
                    "filename": meta.get("filename", ""),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "document_id": meta.get("document_id", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            print(f"  exported {min(offset + BATCH, total)}/{total}")

    print(f"\nDone: {written} rows -> {OUT_PATH}")
    print(f"Embedding dims seen: {seen_dims}")
    assert seen_dims == {384}, f"Unexpected embedding dims: {seen_dims}"
    assert written == total, f"Row mismatch: wrote {written}, expected {total}"


if __name__ == "__main__":
    main()
