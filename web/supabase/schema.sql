-- NLP Textbook RAG — Supabase pgvector schema
-- Run once in the Supabase SQL editor (or via `supabase db push`).
-- 10,170 chunks, 384-dim MiniLM vectors (already L2-normalized).

create extension if not exists vector;

create table if not exists nlp_chunks (
  id           text primary key,        -- e.g. "chunk_0" (stable id from Chroma export)
  document     text not null,           -- chunk text, prefixed "Chapter: {title}\n\n..."
  chapter      text,                    -- raw chapter label (polluted; filename is the clean key)
  filename     text,                    -- clean source key, e.g. "words-and-tokens.txt"
  chunk_index  int,
  document_id  text,                    -- source chapter id
  embedding    vector(384) not null
);

-- HNSW cosine index. Vectors are unit-normalized, so cosine distance (<=>)
-- ranks identically to the original Chroma cosine search.
create index if not exists nlp_chunks_embedding_idx
  on nlp_chunks using hnsw (embedding vector_cosine_ops);

create index if not exists nlp_chunks_filename_idx on nlp_chunks (filename);

-- Top-k similarity search. Returns cosine similarity in [−1, 1]
-- (1 for identical) so the serving layer can reapply the adaptive threshold.
-- Distinct chapter source files, for the Database Viewer's filter dropdown.
create or replace function nlp_filenames()
returns table (filename text)
language sql stable
as $$
  select distinct c.filename from nlp_chunks c order by c.filename;
$$;

create or replace function match_nlp_chunks(
  query_embedding vector(384),
  match_count int default 5
)
returns table (
  id          text,
  document    text,
  chapter     text,
  filename    text,
  chunk_index int,
  similarity  float
)
language sql stable
as $$
  select
    c.id,
    c.document,
    c.chapter,
    c.filename,
    c.chunk_index,
    1 - (c.embedding <=> query_embedding) as similarity
  from nlp_chunks c
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
