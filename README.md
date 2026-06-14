# NLP Textbook RAG

Retrieval-augmented Q&A over Jurafsky & Martin's *Speech and Language Processing*
(35 chapters). Ask a question, get an answer grounded **only** in the textbook,
with citations and the exact retrieved passages — nothing is made up.

**Live:** [nlp-rag.colinsidberry.com](https://nlp-rag.colinsidberry.com) · self-contained on Vercel · free to run.

---

## Why I rewrote it

The original version worked but wasn't truly *live*. It was Python + Streamlit +
ChromaDB, with generation hardcoded to **Mistral-7B via Ollama on a paid GCP VM**.
That VM was the blocker: always-on infrastructure I had to pay for and babysit, so
the app could never just sit at a URL for free.

This rewrite is fully **Vercel-native TypeScript** — no Python at serve time, no
ChromaDB, no Ollama, no separate backend host:

- **Query embedding via a hosted endpoint** running the *exact same* MiniLM model
  the corpus was built with (`all-MiniLM-L6-v2`) — verified: query vectors match
  the original Chroma vectors to cosine 1.0, so the existing vectors are reused
  with no re-embed and retrieval is unchanged.
- **Vectors live in Supabase pgvector** (free tier).
- **Generation is bring-your-own-key**: a free Groq model by default, or paste your
  own Anthropic / OpenAI key. BYOK is the through-line — the app costs nothing to run.

Everything the original did well is preserved: the anti-hallucination prompt, the
adaptive relevance threshold, acronym expansion, citations, and the Database Viewer.

## Architecture

```
query
  → acronym/comparison expansion
  → embed with transformers.js MiniLM (384-dim, in-function)
  → pgvector cosine search (top-5) in Supabase
  → adaptive relevance threshold (out-of-scope guard)
  → 7-point anti-hallucination prompt
  → LLM (Groq default, or BYOK Anthropic/OpenAI)
  → { answer, citations, chunks }
```

| Layer | Choice |
|---|---|
| Frontend / API | Next.js 16 (App Router) + Tailwind, in `web/` |
| Vector store | Supabase `pgvector`, HNSW cosine index |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace Inference API (free, same model as the corpus) |
| Default LLM | Groq `llama-3.3-70b-versatile` (free tier) |
| BYOK | Anthropic (`claude-opus-4-8`) / OpenAI, visitor-supplied key, never stored |
| Auth | Shared `/code`-style password cookie, `.colinsidberry.com` SSO |

Corpus: **10,170 chunks** (280-char sliding window, 20% overlap), one source file
per chapter. Retrieval is bit-identical to the original because the document vectors
are reused verbatim from the original ChromaDB index.

## Project layout

```
data/normalized/nlp_textbook.json   committed source (35 chapters)
scripts/dump_chroma.py              one-time: export Chroma vectors → JSONL
web/                                the Next.js app (deploy root)
  supabase/schema.sql               nlp_chunks table + match/filenames RPCs
  scripts/ingest.ts                 load JSONL → pgvector (npm run ingest)
  src/lib/                          embedder, expand, supabase, rag, llm, auth
  src/app/                          /, /login, /viewer, /api/{query,login,chunks}
.github/workflows/keepalive.yml     daily ping so the free DB never pauses
```

## Local development

```bash
cd web
cp .env.local.example .env.local   # fill in the values below
npm install
npm run ingest                     # one-time: apply schema + load 10,170 rows
npm run dev
```

Environment (`web/.env.local`, all server-only — never `NEXT_PUBLIC_`):

| Var | What |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase secret key (server-side) |
| `DATABASE_URL` | Direct Postgres URL — only used by `npm run ingest` |
| `GROQ_API_KEY` | Free default LLM key ([console.groq.com](https://console.groq.com)) |
| `HF_TOKEN` | Free HuggingFace token for query embedding ([hf.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `SITE_SECRET`, `SITE_PASSWORD` | Password gate (shared across sites) |
| `COOKIE_DOMAIN` | `.colinsidberry.com` in prod; blank locally |

## Re-ingesting from scratch

The vectors are derived once, offline, from the original ChromaDB index:

```bash
# from repo root, with the legacy Python venv
./venv/bin/python scripts/dump_chroma.py     # → data/export/chunks.jsonl (10,170 rows)
cd web && npm run ingest                      # → Supabase pgvector
```

## Deploy

Deployed on Vercel with the project root set to `web/`. Set the env vars above in
the Vercel dashboard (never committed). The Supabase free project pauses after ~7
days of inactivity, so `.github/workflows/keepalive.yml` pings it daily — it needs
two repo secrets: `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
