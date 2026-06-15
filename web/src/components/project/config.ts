// Shared project config — drives the landing page (ProjectLanding) AND the
// persistent SiteHeader on every sub-view (/ask, /viewer, /code). Kept in its
// own module so sub-pages can import the same nav targets the landing uses.

export interface ProjectConfig {
  name: string;
  tagline: string;
  stack: string[];
  /** Back-arrow target — the portfolio hub. */
  hubUrl?: string;
  liveHref: string;
  codeHref: string;
  dbHref: string;
  /** YouTube video id for the embedded demo. Empty = "coming soon" state. */
  demoVideoId?: string;
  /** Stage names rendered as a cobalt chip flow under the hero. */
  pipeline?: string[];
  /** Markdown write-up rendered as the page body. */
  writeup: string;
}

const writeup = `
## The problem

A textbook is a terrible interface for a question. Jurafsky & Martin's *Speech and
Language Processing* is the canonical NLP reference — 35 chapters, ~1,000 pages — and
when you actually have a question ("how does attention work?", "what's an n-gram?")
you don't want to skim a PDF. You want a direct answer. But the usual fix, asking a
general chatbot, trades one problem for a worse one: it answers confidently from its
own training, with no way to know whether it's quoting the book or making something up.

## What it does

Ask a question and get an answer grounded **only** in the textbook — with citations and
the exact passages it used, shown inline. If the book doesn't cover something, the app
says so instead of inventing an answer. Every answer is auditable: the retrieved chunks
are right there, and the **Database** viewer lets you browse all 10,170 indexed passages
directly.

## How it works

This is retrieval-augmented generation (RAG) with an anti-hallucination contract:

- **Chunking** — each chapter is split into ~280-character passages with 20% overlap, so
  a concept that straddles a boundary still lands intact in at least one chunk. That
  yields **10,170 chunks**, one source file per chapter.
- **Embeddings** — both the corpus and each incoming question are encoded with the same
  \`all-MiniLM-L6-v2\` model (384-dim). Using the *identical* model for documents and
  queries is what makes cosine similarity meaningful — the query vectors were verified to
  match the original index to cosine 1.0, so retrieval is bit-identical to the source build.
- **pgvector search** — vectors live in Supabase Postgres with an HNSW cosine index; the
  top-5 nearest passages are pulled per query.
- **Out-of-scope guard** — an adaptive relevance threshold rejects weak matches, so a
  question the book doesn't address returns "not covered" rather than a forced answer.
- **Grounded generation** — the retrieved passages plus a 7-point anti-hallucination
  prompt go to the LLM, which is instructed to answer strictly from the supplied context
  and cite the chapters it drew from. Acronyms and comparison phrasings are expanded first
  to sharpen retrieval.

## Architecture

- **Frontend / API** — Next.js (App Router) on Vercel. Query embedding runs in the request
  via a hosted MiniLM endpoint — no Python, no separate model server.
- **Vector store** — Supabase \`pgvector\`, free tier, kept warm by a daily GitHub Action
  so the database never pauses.
- **LLM** — a free Groq model (\`llama-3.3-70b-versatile\`) by default, or **bring your own
  key** for Anthropic / OpenAI. The key is used only for that request and never stored.
  BYOK is the through-line: the whole app costs nothing to sit live at a URL.

## Why I rewrote it

The original worked but wasn't truly *live* — it was Python + Streamlit + ChromaDB with
generation pinned to a Mistral model on an always-on, paid GCP VM. That VM was the
blocker. This rewrite is fully Vercel-native TypeScript: the document vectors are reused
verbatim from the original ChromaDB index (no re-embed), everything else is free-tier, and
all the parts that made the original good — the anti-hallucination prompt, the adaptive
threshold, acronym expansion, citations, the database viewer — are preserved.

## What's next

- **Streaming answers** so the response renders token-by-token instead of all at once.
- **Reranking** the top-k passages with a cross-encoder before generation to lift precision.
- **Chapter-scoped questions** — let a reader pin retrieval to a specific chapter or section.
`;

export const projectConfig: ProjectConfig = {
  name: 'NLP Textbook RAG',
  tagline:
    'Ask the canonical NLP textbook a question and get an answer grounded only in its pages — with citations and the exact passages, nothing made up.',
  stack: ['Next.js', 'Supabase pgvector', 'Groq', 'MiniLM embeddings', 'Vercel'],
  liveHref: '/ask',
  codeHref: '/code',
  dbHref: '/viewer',
  demoVideoId: '',
  pipeline: ['Chunk', 'Embed (MiniLM)', 'pgvector search', 'Ground', 'Groq answer'],
  writeup,
};
