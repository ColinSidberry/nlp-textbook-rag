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
  /** Markdown write-up fallback. The landing renders a bespoke <NlpRagBody /> instead. */
  writeup?: string;
}

export const projectConfig: ProjectConfig = {
  name: 'NLP Textbook RAG',
  tagline:
    'Ask the canonical NLP textbook a question and get an answer grounded only in its pages, with citations and the exact passages, nothing made up.',
  stack: ['Next.js', 'Supabase pgvector', 'Groq', 'MiniLM embeddings', 'Vercel'],
  liveHref: '/ask',
  codeHref: '/code',
  dbHref: '/viewer',
  demoVideoId: '',
  pipeline: ['Chunk', 'Embed (MiniLM)', 'pgvector search', 'Ground', 'Groq answer'],
};
