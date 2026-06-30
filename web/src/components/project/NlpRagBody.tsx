import type { ReactNode } from 'react';
import {
  MessageCircleQuestion,
  Binary,
  Database,
  ShieldCheck,
  Sparkles,
  ChevronDown,
  Ban,
} from 'lucide-react';
import { Logo } from './media';

/* ------------------------------------------------------------------ helpers */

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-14">
      <h2 className="mb-5 border-b border-border pb-2 text-2xl font-semibold tracking-tight">
        {title}
      </h2>
      {children}
    </section>
  );
}

function P({ children }: { children: ReactNode }) {
  return <p className="mb-4 leading-[1.75] text-muted-foreground">{children}</p>;
}

/* --------------------------------------------------------------------- data */

type Kind = 'io' | 'brand';
type Step = { n: number; name: string; desc: string; icon: typeof Binary; kind: Kind };

const STEPS: Step[] = [
  { n: 1, name: 'Question', desc: 'your question, in plain English', icon: MessageCircleQuestion, kind: 'io' },
  { n: 2, name: 'Embed', desc: 'MiniLM encodes it into a 384-dim vector', icon: Binary, kind: 'brand' },
  { n: 3, name: 'Retrieve', desc: 'pgvector pulls the top-5 nearest of 10,170 chunks', icon: Database, kind: 'brand' },
];

const GATE: Step = {
  n: 4,
  name: 'Relevance gate',
  desc: 'an adaptive threshold drops weak matches',
  icon: ShieldCheck,
  kind: 'brand',
};

const ANSWER: Step = {
  n: 5,
  name: 'Grounded answer',
  desc: 'top passages + anti-hallucination prompt → LLM → answer with citations',
  icon: Sparkles,
  kind: 'brand',
};

type TechItem = { name: string; slug?: string; src?: string; wide?: boolean; mono?: boolean };
const TECH: { group: string; items: TechItem[] }[] = [
  {
    group: 'Frontend',
    items: [
      { name: 'Next.js', slug: 'nextdotjs' },
      { name: 'React', slug: 'react' },
      { name: 'Tailwind', slug: 'tailwindcss' },
    ],
  },
  {
    group: 'Vector store',
    items: [
      { name: 'Supabase', slug: 'supabase' },
      { name: 'pgvector', slug: 'postgresql' },
    ],
  },
  {
    group: 'Embeddings',
    items: [
      { name: 'transformers.js', slug: 'huggingface' },
      { name: 'MiniLM' },
    ],
  },
  {
    group: 'LLM (BYOK)',
    items: [
      { name: 'Groq', src: '/groq.png', wide: true, mono: true },
      { name: 'Anthropic', slug: 'anthropic' },
      { name: 'OpenAI', src: '/openai.svg', mono: true },
    ],
  },
  {
    group: 'Hosting',
    items: [
      { name: 'Vercel', slug: 'vercel' },
      { name: 'GitHub Actions', slug: 'githubactions' },
    ],
  },
];

/* ----------------------------------------------------------------- diagram */

function StepCard({ step }: { step: Step }) {
  const Icon = step.icon;
  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-3">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted font-mono text-xs font-semibold text-muted-foreground">
        {step.n}
      </span>
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-muted">
        <Icon className={`h-4 w-4 ${step.kind === 'brand' ? 'text-brand' : 'text-muted-foreground'}`} />
      </span>
      <span className="min-w-0">
        <span className="block font-mono text-sm font-semibold text-foreground">{step.name}</span>
        <span className="block text-xs text-muted-foreground">{step.desc}</span>
      </span>
    </div>
  );
}

function Connector({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center py-1">
      {label && <span className="text-[11px] text-muted-foreground/70">{label}</span>}
      <ChevronDown className="h-4 w-4 text-muted-foreground/40" />
    </div>
  );
}

/** Top-down RAG pipeline. At the relevance gate, a dashed branch peels off to a
 *  muted "Not covered" card; the main path continues to the grounded answer. */
function RagFlow() {
  return (
    <div className="my-7">
      {STEPS.map((s, i) => (
        <div key={s.n}>
          <StepCard step={s} />
          {i < STEPS.length - 1 && <Connector />}
        </div>
      ))}

      <Connector />

      {/* the gate forks: weak matches peel off to "Not covered" */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
        <div className="flex-1">
          <StepCard step={GATE} />
        </div>

        <div className="hidden self-center text-muted-foreground/40 sm:flex sm:items-center sm:gap-1">
          <span className="h-px w-6 border-t border-dashed border-border" />
        </div>

        <div className="flex items-start gap-2 rounded-xl border border-dashed border-border bg-muted/40 px-3 py-3 sm:w-56 sm:shrink-0">
          <Ban className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="min-w-0">
            <span className="block text-sm font-semibold text-foreground">Not covered</span>
            <span className="block text-[11px] leading-snug text-muted-foreground">
              weak matches stop here: the book doesn&apos;t address it, so it says so instead of guessing
            </span>
          </span>
        </div>
      </div>

      <Connector label="passes the bar" />

      <StepCard step={ANSWER} />

      {/* legend */}
      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Database className="h-3.5 w-3.5 text-brand" /> 10,170 indexed chunks
        </span>
        <span className="flex items-center gap-1.5">
          <ShieldCheck className="h-3.5 w-3.5 text-brand" /> out-of-scope gate
        </span>
        <span className="flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5 text-brand" /> the one LLM call
        </span>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- body */

export function NlpRagBody() {
  return (
    <div>
      {/* 1 — Problem */}
      <Section title="The problem">
        <P>
          I took a natural language processing class as part of my master&apos;s, and I wanted to
          actually understand how RAG works under the hood, not just import a library and call it
          done. So I built one end to end.
        </P>
        <P>
          The example is almost too simple, which is kind of the point. The canonical NLP reference,
          Jurafsky &amp; Martin&apos;s <em>Speech and Language Processing</em>, runs 35 chapters and
          about 1,000 pages. When you have a specific question (&ldquo;how does attention
          work?&rdquo;, &ldquo;what&apos;s an n-gram?&rdquo;), a PDF is a bad interface. You want a
          direct answer. The obvious shortcut, asking a general chatbot, swaps that for a worse
          problem: it answers confidently from its own training, and you can&apos;t tell whether
          it&apos;s quoting the book or making something up.
        </P>
        <P>
          So the real exercise was RAG you can actually trust: every answer grounded in the textbook
          itself, the exact passages shown, and an honest &ldquo;not covered&rdquo; when the book
          doesn&apos;t address it.
        </P>
      </Section>

      {/* 2 — How it works (the diagram is the centerpiece) */}
      <Section title="How it works">
        <P>
          Ask a question and it runs through a fixed retrieval-augmented pipeline. Nothing in the
          answer comes from the model&apos;s own training: it only sees passages pulled from the
          book, and if nothing relevant comes back, it says so.
        </P>

        <RagFlow />

        <P>
          The corpus is the whole textbook split into 10,170 passages (about 280 characters each,
          20% overlap so a concept that straddles a boundary still lands intact in some chunk).
          Questions and passages are embedded with the <em>same</em> MiniLM model, which is the only
          reason the cosine similarity means anything. The final step hands the LLM a strict
          anti-hallucination prompt: answer only from the supplied passages, cite the chapters you
          drew from.
        </P>
      </Section>

      {/* 3 — Architecture */}
      <Section title="Architecture">
        <P>
          The whole thing is Vercel-native TypeScript. The earlier version worked but couldn&apos;t
          really sit live: Python + Streamlit + ChromaDB, with generation pinned to a Mistral model
          on an always-on, paid GCP VM, and that VM was the blocker. I reused the document vectors
          verbatim from the old ChromaDB index (no re-embed) and verified the query vectors match it
          to cosine 1.0, so retrieval is identical to the original build. Everything else runs on
          free tiers.
        </P>
        <P>
          Query embedding happens in-request through a hosted MiniLM endpoint, so there&apos;s no
          Python and no separate model server. Vectors live in Supabase{' '}
          <code className="rounded bg-brand/5 px-1 py-0.5 font-mono text-sm text-brand">pgvector</code>,
          kept warm by a daily GitHub Action so the database never pauses. Answers default to a free
          Groq model, or bring your own key for Anthropic or OpenAI, used for that one request and
          never stored. BYOK is the through-line: the whole app costs nothing to sit live at a URL.
        </P>
        <P>
          Really this is a backend data pipeline, the kind of thing that usually runs in a terminal.
          To make it something you can actually look at, the app puts two surfaces over it. A live
          run you drive yourself: ask a question, it embeds, retrieves, and answers with Groq, with
          the exact passages it pulled shown inline. And a database viewer that browses all 10,170
          indexed chunks directly, so the index isn&apos;t a black box either.
        </P>

      </Section>

      {/* 4 — Technologies */}
      <Section title="Technologies">
        <div className="grid gap-4 sm:grid-cols-2">
          {TECH.map((t) => (
            <div
              key={t.group}
              className="flex min-h-52 flex-col rounded-xl border border-border bg-card/40 px-4 py-5"
            >
              <div className="mb-4 text-center font-mono text-xs uppercase tracking-wider text-muted-foreground/70">
                {t.group}
              </div>
              <div className="mx-auto flex max-w-[12rem] flex-1 flex-wrap content-center items-center justify-center gap-4">
                {t.items.map((it) => (
                  <Logo key={it.name} name={it.name} slug={it.slug} src={it.src} wide={it.wide} mono={it.mono} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
