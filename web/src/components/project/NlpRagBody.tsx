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
  { n: 1, name: 'Question', desc: 'your question, in plain English', icon: MessageCircleQuestion, kind: 'brand' },
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
    group: 'LLM',
    items: [
      { name: 'Groq', src: '/groq.png', wide: true, mono: true },
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

/** Tree fork under the gate: one stem drops from the center, splits into a
 *  horizontal bar, and two drops line up over the two branch columns below.
 *  Desktop only; on mobile the branches just stack under a plain connector. */
function Fork() {
  return (
    <>
      <div className="sm:hidden">
        <Connector />
      </div>
      <div className="relative hidden h-6 sm:block" aria-hidden>
        <div className="absolute left-1/2 top-0 h-3 w-px -translate-x-1/2 bg-border" />
        <div className="absolute left-1/4 right-1/4 top-3 border-t border-border" />
        <div className="absolute left-1/4 top-3 h-3 w-px -translate-x-1/2 bg-border" />
        <div className="absolute left-3/4 top-3 h-3 w-px -translate-x-1/2 bg-border" />
      </div>
    </>
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

      <StepCard step={GATE} />

      {/* the gate forks like a tree: too weak dead-ends at "Not covered",
          otherwise the strong matches continue to the grounded answer */}
      <Fork />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {/* dead-end branch */}
        <div>
          <span className="mb-1.5 block text-center text-[11px] uppercase tracking-wider text-muted-foreground/60">
            too weak
          </span>
          <div className="flex items-start gap-3 rounded-xl border border-dashed border-border bg-muted/40 px-4 py-3">
            <Ban className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <span className="min-w-0">
              <span className="block text-sm font-semibold text-foreground">Not covered</span>
              <span className="block text-xs leading-snug text-muted-foreground">
                it stops here: the book doesn&apos;t address it, so it says so instead of guessing
              </span>
            </span>
          </div>
        </div>

        {/* answer branch */}
        <div>
          <span className="mb-1.5 block text-center text-[11px] uppercase tracking-wider text-muted-foreground/60">
            passes
          </span>
          <StepCard step={ANSWER} />
        </div>
      </div>

      {/* legend */}
      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Database className="h-3.5 w-3.5 text-brand" /> 10,170 indexed chunks
        </span>
        <span className="flex items-center gap-1.5">
          <ShieldCheck className="h-3.5 w-3.5 text-brand" /> out-of-scope gate
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
          I took a natural language processing class for my master&apos;s, and I wanted to actually
          understand how RAG works under the hood. So I built one end to end.
        </P>
        <P>
          The goal was a system built on our course textbook, Jurafsky &amp; Martin&apos;s{' '}
          <em>Speech and Language Processing</em>{' '}(35 chapters, ~1,000 pages), where the LLM answers
          from the book itself rather than from whatever it absorbed during training. Every answer
          traces back to the actual pages, and when the book doesn&apos;t cover something, it says
          so instead of guessing.
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
          The whole app is serverless TypeScript on Vercel: no standing server, and no Python even
          for the embeddings. It breaks into three layers.
        </P>

        <h3 className="mb-2 mt-7 text-base font-semibold text-foreground">Frontend</h3>
        <p className="mb-3 leading-[1.75] text-muted-foreground">
          A Next.js app with two views over the pipeline:
        </p>
        <ol className="mb-4 ml-5 list-decimal space-y-2 leading-[1.75] text-muted-foreground marker:font-mono marker:text-muted-foreground/70">
          <li>
            <span className="font-medium text-foreground">A live run</span> you drive yourself: ask a
            question and see the answer with the exact passages it retrieved shown inline.
          </li>
          <li>
            <span className="font-medium text-foreground">A database viewer</span>{' '}that browses
            all 10,170 indexed chunks directly, so the index isn&apos;t a black box.
          </li>
        </ol>

        <h3 className="mb-2 mt-7 text-base font-semibold text-foreground">Backend</h3>
        <P>
          Retrieval runs in Vercel serverless functions, with no model server of its own
          (a past life of this ran generation via Ollama on an always-on, paid GCP VM; going
          serverless is what makes this version cheap enough to keep live). The
          question is embedded in-request through a hosted MiniLM endpoint, the same model used on
          the corpus, so the cosine distances are actually comparable. The vectors live in Supabase{' '}
          pgvector, which runs the nearest-neighbor search, and a daily GitHub Action pings the database so the
          free tier never pauses.
        </P>

        <h3 className="mb-2 mt-7 text-base font-semibold text-foreground">LLM</h3>
        <P>
          The final step sends the retrieved passages to a generative model. It runs on a free Groq
          model today, with bring-your-own-model support as a planned next step. The model gets a
          strict anti-hallucination prompt: answer only from the supplied passages, and cite the
          chapters it drew from.
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
