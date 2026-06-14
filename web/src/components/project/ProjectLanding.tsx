'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { ArrowLeft, Play, ArrowUpRight, Code2, Database, ChevronRight, X } from 'lucide-react';
import { ThemeToggle } from '@/components/ui/theme-toggle';

/**
 * Reusable project-page template shared across portfolio projects
 * (writing-agent, homes, nlp-rag). Drive it with a ProjectConfig.
 *
 * Sticky centered nav (Demo · Live · Code · Database) + hero + pipeline strip +
 * markdown write-up + closing CTA. "Demo" opens an in-page video modal.
 */
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

const navCls =
  'inline-flex items-center gap-1.5 text-sm font-medium px-2.5 py-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors';

export function ProjectLanding({ config }: { config: ProjectConfig }) {
  const [demoOpen, setDemoOpen] = useState(false);
  const hub = config.hubUrl ?? 'https://colinsidberry.com';

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Sticky header nav */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-3xl mx-auto px-4 h-14 grid grid-cols-[1fr_auto_1fr] items-center">
          <a href={hub} aria-label="Back to portfolio" className="justify-self-start text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="h-5 w-5" />
          </a>

          <nav className="justify-self-center flex items-center gap-0.5 sm:gap-1">
            <button onClick={() => setDemoOpen(true)} className={navCls}>
              <Play className="h-4 w-4 text-brand" />
              <span className="hidden sm:inline">Demo</span>
            </button>
            <a href={config.liveHref} className={navCls}>
              <ArrowUpRight className="h-4 w-4" />
              <span className="hidden sm:inline">Live</span>
            </a>
            <a href={config.codeHref} className={navCls}>
              <Code2 className="h-4 w-4" />
              <span className="hidden sm:inline">Code</span>
            </a>
            <a href={config.dbHref} className={navCls}>
              <Database className="h-4 w-4" />
              <span className="hidden sm:inline">Database</span>
            </a>
          </nav>

          <div className="justify-self-end">
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-5">
        {/* Hero */}
        <section className="pt-16 sm:pt-20 pb-8">
          <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight leading-[1.05]">{config.name}</h1>
          <p className="mt-5 text-lg sm:text-xl text-muted-foreground max-w-2xl leading-relaxed">{config.tagline}</p>
        </section>

        {/* Pipeline strip */}
        {config.pipeline && config.pipeline.length > 0 && (
          <div className="flex flex-wrap items-center gap-y-2 pb-12">
            {config.pipeline.map((s, i) => (
              <span key={s} className="flex items-center">
                <span className="font-mono text-xs rounded-full border border-brand/40 bg-brand/5 text-brand px-2.5 py-1">
                  {s}
                </span>
                {i < config.pipeline!.length - 1 && (
                  <ChevronRight className="h-4 w-4 text-muted-foreground/40 mx-0.5 shrink-0" />
                )}
              </span>
            ))}
          </div>
        )}

        {/* Write-up */}
        <article
          className="max-w-none
            [&_h2]:text-2xl [&_h2]:font-semibold [&_h2]:tracking-tight [&_h2]:mt-12 [&_h2]:mb-3 [&_h2]:scroll-mt-20
            [&_h2]:pb-2 [&_h2]:border-b [&_h2]:border-border
            [&_h3]:text-base [&_h3]:font-semibold [&_h3]:mt-6 [&_h3]:mb-2
            [&_p]:text-muted-foreground [&_p]:leading-[1.75] [&_p]:mb-4
            [&_ul]:list-none [&_ul]:pl-0 [&_ul]:mb-4 [&_ul]:space-y-2
            [&_li]:text-muted-foreground [&_li]:pl-6 [&_li]:relative
            [&_li]:before:content-['→'] [&_li]:before:absolute [&_li]:before:left-0 [&_li]:before:text-brand
            [&_strong]:text-foreground [&_strong]:font-semibold
            [&_code]:font-mono [&_code]:text-sm [&_code]:text-brand [&_code]:bg-brand/5 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded
            [&_a]:text-foreground [&_a]:underline [&_a]:underline-offset-2"
        >
          <ReactMarkdown>{config.writeup}</ReactMarkdown>
        </article>

        {/* Closing CTA */}
        <div className="mt-10 pb-24">
          <a
            href={config.liveHref}
            className="inline-flex items-center gap-2 rounded-lg bg-brand text-brand-foreground px-5 py-2.5 text-sm font-semibold hover:opacity-90 transition-opacity"
          >
            See it live
            <ArrowUpRight className="h-4 w-4" />
          </a>
        </div>
      </main>

      <footer className="border-t border-border">
        <div className="max-w-3xl mx-auto px-5 py-6 flex flex-wrap justify-between gap-2 font-mono text-xs text-muted-foreground">
          <span>Colin Sidberry</span>
          <span>{config.stack.join(' · ')}</span>
        </div>
      </footer>

      {/* Demo modal */}
      {demoOpen && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4" onClick={() => setDemoOpen(false)}>
          <div
            className="relative w-full max-w-3xl aspect-video bg-card rounded-xl border border-border overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setDemoOpen(false)}
              aria-label="Close"
              className="absolute top-2 right-2 z-10 text-white/80 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
            {config.demoVideoId ? (
              <iframe
                className="w-full h-full"
                src={`https://www.youtube.com/embed/${config.demoVideoId}?autoplay=1`}
                title={`${config.name} demo`}
                allow="accelerated-encoder; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-center gap-2 px-6">
                <div className="font-mono text-sm text-muted-foreground">demo video coming soon</div>
                <a href={config.liveHref} className="font-mono text-sm text-brand underline underline-offset-2">
                  → try it live instead
                </a>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
