'use client';

import ReactMarkdown from 'react-markdown';
import { ArrowUpRight, ChevronRight } from 'lucide-react';
import { SiteHeader } from './SiteHeader';
import type { ProjectConfig } from './config';

export type { ProjectConfig } from './config';

/**
 * Reusable project-page template shared across portfolio projects
 * (writing-agent, homes, nlp-rag). Drive it with a ProjectConfig.
 *
 * Persistent SiteHeader (Demo · Live · Code · Database) + hero + pipeline strip +
 * markdown write-up + closing CTA. The same SiteHeader rides every sub-view.
 */

export function ProjectLanding({ config }: { config: ProjectConfig }) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <SiteHeader config={config} />

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
    </div>
  );
}
