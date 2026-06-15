'use client';

import { useState, type ReactNode } from 'react';
import { ArrowLeft, Play, ArrowUpRight, Code2, Database, X } from 'lucide-react';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import type { ProjectConfig } from './config';

/**
 * Persistent project header shared by the landing page and every sub-view
 * (/ask, /viewer, /code). Renders: back-to-hub arrow, the centered
 * Demo · Live · Code · Database nav (current section highlighted), and the
 * ThemeToggle. "Demo" opens the video modal in place — the modal lives here so
 * the behavior is identical on the landing and on every sub-page.
 */

export type Section = 'demo' | 'live' | 'code' | 'database';

const base =
  'inline-flex items-center gap-1.5 text-sm font-medium px-2.5 py-1.5 rounded-lg transition-colors';

export function SiteHeader({
  config,
  active,
  fluid = false,
  variant = 'default',
  rightSlot,
}: {
  config: ProjectConfig;
  active?: Section;
  /** Full-width bar (app/sub-pages) vs the narrow max-w-3xl column (landing). */
  fluid?: boolean;
  /** 'dark' blends with the GitHub-style code skin (always-dark surface). */
  variant?: 'default' | 'dark';
  /** Extra controls pinned to the right of the ThemeToggle (e.g. a logout button). */
  rightSlot?: ReactNode;
}) {
  const [demoOpen, setDemoOpen] = useState(false);
  const hub = config.hubUrl ?? 'https://colinsidberry.com';
  const dark = variant === 'dark';

  const item = (on: boolean) =>
    dark
      ? `${base} ${on ? 'bg-[#21262d] text-[#e6edf3]' : 'text-[#7d8590] hover:text-[#e6edf3] hover:bg-[#161b22]'}`
      : `${base} ${on ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-muted'}`;

  const headerCls = dark
    ? 'sticky top-0 z-40 border-b border-[#30363d] bg-[#0d1117]'
    : 'sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60';

  const backCls = dark
    ? 'justify-self-start text-[#7d8590] hover:text-[#e6edf3] transition-colors'
    : 'justify-self-start text-muted-foreground hover:text-foreground transition-colors';

  const container = fluid ? 'px-5 h-14' : 'max-w-3xl mx-auto px-4 h-14';

  return (
    <header className={headerCls}>
      <div className={`${container} grid grid-cols-[1fr_auto_1fr] items-center`}>
        <a href={hub} aria-label="Back to portfolio" className={backCls}>
          <ArrowLeft className="h-5 w-5" />
        </a>

        <nav className="justify-self-center flex items-center gap-0.5 sm:gap-1">
          <button onClick={() => setDemoOpen(true)} className={item(active === 'demo')}>
            <Play className="h-4 w-4 text-brand" />
            <span className="hidden sm:inline">Demo</span>
          </button>
          <a href={config.liveHref} className={item(active === 'live')}>
            <ArrowUpRight className="h-4 w-4" />
            <span className="hidden sm:inline">Live</span>
          </a>
          <a href={config.codeHref} className={item(active === 'code')}>
            <Code2 className="h-4 w-4" />
            <span className="hidden sm:inline">Code</span>
          </a>
          <a href={config.dbHref} className={item(active === 'database')}>
            <Database className="h-4 w-4" />
            <span className="hidden sm:inline">Database</span>
          </a>
        </nav>

        <div className="justify-self-end flex items-center gap-2">
          <ThemeToggle />
          {rightSlot}
        </div>
      </div>

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
    </header>
  );
}
