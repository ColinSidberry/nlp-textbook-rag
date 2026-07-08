import type { ProjectConfig } from './config';

/**
 * Shared site footer — Colin Sidberry (left) + the project tech stack (right).
 * Extracted from ProjectLanding so every document view (/ask, /viewer, /code,
 * /login) can render the same footer. nlp-rag has no full-bleed map, so all
 * pages get it.
 */
export function SiteFooter({ config }: { config: ProjectConfig }) {
  return (
    <footer className="border-t border-border">
      <div className="max-w-3xl mx-auto px-5 py-6 flex flex-wrap justify-between gap-2 font-mono text-xs text-muted-foreground">
        <span>Colin Sidberry</span>
        <span>{config.stack.join(' · ')}</span>
      </div>
    </footer>
  );
}
