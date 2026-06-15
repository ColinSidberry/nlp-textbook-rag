'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';
import { ChevronDown, ChevronRight, FileCode } from 'lucide-react';
import { SiteHeader } from '@/components/project/SiteHeader';
import { projectConfig } from '@/components/project/config';
import type { CodeData, TreeNode } from '@/lib/code-files';

// Lazy shiki highlighter singleton — only loaded when /code is opened.
let highlighterPromise: Promise<{
  codeToHtml: (code: string, opts: { lang: string; theme: string }) => string;
}> | null = null;

async function highlight(code: string, lang: string, theme: string): Promise<string> {
  if (!highlighterPromise) {
    const { createHighlighter } = await import('shiki');
    highlighterPromise = createHighlighter({
      themes: ['github-light', 'github-dark'],
      langs: ['typescript', 'tsx', 'javascript', 'jsx', 'python', 'json', 'css', 'markdown', 'yaml', 'bash', 'sql'],
    });
  }
  const hl = await highlighterPromise;
  return hl.codeToHtml(code, { lang, theme });
}

function firstFile(nodes: TreeNode[]): string | null {
  for (const n of nodes) {
    if (n.type === 'file') return n.path;
    const found = firstFile(n.children);
    if (found) return found;
  }
  return null;
}

function Tree({
  nodes,
  selected,
  onSelect,
  depth,
}: {
  nodes: TreeNode[];
  selected: string | null;
  onSelect: (p: string) => void;
  depth: number;
}) {
  return (
    <ul>
      {nodes.map((n) =>
        n.type === 'dir' ? (
          <DirItem key={n.path} node={n} selected={selected} onSelect={onSelect} depth={depth} />
        ) : (
          <li key={n.path}>
            <button
              onClick={() => onSelect(n.path)}
              style={{ paddingLeft: depth * 12 + 10 }}
              className={`w-full flex items-center gap-1.5 pr-2 py-1 text-left text-[13px] font-mono transition-colors ${
                selected === n.path
                  ? 'bg-brand/10 text-brand'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              }`}
            >
              <FileCode className="h-3.5 w-3.5 shrink-0 opacity-70" />
              <span className="truncate">{n.name}</span>
            </button>
          </li>
        )
      )}
    </ul>
  );
}

function DirItem({
  node,
  selected,
  onSelect,
  depth,
}: {
  node: Extract<TreeNode, { type: 'dir' }>;
  selected: string | null;
  onSelect: (p: string) => void;
  depth: number;
}) {
  const [open, setOpen] = useState(depth < 1);
  return (
    <li>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{ paddingLeft: depth * 12 + 8 }}
        className="w-full flex items-center gap-1 pr-2 py-1 text-left text-[13px] font-mono text-foreground hover:bg-muted/50"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5 shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0" />}
        <span className="truncate font-medium">{node.name}</span>
      </button>
      {open && <Tree nodes={node.children} selected={selected} onSelect={onSelect} depth={depth + 1} />}
    </li>
  );
}

export function CodeBrowser({ data }: { data: CodeData }) {
  const { resolvedTheme } = useTheme();
  const [selected, setSelected] = useState<string | null>(() => firstFile(data.tree));
  const [html, setHtml] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selected) return;
    const file = data.files[selected];
    if (!file) return;
    let cancelled = false;
    setLoading(true);
    highlight(file.content, file.lang, resolvedTheme === 'dark' ? 'github-dark' : 'github-light')
      .then((h) => {
        if (!cancelled) {
          setHtml(h);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHtml(`<pre>${file.content.replace(/</g, '&lt;')}</pre>`);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selected, resolvedTheme, data.files]);

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      <SiteHeader config={projectConfig} active="code" fluid />

      <div className="flex-1 flex min-h-0">
        {/* File tree */}
        <aside className="w-64 shrink-0 border-r border-border overflow-y-auto py-2">
          <Tree nodes={data.tree} selected={selected} onSelect={setSelected} depth={0} />
        </aside>

        {/* Code pane */}
        <main className="flex-1 min-w-0 flex flex-col">
          <div className="border-b border-border px-4 py-2 shrink-0">
            <span className="font-mono text-xs text-muted-foreground">{selected ?? 'select a file'}</span>
          </div>
          <div className="flex-1 overflow-auto">
            {loading ? (
              <div className="p-4 font-mono text-sm text-muted-foreground animate-pulse">highlighting…</div>
            ) : (
              <div
                className="text-[13px] [&_pre]:p-4 [&_pre]:min-h-full [&_pre]:!bg-transparent [&_code]:font-mono"
                dangerouslySetInnerHTML={{ __html: html }}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
