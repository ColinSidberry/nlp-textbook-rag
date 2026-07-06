"use client";

import { useCallback, useEffect, useState } from "react";
import { SiteHeader } from "@/components/project/SiteHeader";
import { projectConfig } from "@/components/project/config";

type Entry = {
  id: string;
  document: string;
  chapter: string;
  filename: string;
  chunk_index: number;
};

type Feed = {
  entries: Entry[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  filenames: string[];
};

export default function ViewerPage() {
  const [page, setPage] = useState(1);
  const [filename, setFilename] = useState(""); // "" = all chapters
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState(""); // applied search
  const [feed, setFeed] = useState<Feed | null>(null);
  const [filenames, setFilenames] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams({ page: String(page) });
    if (filename) params.set("filename", filename);
    if (query) params.set("q", query);
    try {
      const res = await fetch(`/api/chunks?${params.toString()}`);
      const data = await res.json();
      if (res.ok) {
        setFeed(data);
        if (data.filenames?.length) setFilenames(data.filenames);
      }
    } finally {
      setLoading(false);
    }
  }, [page, filename, query]);

  useEffect(() => {
    load();
  }, [load]);

  const chapters = [{ value: "", label: "All chapters" }, ...filenames.map((f) => ({ value: f, label: f }))];

  return (
    // Single vector-chunk store (not multiple collections) — presented in the same
    // two-pane shell as the sibling database views: chapters pick on the left,
    // that chapter's chunks on the right.
    <div className="gh min-h-screen bg-background text-foreground">
      <SiteHeader config={projectConfig} active="database" fluid />

      <div className="flex flex-col md:flex-row">
        {/* Left: chapters (the "collections") */}
        <aside className="md:w-72 md:shrink-0 border-b md:border-b-0 md:border-r border-border md:h-[calc(100vh-57px)] md:sticky md:top-14 overflow-y-auto">
          <div className="px-4 pt-4 pb-2 text-[11px] font-mono uppercase tracking-widest text-muted-foreground">
            Chapters
          </div>
          <nav className="pb-4">
            {chapters.map(({ value, label }) => {
              const isActive = filename === value;
              return (
                <button
                  key={value || "__all"}
                  onClick={() => {
                    setFilename(value);
                    setPage(1);
                  }}
                  className={`w-full text-left px-4 py-2 flex items-center gap-2 border-l-2 transition-colors ${
                    isActive ? "border-brand bg-muted/60" : "border-transparent hover:bg-muted/40"
                  }`}
                >
                  <span className="font-mono text-sm truncate">{label}</span>
                </button>
              );
            })}
            {filenames.length === 0 && !loading && (
              <div className="px-4 py-2 text-sm text-muted-foreground font-mono animate-pulse">connecting…</div>
            )}
          </nav>
        </aside>

        {/* Right: chunks (the "documents") */}
        <main className="flex-1 min-w-0">
          <div className="border-b border-border px-5 py-3 flex flex-wrap items-center gap-x-3 gap-y-2">
            <h2 className="font-mono text-base font-semibold">{filename || "All chapters"}</h2>
            <span className="text-xs text-muted-foreground font-mono">
              {feed ? feed.total.toLocaleString() : "—"} chunks · MiniLM 384-dim · cosine
            </span>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                setQuery(search.trim());
                setPage(1);
              }}
              className="ml-auto flex gap-2"
            >
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search chunk text…"
                className="min-w-[11rem] rounded-md border border-border bg-card px-3 py-1.5 text-sm outline-none focus:border-brand"
              />
              <button
                type="submit"
                className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted transition-colors"
              >
                Search
              </button>
            </form>
          </div>

          <div className="p-5">
            {loading && <p className="text-sm text-muted-foreground font-mono animate-pulse">loading…</p>}

            {feed && (
              <>
                <div className="space-y-3">
                  {feed.entries.map((e) => (
                    <div key={e.id} className="rounded-lg border border-border bg-card overflow-hidden">
                      <div className="px-3 py-1.5 border-b border-border/60 bg-muted/30 flex items-center justify-between gap-2">
                        <span className="font-mono text-[11px] text-muted-foreground truncate">
                          {e.filename} · #{e.chunk_index}
                        </span>
                        <span className="font-mono text-[11px] text-muted-foreground truncate">{e.id}</span>
                      </div>
                      <div className="p-3 overflow-x-auto">
                        <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground/90">
                          {e.document}
                        </p>
                      </div>
                    </div>
                  ))}
                  {feed.entries.length === 0 && (
                    <p className="text-sm text-muted-foreground font-mono">no matching chunks.</p>
                  )}
                </div>

                {feed.totalPages > 1 && (
                  <div className="mt-6 flex items-center justify-between text-sm">
                    <button
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page <= 1}
                      className="rounded-md border border-border px-3 py-1.5 disabled:opacity-40 hover:bg-muted transition-colors"
                    >
                      ← Prev
                    </button>
                    <span className="text-muted-foreground font-mono">
                      Page {feed.page} of {feed.totalPages}
                    </span>
                    <button
                      onClick={() => setPage((p) => Math.min(feed.totalPages, p + 1))}
                      disabled={page >= feed.totalPages}
                      className="rounded-md border border-border px-3 py-1.5 disabled:opacity-40 hover:bg-muted transition-colors"
                    >
                      Next →
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
