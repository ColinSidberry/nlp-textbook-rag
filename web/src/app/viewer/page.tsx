"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";

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
  const [filename, setFilename] = useState("");
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState(""); // debounced/applied search
  const [feed, setFeed] = useState<Feed | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams({ page: String(page) });
    if (filename) params.set("filename", filename);
    if (query) params.set("q", query);
    try {
      const res = await fetch(`/api/chunks?${params.toString()}`);
      const data = await res.json();
      if (res.ok) setFeed(data);
    } finally {
      setLoading(false);
    }
  }, [page, filename, query]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <main className="mx-auto w-full max-w-4xl px-5 py-12">
      <header className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Database Viewer</h1>
          <p className="mt-1 text-sm text-stone-500">
            {feed ? feed.total.toLocaleString() : "—"} chunks · MiniLM 384-dim ·
            cosine
          </p>
        </div>
        <Link
          href="/"
          className="text-sm font-medium text-indigo-600 hover:underline dark:text-indigo-400"
        >
          ← Back to search
        </Link>
      </header>

      <div className="mb-5 flex flex-wrap gap-3">
        <select
          value={filename}
          onChange={(e) => {
            setFilename(e.target.value);
            setPage(1);
          }}
          className="rounded-lg border border-stone-300 bg-white px-3 py-2 text-sm dark:border-stone-700 dark:bg-stone-900"
        >
          <option value="">All chapters</option>
          {feed?.filenames.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            setQuery(search.trim());
            setPage(1);
          }}
          className="flex flex-1 gap-2"
        >
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search within chunk text…"
            className="min-w-[12rem] flex-1 rounded-lg border border-stone-300 bg-white px-3 py-2 text-sm dark:border-stone-700 dark:bg-stone-900"
          />
          <button
            type="submit"
            className="rounded-lg border border-stone-300 px-3 py-2 text-sm dark:border-stone-700"
          >
            Search
          </button>
        </form>
      </div>

      {loading && <p className="text-sm text-stone-500">Loading…</p>}

      {feed && (
        <>
          <div className="space-y-3">
            {feed.entries.map((e) => (
              <div
                key={e.id}
                className="rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900"
              >
                <div className="mb-2 flex items-center justify-between text-xs text-stone-500">
                  <span className="font-mono">
                    {e.filename} · #{e.chunk_index}
                  </span>
                  <span className="font-mono">{e.id}</span>
                </div>
                <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-stone-700 dark:text-stone-300">
                  {e.document}
                </p>
              </div>
            ))}
            {feed.entries.length === 0 && (
              <p className="text-sm text-stone-500">No matching chunks.</p>
            )}
          </div>

          <div className="mt-6 flex items-center justify-between text-sm">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="rounded-lg border border-stone-300 px-3 py-1.5 disabled:opacity-40 dark:border-stone-700"
            >
              ← Prev
            </button>
            <span className="text-stone-500">
              Page {feed.page} of {feed.totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(feed.totalPages, p + 1))}
              disabled={page >= feed.totalPages}
              className="rounded-lg border border-stone-300 px-3 py-1.5 disabled:opacity-40 dark:border-stone-700"
            >
              Next →
            </button>
          </div>
        </>
      )}
    </main>
  );
}
