"use client";

import { useState } from "react";
import { SiteHeader } from "@/components/project/SiteHeader";
import { projectConfig } from "@/components/project/config";

type Chunk = {
  id: string;
  document: string;
  chapter: string;
  filename: string;
  chunk_index: number;
  similarity: number;
};

type QueryResult = {
  answer: string;
  citations: string[];
  chunks: Chunk[];
};

type ProviderChoice = "default" | "anthropic" | "openai" | "groq";

const SAMPLES = [
  "How do transformers use attention mechanisms?",
  "What are word embeddings and how are they created?",
  "How do n-gram language models work?",
  "Explain logistic regression for text classification",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const [showByok, setShowByok] = useState(false);
  const [provider, setProvider] = useState<ProviderChoice>("default");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");

  async function ask(q: string) {
    if (q.trim().length < 3) {
      setError("Please enter a longer question (at least 3 characters).");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(null);
    const started = performance.now();

    const payload: Record<string, string> = { question: q.trim() };
    if (provider !== "default" && apiKey.trim()) {
      payload.provider = provider;
      payload.apiKey = apiKey.trim();
      if (model.trim()) payload.model = model.trim();
    }

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.error || "Something went wrong.");
      } else {
        setResult(data);
        setElapsed((performance.now() - started) / 1000);
      }
    } catch {
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
    <SiteHeader config={projectConfig} active="live" fluid />
    <main className="mx-auto w-full max-w-3xl px-5 py-12 sm:py-16">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          NLP Textbook RAG
        </h1>
        <p className="mt-2 text-sm text-stone-600 dark:text-stone-400">
          Ask about NLP concepts from Jurafsky &amp; Martin&apos;s{" "}
          <em>Speech and Language Processing</em>. Answers are grounded in 10,170
          indexed chunks and cited — nothing is made up.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
        className="space-y-3"
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              ask(question);
            }
          }}
          rows={3}
          placeholder="e.g. How do transformers use attention mechanisms?"
          className="w-full resize-y rounded-xl border border-stone-300 bg-white px-4 py-3 text-sm shadow-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 dark:border-stone-700 dark:bg-stone-900"
        />

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-500 disabled:opacity-60"
          >
            {loading ? "Searching…" : "Ask"}
          </button>
          <span className="text-xs text-stone-500">⌘/Ctrl + Enter</span>
          <button
            type="button"
            onClick={() => setShowByok((s) => !s)}
            className="ml-auto text-xs font-medium text-stone-600 hover:underline dark:text-stone-400"
          >
            {showByok ? "Hide API key options" : "Use your own API key"}
          </button>
        </div>
      </form>

      {showByok && (
        <div className="mt-3 space-y-3 rounded-xl border border-stone-200 bg-stone-50 p-4 text-sm dark:border-stone-800 dark:bg-stone-900/50">
          <p className="text-xs text-stone-500">
            Default uses a free Groq model. Bring your own key for Anthropic or
            OpenAI — it&apos;s sent only with this request and never stored.
          </p>
          <div className="flex flex-wrap gap-3">
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value as ProviderChoice)}
              className="rounded-lg border border-stone-300 bg-white px-3 py-2 dark:border-stone-700 dark:bg-stone-900"
            >
              <option value="default">Default (free Groq)</option>
              <option value="anthropic">Anthropic</option>
              <option value="openai">OpenAI</option>
              <option value="groq">Groq (own key)</option>
            </select>
            {provider !== "default" && (
              <>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="API key"
                  className="min-w-[14rem] flex-1 rounded-lg border border-stone-300 bg-white px-3 py-2 dark:border-stone-700 dark:bg-stone-900"
                />
                <input
                  type="text"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="model (optional)"
                  className="w-40 rounded-lg border border-stone-300 bg-white px-3 py-2 dark:border-stone-700 dark:bg-stone-900"
                />
              </>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="mt-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      )}

      {!result && !error && !loading && (
        <div className="mt-8">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-stone-500">
            Try
          </p>
          <div className="flex flex-wrap gap-2">
            {SAMPLES.map((s) => (
              <button
                key={s}
                onClick={() => {
                  setQuestion(s);
                  ask(s);
                }}
                className="rounded-full border border-stone-300 px-3 py-1.5 text-xs text-stone-700 transition hover:border-indigo-400 hover:text-indigo-600 dark:border-stone-700 dark:text-stone-300"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {result && (
        <section className="mt-8 space-y-6">
          <div>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
              Answer
            </h2>
            <div className="whitespace-pre-wrap rounded-xl border border-stone-200 bg-white p-4 text-[15px] leading-relaxed shadow-sm dark:border-stone-800 dark:bg-stone-900">
              {result.answer}
            </div>
            {elapsed !== null && (
              <p className="mt-2 text-xs text-stone-500">
                {elapsed.toFixed(1)}s · {result.chunks.length} chunks ·{" "}
                {result.citations.length} sources
              </p>
            )}
          </div>

          {result.citations.length > 0 && (
            <div>
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
                Citations
              </h2>
              <ul className="flex flex-wrap gap-2">
                {result.citations.map((c) => (
                  <li
                    key={c}
                    className="rounded-lg border-l-2 border-indigo-500 bg-stone-100 px-3 py-1.5 font-mono text-xs dark:bg-stone-800"
                  >
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.chunks.length > 0 && (
            <details className="group rounded-xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
              <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-stone-700 dark:text-stone-300">
                Retrieved chunks ({result.chunks.length})
                <span className="ml-2 text-xs text-stone-400 group-open:hidden">
                  show
                </span>
              </summary>
              <div className="space-y-4 border-t border-stone-200 px-4 py-4 dark:border-stone-800">
                {result.chunks.map((c, i) => (
                  <div key={c.id}>
                    <div className="mb-1 flex items-center justify-between text-xs text-stone-500">
                      <span className="font-mono">
                        #{i + 1} · {c.filename}
                      </span>
                      <span>sim {c.similarity.toFixed(3)}</span>
                    </div>
                    <p className="rounded-lg bg-stone-50 p-3 font-mono text-xs leading-relaxed text-stone-700 dark:bg-stone-950 dark:text-stone-300">
                      {c.document}
                    </p>
                  </div>
                ))}
              </div>
            </details>
          )}
        </section>
      )}
    </main>
    </>
  );
}
