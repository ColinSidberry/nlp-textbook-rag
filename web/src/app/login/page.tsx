"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { SiteHeader } from "@/components/project/SiteHeader";
import { projectConfig } from "@/components/project/config";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const from = params.get("from") || "/";

  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        router.replace(from);
        router.refresh();
      } else {
        const data = await res.json().catch(() => ({}));
        setError(data?.error || "Incorrect password.");
      }
    } catch {
      setError("Network error.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form
      onSubmit={submit}
      className="w-full max-w-sm space-y-4 rounded-2xl border border-stone-200 bg-white p-6 shadow-sm dark:border-stone-800 dark:bg-stone-900"
    >
      <div>
        <h1 className="text-lg font-semibold">NLP Textbook RAG</h1>
        <p className="mt-1 text-sm text-stone-500">Enter the password to continue.</p>
      </div>
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        autoFocus
        className="w-full rounded-lg border border-stone-300 bg-white px-3 py-2 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 dark:border-stone-700 dark:bg-stone-950"
      />
      {error && <p className="text-sm text-red-600">{error}</p>}
      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:opacity-60"
      >
        {loading ? "Checking…" : "Enter"}
      </button>
    </form>
  );
}

export default function LoginPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader config={projectConfig} fluid />
      <div className="flex flex-1 items-center justify-center px-5 py-16">
        <Suspense fallback={null}>
          <LoginForm />
        </Suspense>
      </div>
    </div>
  );
}
