import { NextResponse } from "next/server";
import { embedQuery } from "@/lib/embedder";
import { expandQuery } from "@/lib/expand";
import { matchChunks } from "@/lib/supabase";
import { resolveProvider, generateAnswer } from "@/lib/llm";
import {
  TOP_K,
  SYSTEM_PROMPT,
  isOutOfScope,
  OUT_OF_SCOPE_ANSWER,
  formatContext,
  extractCitations,
  type QueryResult,
} from "@/lib/rag";

export const runtime = "nodejs";
// Generous ceiling: the first request in a cold instance loads the MiniLM model.
export const maxDuration = 60;

export async function POST(request: Request) {
  let question = "";
  let provider: string | undefined;
  let apiKey: string | undefined;
  let model: string | undefined;
  try {
    const body = await request.json();
    question = typeof body?.question === "string" ? body.question.trim() : "";
    provider = typeof body?.provider === "string" ? body.provider : undefined;
    apiKey = typeof body?.apiKey === "string" ? body.apiKey : undefined;
    model = typeof body?.model === "string" && body.model ? body.model : undefined;
  } catch {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }

  if (question.length < 3) {
    return NextResponse.json(
      { error: "Please enter a longer question (at least 3 characters)." },
      { status: 400 }
    );
  }

  const llm = resolveProvider(provider, apiKey);
  if (!llm) {
    return NextResponse.json(
      { error: "No LLM available. Paste your own API key, or set a default server key." },
      { status: 400 }
    );
  }

  let chunks;
  try {
    const embedding = await embedQuery(expandQuery(question));
    chunks = await matchChunks(embedding, TOP_K);
  } catch (err) {
    console.error("Retrieval error:", err);
    return NextResponse.json({ error: "Retrieval failed." }, { status: 500 });
  }

  if (isOutOfScope(chunks)) {
    const result: QueryResult = { answer: OUT_OF_SCOPE_ANSWER, citations: [], chunks: [] };
    return NextResponse.json(result);
  }

  const userContent = `Context from textbook:\n${formatContext(chunks)}\n\nQuestion: ${question}`;

  let answer: string;
  try {
    answer = await generateAnswer({
      provider: llm.provider,
      apiKey: llm.apiKey,
      model,
      system: SYSTEM_PROMPT,
      userContent,
    });
  } catch (err) {
    console.error("Generation error:", err);
    const message =
      apiKey && err instanceof Error
        ? `Generation failed: ${err.message}`
        : "Generation failed. Check the API key or try again.";
    return NextResponse.json({ error: message }, { status: 502 });
  }

  const result: QueryResult = {
    answer,
    citations: extractCitations(chunks),
    chunks,
  };
  return NextResponse.json(result);
}
