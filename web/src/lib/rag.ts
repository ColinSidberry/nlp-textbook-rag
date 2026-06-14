/**
 * RAG core — ported from rag_pipeline.py: the adaptive relevance threshold,
 * context formatting, citation extraction, and the 7-point anti-hallucination
 * prompt (kept verbatim).
 */
import type { MatchedChunk } from "./supabase";

export const TOP_K = 5;

/** The 7-point anti-hallucination system prompt, preserved from the original. */
export const SYSTEM_PROMPT = `You are an AI assistant helping students understand concepts from the NLP textbook "Speech and Language Processing" by Jurafsky & Martin.

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. SCOPE: ONLY answer questions about NLP/linguistics concepts (transformers, embeddings, parsing, language models, etc.)
2. GROUNDING: ONLY use information explicitly stated in the context provided - NEVER use external knowledge
3. OUT-OF-SCOPE QUESTIONS: If the question asks about current events, general knowledge, cooking, geography, politics, or ANY non-NLP topic, respond EXACTLY with: "This question is outside the scope of the NLP textbook. Please ask about natural language processing concepts like transformers, embeddings, language models, parsing, etc."
4. INSUFFICIENT CONTEXT: If the context lacks information for a valid NLP question, say: "I don't have enough information in the retrieved context to answer this question. Try rephrasing or asking about a different aspect."
5. NO FABRICATION: Do NOT make up facts, examples, names, dates, or connections not in the context
6. NO EXTERNAL KNOWLEDGE: Even if you know the answer from general knowledge, ONLY use the context
7. CITATION ACCURACY: Only cite chapters/sources that appear in the context excerpts provided

Respond with only the final answer, using ONLY the context, and decline if out-of-scope.`;

/**
 * Adaptive relevance gate (verbatim thresholds from the original): treat the
 * question as out-of-scope when avg similarity < 0.35, or when avg < 0.4 and
 * max < 0.42. Handles typos while blocking truly off-topic questions.
 */
export function isOutOfScope(chunks: MatchedChunk[]): boolean {
  if (chunks.length === 0) return true;
  const sims = chunks.map((c) => c.similarity);
  const avg = sims.reduce((a, b) => a + b, 0) / sims.length;
  const max = Math.max(...sims);
  return avg < 0.35 || (avg < 0.4 && max < 0.42);
}

export const OUT_OF_SCOPE_ANSWER =
  "This question is outside the scope of the NLP textbook. Please ask about natural language processing concepts like transformers, embeddings, language models, parsing, sentiment analysis, etc.";

/** Build the LLM context block from retrieved chunks. */
export function formatContext(chunks: MatchedChunk[]): string {
  return chunks
    .map(
      (c, i) =>
        `[Excerpt ${i + 1} - Source: ${c.filename}, Relevance: ${c.similarity.toFixed(2)}]\n${c.document}`
    )
    .join("\n\n");
}

/** Deduplicated source citations (filename is the clean key; chapter metadata is noisy). */
export function extractCitations(chunks: MatchedChunk[]): string[] {
  const seen = new Set<string>();
  const citations: string[] = [];
  for (const c of chunks) {
    if (!seen.has(c.filename)) {
      seen.add(c.filename);
      citations.push(c.filename);
    }
  }
  return citations;
}

export type QueryResult = {
  answer: string;
  citations: string[];
  chunks: MatchedChunk[];
};
