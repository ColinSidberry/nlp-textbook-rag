/**
 * Server-only Supabase client. Uses the service-role key, which must NEVER be
 * exposed to the browser — only import this from API routes / server code.
 */
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

let client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (!client) {
    const url = process.env.SUPABASE_URL;
    const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
    if (!url || !key) {
      throw new Error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
    }
    client = createClient(url, key, { auth: { persistSession: false } });
  }
  return client;
}

export type MatchedChunk = {
  id: string;
  document: string;
  chapter: string;
  filename: string;
  chunk_index: number;
  similarity: number;
};

/** Top-k cosine similarity search via the match_nlp_chunks RPC. */
export async function matchChunks(
  embedding: number[],
  matchCount: number
): Promise<MatchedChunk[]> {
  const { data, error } = await getSupabase().rpc("match_nlp_chunks", {
    query_embedding: embedding,
    match_count: matchCount,
  });
  if (error) throw new Error(`pgvector search failed: ${error.message}`);
  return (data ?? []) as MatchedChunk[];
}
