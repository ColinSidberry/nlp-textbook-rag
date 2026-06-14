import { NextResponse } from "next/server";
import { getSupabase } from "@/lib/supabase";

export const runtime = "nodejs";

const PAGE_SIZE = 50;

/** Read-only Database Viewer feed. The Supabase key stays server-side. */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const page = Math.max(1, parseInt(searchParams.get("page") || "1", 10) || 1);
  const filename = searchParams.get("filename") || "";
  const search = (searchParams.get("q") || "").trim();

  const supabase = getSupabase();

  let query = supabase
    .from("nlp_chunks")
    .select("id, document, chapter, filename, chunk_index", { count: "exact" });

  if (filename) query = query.eq("filename", filename);
  if (search) query = query.ilike("document", `%${search}%`);

  const from = (page - 1) * PAGE_SIZE;
  const to = from + PAGE_SIZE - 1;
  query = query.order("filename").order("chunk_index").range(from, to);

  const { data, count, error } = await query;
  if (error) {
    console.error("Viewer query error:", error);
    return NextResponse.json({ error: "Failed to load entries." }, { status: 500 });
  }

  const total = count ?? 0;
  const { data: fnData } = await supabase.rpc("nlp_filenames");
  const filenames = ((fnData ?? []) as { filename: string }[]).map((r) => r.filename);

  return NextResponse.json({
    entries: data ?? [],
    total,
    page,
    pageSize: PAGE_SIZE,
    totalPages: Math.max(1, Math.ceil(total / PAGE_SIZE)),
    filenames,
  });
}
