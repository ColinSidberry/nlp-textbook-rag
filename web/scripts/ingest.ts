/**
 * Phase 1 ingestion — apply the schema and load the exported Chroma vectors into
 * Supabase pgvector over a direct Postgres connection (DATABASE_URL).
 *
 * Reads data/export/chunks.jsonl (from scripts/dump_chroma.py) and upserts every
 * chunk into nlp_chunks. No re-embedding: the 384-dim MiniLM vectors are reused
 * verbatim, so retrieval is identical to the original app.
 *
 * Usage (from web/):  npm run ingest
 * Requires .env.local:  DATABASE_URL
 */
import { createInterface } from "node:readline";
import { createReadStream, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { config } from "dotenv";
import { Client } from "pg";

config({ path: ".env.local" });

const DATABASE_URL = process.env.DATABASE_URL;
const JSONL_PATH = resolve(process.cwd(), "..", "data", "export", "chunks.jsonl");
const SCHEMA_PATH = resolve(process.cwd(), "supabase", "schema.sql");
const EXPECTED_ROWS = 10170;
const BATCH_SIZE = 500;

if (!DATABASE_URL) {
  console.error("Missing DATABASE_URL in .env.local");
  process.exit(1);
}

type Row = {
  id: string;
  document: string;
  embedding: number[];
  chapter: string;
  filename: string;
  chunk_index: number;
  document_id: string;
};

async function main(): Promise<void> {
  const client = new Client({ connectionString: DATABASE_URL });
  await client.connect();

  console.log("Applying schema…");
  await client.query(readFileSync(SCHEMA_PATH, "utf-8"));

  console.log("Ingesting chunks…");
  const rl = createInterface({
    input: createReadStream(JSONL_PATH, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });

  let batch: Row[] = [];
  let total = 0;
  const dims = new Set<number>();

  const flush = async () => {
    if (batch.length === 0) return;
    const cols = 7;
    const values: unknown[] = [];
    const tuples = batch.map((r, i) => {
      const b = i * cols;
      values.push(
        r.id,
        r.document,
        r.chapter,
        r.filename,
        r.chunk_index,
        r.document_id,
        `[${r.embedding.join(",")}]`
      );
      return `($${b + 1},$${b + 2},$${b + 3},$${b + 4},$${b + 5},$${b + 6},$${b + 7}::vector)`;
    });
    await client.query(
      `insert into nlp_chunks (id, document, chapter, filename, chunk_index, document_id, embedding)
       values ${tuples.join(",")}
       on conflict (id) do update set
         document = excluded.document,
         chapter = excluded.chapter,
         filename = excluded.filename,
         chunk_index = excluded.chunk_index,
         document_id = excluded.document_id,
         embedding = excluded.embedding`,
      values
    );
    total += batch.length;
    process.stdout.write(`\r  upserted ${total}/${EXPECTED_ROWS}`);
    batch = [];
  };

  for await (const line of rl) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as Row;
    dims.add(row.embedding.length);
    batch.push(row);
    if (batch.length >= BATCH_SIZE) await flush();
  }
  await flush();

  console.log(`\nDone. Upserted ${total} rows. Embedding dims seen: ${[...dims]}`);

  const { rows } = await client.query<{ count: string }>("select count(*) from nlp_chunks");
  const count = Number(rows[0].count);
  console.log(`Table row count: ${count}`);
  await client.end();

  if (count !== EXPECTED_ROWS) {
    console.warn(`WARNING: expected ${EXPECTED_ROWS} rows, found ${count}`);
    process.exit(1);
  }
  console.log("Row count matches expected 10,170 ✓");
}

main().catch((err) => {
  console.error("\nIngestion failed:", err);
  process.exit(1);
});
