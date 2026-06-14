/**
 * Query embedding via the HuggingFace Inference API — the SAME model the corpus
 * was built with (sentence-transformers/all-MiniLM-L6-v2), so query vectors land
 * in the same 384-dim space as the stored document vectors. Running it as a
 * hosted call (instead of transformers.js in-function) keeps the Vercel function
 * tiny and avoids bundling onnxruntime's native binaries.
 *
 * Requires HF_TOKEN (free at https://huggingface.co/settings/tokens).
 * Output is L2-normalized to match the normalized document vectors.
 */
const MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const ENDPOINT = `https://router.huggingface.co/hf-inference/models/${MODEL}/pipeline/feature-extraction`;

function l2normalize(v: number[]): number[] {
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  return v.map((x) => x / norm);
}

export async function embedQuery(text: string): Promise<number[]> {
  const token = process.env.HF_TOKEN;
  if (!token) throw new Error("Missing HF_TOKEN");

  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ inputs: text, options: { wait_for_model: true } }),
  });

  if (!res.ok) {
    throw new Error(`HF embedding failed (${res.status}): ${(await res.text()).slice(0, 200)}`);
  }

  const data = (await res.json()) as number[] | number[][];
  // feature-extraction returns a flat 384-vector for a single string; some
  // deployments wrap it as [[...]]. Handle both.
  const vec = Array.isArray(data[0]) ? (data[0] as number[]) : (data as number[]);
  if (vec.length !== 384) {
    throw new Error(`Unexpected embedding length ${vec.length}`);
  }
  return l2normalize(vec);
}
