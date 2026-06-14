/**
 * Query embedding via transformers.js — the EXACT MiniLM model the corpus was
 * embedded with (Xenova/all-MiniLM-L6-v2, 384-dim, mean-pooled + L2-normalized).
 *
 * No embedding API key anywhere: the model runs in the Vercel Node function.
 * `dtype: "fp32"` matches the fp32 doc vectors exported from Chroma, so query
 * and document vectors live in the same space. The first call pays a cold-start
 * cost (model download to /tmp + load); the module-level singleton keeps the
 * extractor warm for subsequent calls in the same instance.
 */
import { pipeline, env, type FeatureExtractionPipeline } from "@huggingface/transformers";

// Vercel's bundle is read-only except /tmp; always fetch the model remotely and
// cache it there rather than trying to write into the deployment directory.
env.allowLocalModels = false;
env.cacheDir = "/tmp/hf-cache";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_ID, { dtype: "fp32" });
  }
  return extractorPromise;
}

/** Embed a query string into a 384-dim L2-normalized vector. */
export async function embedQuery(text: string): Promise<number[]> {
  const extractor = await getExtractor();
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}
