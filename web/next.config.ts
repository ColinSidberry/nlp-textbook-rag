import type { NextConfig } from "next";
import path from "node:path";

const root = path.resolve(import.meta.dirname);

const nextConfig: NextConfig = {
  // transformers.js + onnxruntime-node must use native require, not be bundled.
  // (Next 16 auto-externalizes these, but make it explicit.)
  serverExternalPackages: ["@huggingface/transformers", "onnxruntime-node"],
  // Pin the workspace root to web/ so file tracing bundles onnxruntime's native
  // binaries correctly (a stray parent lockfile otherwise misleads inference).
  turbopack: { root },
  outputFileTracingRoot: root,
};

export default nextConfig;
