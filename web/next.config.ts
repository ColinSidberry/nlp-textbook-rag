import type { NextConfig } from "next";
import path from "node:path";

const root = path.resolve(import.meta.dirname);

const nextConfig: NextConfig = {
  // Pin the workspace root to web/ so file tracing resolves correctly (a stray
  // parent lockfile otherwise misleads inference).
  turbopack: { root },
  outputFileTracingRoot: root,
};

export default nextConfig;
