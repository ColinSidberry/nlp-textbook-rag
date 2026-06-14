/**
 * LLM generation with bring-your-own-key. Default is Groq Llama 3.3 70B (free
 * tier, server-side key); a visitor can override with their own Anthropic or
 * OpenAI key. BYOK keys are used only to make the request and are never stored.
 */
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export type Provider = "groq" | "openai" | "anthropic";

export const DEFAULT_MODELS: Record<Provider, string> = {
  groq: "llama-3.3-70b-versatile",
  openai: "gpt-4o-mini",
  anthropic: "claude-opus-4-8",
};

const GROQ_BASE_URL = "https://api.groq.com/openai/v1";

export type GenerateOptions = {
  provider: Provider;
  apiKey: string;
  model?: string;
  system: string;
  userContent: string;
  maxTokens?: number;
};

/**
 * Resolve which provider + key to use. A visitor-supplied key wins (BYOK);
 * otherwise fall back to the server's Groq key. Returns null if no key is
 * available at all (e.g. BYOK declined and no server default configured).
 */
export function resolveProvider(byokProvider?: string, byokKey?: string):
  | { provider: Provider; apiKey: string }
  | null {
  if (byokKey && byokProvider && ["groq", "openai", "anthropic"].includes(byokProvider)) {
    return { provider: byokProvider as Provider, apiKey: byokKey };
  }
  const serverGroq = process.env.GROQ_API_KEY;
  if (serverGroq) return { provider: "groq", apiKey: serverGroq };
  return null;
}

export async function generateAnswer(opts: GenerateOptions): Promise<string> {
  const { provider, apiKey, system, userContent } = opts;
  const model = opts.model || DEFAULT_MODELS[provider];
  const maxTokens = opts.maxTokens ?? 1024;

  if (provider === "anthropic") {
    const client = new Anthropic({ apiKey });
    // Opus 4.8: no temperature / no thinking config (both 400). The system
    // prompt already forces a final-answer-only, grounded response.
    const message = await client.messages.create({
      model,
      max_tokens: maxTokens,
      system,
      messages: [{ role: "user", content: userContent }],
    });
    return message.content
      .filter((b): b is Anthropic.TextBlock => b.type === "text")
      .map((b) => b.text)
      .join("")
      .trim();
  }

  // Groq + OpenAI share the OpenAI-compatible chat completions API.
  const client = new OpenAI({
    apiKey,
    baseURL: provider === "groq" ? GROQ_BASE_URL : undefined,
  });
  const completion = await client.chat.completions.create({
    model,
    max_tokens: maxTokens,
    temperature: 0.7,
    messages: [
      { role: "system", content: system },
      { role: "user", content: userContent },
    ],
  });
  return (completion.choices[0]?.message?.content ?? "").trim();
}
