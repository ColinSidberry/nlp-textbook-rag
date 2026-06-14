/**
 * Query expansion — port of `expand_query` from the original rag_pipeline.py.
 * Expands the first NLP acronym it finds and augments comparison questions,
 * so retrieval matches the same way it did in the Streamlit/Chroma version.
 */

const ACRONYM_EXPANSIONS: Record<string, string> = {
  RAG: "retrieval-augmented generation retrieval augmented generation",
  NER: "named entity recognition",
  POS: "part-of-speech part of speech tagging",
  LLM: "large language model",
  LSTM: "long short-term memory",
  RNN: "recurrent neural network",
  CNN: "convolutional neural network",
  GRU: "gated recurrent unit",
  NLP: "natural language processing",
  CRF: "conditional random field",
  HMM: "hidden markov model",
  "TF-IDF": "term frequency inverse document frequency",
  BERT: "bidirectional encoder representations from transformers",
  GPT: "generative pre-trained transformer",
  IR: "information retrieval",
};

const COMPARISON_PATTERNS: RegExp[] = [
  /difference between (\w+) and (\w+)/,
  /compare (\w+) (?:and|vs|versus) (\w+)/,
  /(\w+) vs (\w+)/,
  /(\w+) versus (\w+)/,
];

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function expandQuery(query: string): string {
  // 1. Expand the first matching acronym (augment, don't replace).
  for (const [acronym, expansion] of Object.entries(ACRONYM_EXPANSIONS)) {
    const pattern = new RegExp(`\\b${escapeRegExp(acronym)}\\b`, "i");
    if (pattern.test(query)) {
      return `${query} ${expansion}`;
    }
  }

  // 2. Comparison questions: surface both terms.
  const lower = query.toLowerCase();
  for (const pattern of COMPARISON_PATTERNS) {
    const match = lower.match(pattern);
    if (match) {
      const [, term1, term2] = match;
      return `${query} ${term1} ${term2} comparison`;
    }
  }

  return query;
}
