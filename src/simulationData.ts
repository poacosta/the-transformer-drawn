export type StepId =
  | 'input'
  | 'embedding'
  | 'positional'
  | 'qkv'
  | 'dot-product'
  | 'softmax'
  | 'value-agg'
  | 'ffn'
  | 'unembedding'
  | 'output';

export type LessonStep = {
  id: StepId;
  title: string;
  shortTitle: string;
  description: string;
  readoutTitle: string;
  note: string;
};

export type VectorSet = {
  query: number[];
  key: number[];
  value: number[];
};

export type TokenDatum = {
  text: string;
  embedding: number[];
  positional: number[];
  combined: number[];
  qkv: VectorSet;
};

export type AttentionDatum = {
  token: string;
  tokenIndex: number;
  rawScore: number;
  weight: number;
};

export type AttentionHeadDatum = {
  id: string;
  label: string;
  color: string;
  description: string;
  attention: AttentionDatum[];
};

export type VocabularyDatum = {
  token: string;
  logit: number;
  probability: number;
};

export type Simulation = {
  sentence: string;
  tokens: string[];
  tokenData: TokenDatum[];
  vocabLogits: { token: string; logit: number }[];
  nextToken: string;
};

export const STEPS: LessonStep[] = [
  {
    id: 'input',
    title: 'Input Tokens',
    shortTitle: 'Tokens',
    description: 'The sentence is split into reusable pieces. Pick any token to see what it can attend to while predicting the next token.',
    readoutTitle: 'Sentence pieces',
    note: 'toy run · d_model = 8 · causal',
  },
  {
    id: 'embedding',
    title: 'Token Embeddings',
    shortTitle: 'Embed',
    description: 'Each token becomes a compact vector. Amber cells are positive features; blue cells are negative features.',
    readoutTitle: 'Active embedding',
    note: 'x = E[token] ∈ ℝ⁸',
  },
  {
    id: 'positional',
    title: 'Positional Encoding',
    shortTitle: 'Position',
    description: 'A small wave pattern is added to each vector so equal words in different places can still feel different to the model.',
    readoutTitle: 'Position offset',
    note: 'x ← x + 0.35 · PE(pos)',
  },
  {
    id: 'qkv',
    title: 'Self-Attention: Q, K, V',
    shortTitle: 'Q K V',
    description: 'Every token splits into Query, Key, and Value lanes: what I seek, what I match against, and what information I can pass forward.',
    readoutTitle: 'Active Q/K/V sample',
    note: 'Q = xW_Q · K = xW_K · V = xW_V',
  },
  {
    id: 'dot-product',
    title: 'Attention Scores',
    shortTitle: 'Scores',
    description: 'Each attention head compares the active Query with allowed Keys. The heads learn different relevance patterns.',
    readoutTitle: 'Raw scores',
    note: 'score = Q·K⊤ / √d',
  },
  {
    id: 'softmax',
    title: 'Softmax Weights',
    shortTitle: 'Softmax',
    description: 'Each head runs its own softmax. The percentages sum to 100% inside each head, not across all heads.',
    readoutTitle: 'Attention weights',
    note: 'w = eˢ / Σ eˢ · per head',
  },
  {
    id: 'value-agg',
    title: 'Value Aggregation',
    shortTitle: 'Values',
    description: 'Every head makes its own weighted Value blend. Those head outputs are then merged into one context vector.',
    readoutTitle: 'Weighted value sum',
    note: 'ctx = concat_h(Σ w·V) · W_O',
  },
  {
    id: 'ffn',
    title: 'Feed-Forward Network',
    shortTitle: 'FFN',
    description: 'The context vector expands, features interact in a hidden layer, and the result contracts back to the model width.',
    readoutTitle: 'Hidden layer pulse',
    note: 'h = σ(xW₁)W₂ · 8 → 16 → 8',
  },
  {
    id: 'unembedding',
    title: 'Vocabulary Projection',
    shortTitle: 'Vocab',
    description: 'The final vector is compared with vocabulary rows, producing logits for possible next tokens.',
    readoutTitle: 'Candidate logits',
    note: 'logits = h · E⊤',
  },
  {
    id: 'output',
    title: 'Final Softmax',
    shortTitle: 'Output',
    description: 'The logits become next-token probabilities. Drag the temperature to reshape the distribution before sampling.',
    readoutTitle: 'Next token probabilities',
    note: 'p = softmax(logits / T)',
  },
];

export const DEFAULT_SENTENCE = 'The cat sat on the';
export const MAX_TOKENS = 6;
export const DIMENSIONS = 8;
export const LAYER_COUNT = 12;

const seededRandom = (seed: number) => {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
};

export const generateVector = (seed: number, dim: number) =>
  Array.from({ length: dim }, (_, i) => Number((seededRandom(seed + i * 1.91) * 2 - 1).toFixed(2)));

const hashString = (text: string) => {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
};

const addVectors = (left: number[], right: number[]) =>
  left.map((value, index) => Number((value + right[index] * 0.35).toFixed(2)));

const softmax = (values: number[], temperature = 1) => {
  const safeT = Math.max(temperature, 0.05);
  const scaled = values.map((value) => value / safeT);
  const max = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => Number((value / sum).toFixed(3)));
};

const positionalVector = (tokenIndex: number) =>
  Array.from({ length: DIMENSIONS }, (_, dim) => {
    const wave = dim % 2 === 0 ? Math.sin(tokenIndex / (dim + 1)) : Math.cos(tokenIndex / (dim + 1));
    return Number((wave * 0.85).toFixed(2));
  });

const DEFAULT_VOCAB = [
  { token: 'rug', logit: 1.25 },
  { token: 'mat', logit: 2.65 },
  { token: 'floor', logit: 0.9 },
  { token: 'sofa', logit: 0.42 },
  { token: 'nap', logit: -0.2 },
];

const CANDIDATE_POOL = [
  'time', 'way', 'day', 'world', 'life', 'hand', 'part', 'eye', 'place', 'work',
  'week', 'house', 'room', 'door', 'light', 'water', 'story', 'fact', 'idea', 'moon',
  'tree', 'road', 'song', 'rain', 'fire', 'bird', 'stone', 'dream', 'wind', 'book',
  'night', 'city', 'star', 'sea', 'dog', 'floor', 'chair', 'table', 'garden', 'river',
];

const buildVocab = (tokens: string[]) => {
  if (tokens.join(' ').toLowerCase() === DEFAULT_SENTENCE.toLowerCase()) {
    return DEFAULT_VOCAB;
  }

  const seed = hashString(tokens.join(' ').toLowerCase());
  const candidates: string[] = [];
  let cursor = seed % CANDIDATE_POOL.length;
  while (candidates.length < 5) {
    const word = CANDIDATE_POOL[cursor % CANDIDATE_POOL.length];
    if (!candidates.includes(word) && !tokens.includes(word)) {
      candidates.push(word);
    }
    cursor += 7;
  }

  const winnerIndex = seed % candidates.length;
  return candidates.map((token, index) => ({
    token,
    logit: Number((seededRandom(seed + index * 13) * 2.2 - 0.3 + (index === winnerIndex ? 1.6 : 0)).toFixed(2)),
  }));
};

export const buildSimulation = (sentence: string): Simulation => {
  const tokens = sentence.trim().split(/\s+/).filter(Boolean).slice(0, MAX_TOKENS);
  const tokenData: TokenDatum[] = tokens.map((text, index) => {
    const tokenSeed = (hashString(text.toLowerCase()) % 997) + index * 13;
    const embedding = generateVector(11 + tokenSeed * 1.7, DIMENSIONS);
    const positional = positionalVector(index);

    return {
      text,
      embedding,
      positional,
      combined: addVectors(embedding, positional),
      qkv: {
        query: generateVector(101 + tokenSeed * 1.9, DIMENSIONS),
        key: generateVector(203 + tokenSeed * 2.3, DIMENSIONS),
        value: generateVector(307 + tokenSeed * 2.9, DIMENSIONS),
      },
    };
  });

  const vocabLogits = buildVocab(tokens);
  const nextToken = vocabLogits.reduce((best, item) => (item.logit > best.logit ? item : best), vocabLogits[0]).token;

  return { sentence: tokens.join(' '), tokens, tokenData, vocabLogits, nextToken };
};

const HEAD_CONFIG = [
  {
    id: 'h1',
    label: 'Head 1',
    color: '#ffb454',
    description: 'nearby syntax',
    bias: (activeIndex: number, tokenIndex: number) => 1.4 - Math.abs(activeIndex - tokenIndex) * 0.42,
  },
  {
    id: 'h2',
    label: 'Head 2',
    color: '#ff7a9c',
    description: 'subject link',
    bias: (_activeIndex: number, tokenIndex: number) => (tokenIndex === 1 ? 1.15 : tokenIndex === 0 ? 0.55 : 0.1),
  },
  {
    id: 'h3',
    label: 'Head 3',
    color: '#7bd88f',
    description: 'position rhythm',
    bias: (activeIndex: number, tokenIndex: number) => Math.cos((activeIndex + 1) * (tokenIndex + 1) * 0.7) * 0.55,
  },
];

export const getAttentionHeads = (sim: Simulation, activeTokenIndex: number): AttentionHeadDatum[] => {
  const allowedTokens = sim.tokenData.slice(0, activeTokenIndex + 1);

  return HEAD_CONFIG.map((head, headIndex) => {
    const rawScores = allowedTokens.map((_, tokenIndex) => {
      const query = sim.tokenData[activeTokenIndex].qkv.query;
      const key = sim.tokenData[tokenIndex].qkv.key;
      const dot = query.reduce((total, value, dim) => total + value * key[dim], 0) / Math.sqrt(DIMENSIONS);
      return Number((dot * 0.34 + head.bias(activeTokenIndex, tokenIndex) + headIndex * 0.08).toFixed(2));
    });
    const weights = softmax(rawScores);

    return {
      id: head.id,
      label: head.label,
      color: head.color,
      description: head.description,
      attention: allowedTokens.map((token, tokenIndex) => ({
        token: token.text,
        tokenIndex,
        rawScore: rawScores[tokenIndex],
        weight: weights[tokenIndex],
      })),
    };
  });
};

export const getContextVector = (sim: Simulation, activeTokenIndex: number) => {
  const heads = getAttentionHeads(sim, activeTokenIndex);

  return Array.from({ length: DIMENSIONS }, (_, dim) => {
    const value = heads.reduce((headTotal, head) => {
      const headValue = head.attention.reduce(
        (tokenTotal, item) => tokenTotal + sim.tokenData[item.tokenIndex].qkv.value[dim] * item.weight,
        0,
      );
      return headTotal + headValue;
    }, 0) / heads.length;

    return Number(value.toFixed(2));
  });
};

export const FFN_EXPANDED_VECTOR = generateVector(971, 16);

export const getFfnOutputVector = (sim: Simulation, activeTokenIndex: number) => {
  const contextVector = getContextVector(sim, activeTokenIndex);
  const residual = generateVector(449 + activeTokenIndex * 13, DIMENSIONS);

  return contextVector.map((value, index) => Number((value * 0.7 + residual[index] * 0.3).toFixed(2)));
};

export const getVocabulary = (sim: Simulation, temperature = 1): VocabularyDatum[] => {
  const probabilities = softmax(sim.vocabLogits.map((item) => item.logit), temperature);

  return sim.vocabLogits
    .map((item, index) => ({ ...item, probability: probabilities[index] }))
    .sort((a, b) => b.logit - a.logit);
};

const formatVector = (values: number[], maxItems = 4) =>
  `[${values.slice(0, maxItems).map((value) => value.toFixed(2)).join(', ')}${values.length > maxItems ? ', ...' : ''}]`;

const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

export const getReadoutRows = (sim: Simulation, stepIndex: number, activeTokenIndex: number, temperature = 1) => {
  const step = STEPS[stepIndex];
  const activeToken = sim.tokenData[activeTokenIndex];
  const heads = getAttentionHeads(sim, activeTokenIndex);
  const contextVector = getContextVector(sim, activeTokenIndex);
  const ffnOutputVector = getFfnOutputVector(sim, activeTokenIndex);

  switch (step.id) {
    case 'input':
      return [
        { label: 'Input', value: sim.sentence },
        { label: 'Active token', value: `"${activeToken.text}"` },
        { label: 'Allowed context', value: sim.tokens.slice(0, activeTokenIndex + 1).join(' ') },
        { label: 'Goal', value: `predict "${sim.nextToken}"` },
      ];
    case 'embedding':
      return [
        { label: 'Vector width', value: `${DIMENSIONS} toy features` },
        { label: 'Embedding', value: formatVector(activeToken.embedding) },
        { label: 'Strongest feature', value: activeToken.embedding.reduce((best, value) => (Math.abs(value) > Math.abs(best) ? value : best), 0).toFixed(2) },
      ];
    case 'positional':
      return [
        { label: 'Token index', value: String(activeTokenIndex) },
        { label: 'Position wave', value: formatVector(activeToken.positional) },
        { label: 'Combined', value: formatVector(activeToken.combined) },
      ];
    case 'qkv':
      return [
        { label: 'Query asks', value: formatVector(activeToken.qkv.query, 3) },
        { label: 'Key matches', value: formatVector(activeToken.qkv.key, 3) },
        { label: 'Value carries', value: formatVector(activeToken.qkv.value, 3) },
      ];
    case 'dot-product':
      return heads.map((head) => ({
        label: head.label,
        value: head.attention.map((item) => `${item.token} ${item.rawScore.toFixed(2)}`).join(' / '),
      }));
    case 'softmax':
      return heads.map((head) => ({
        label: head.label,
        value: head.attention.map((item) => `${item.token} ${formatPercent(item.weight)}`).join(' / '),
      }));
    case 'value-agg':
      return [
        { label: 'Merged heads', value: `${heads.length} head outputs` },
        { label: 'Context vector', value: formatVector(contextVector) },
        { label: 'Top focus', value: heads.map((head) => `${head.label}: ${head.attention.reduce((best, item) => (item.weight > best.weight ? item : best), head.attention[0]).token}`).join(' / ') },
      ];
    case 'ffn':
      return [
        { label: 'Input width', value: `${DIMENSIONS}` },
        { label: 'Hidden width', value: `${FFN_EXPANDED_VECTOR.length}` },
        { label: 'Output sample', value: formatVector(ffnOutputVector) },
      ];
    case 'unembedding':
      return getVocabulary(sim).map((item) => ({
        label: item.token,
        value: `logit ${item.logit.toFixed(2)}`,
      }));
    case 'output':
      return [
        { label: 'Temperature', value: `T = ${temperature.toFixed(2)}` },
        ...getVocabulary(sim, temperature).map((item) => ({
          label: item.token,
          value: formatPercent(item.probability),
        })),
      ];
    default:
      return [];
  }
};
