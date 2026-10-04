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
  normed: number[];
  qkv: VectorSet[];
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
  output: number[];
};

export type PositionTrace = {
  heads: AttentionHeadDatum[];
  concat: number[];
  context: number[];
  residual: number[];
  ffnInput: number[];
  ffnHidden: number[];
  ffnOutput: number[];
  blockOutput: number[];
  final: number[];
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
  positions: PositionTrace[];
  vocabLogits: { token: string; logit: number }[];
  nextToken: string;
};

// Head personalities are a teaching device: a small hand-written additive bias on each head's
// scores, so the three patterns are easy to tell apart. Set to false to let the seeded
// projections alone decide.
export const TEACHING_BIAS = true;

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
    description: 'Each combined vector is layer-normalized, then every head projects it into its own Query, Key, and Value lanes: what I seek, what I match against, and what I can pass forward.',
    readoutTitle: 'Active Q/K/V sample',
    note: 'q = LN₁(x) · W_Qʰ\n(same for k, v with W_Kʰ, W_Vʰ)',
  },
  {
    id: 'dot-product',
    title: 'Attention Scores',
    shortTitle: 'Scores',
    description: TEACHING_BIAS
      ? 'Each attention head compares the active Query with allowed Keys through its own projections, plus a small hand-set bias that gives it a readable personality.'
      : 'Each attention head compares the active Query with allowed Keys through its own projections.',
    readoutTitle: 'Raw scores',
    note: TEACHING_BIAS ? 'score = Q·K⊤ / √d_k + b_head' : 'score = Q·K⊤ / √d_k',
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
    description: 'Every head makes its own weighted Value blend. The blends are concatenated, projected back to 8 features, and added to the token’s own vector (the residual connection).',
    readoutTitle: 'Weighted value sum',
    note: 'ctx = concat_h(Σ w·V) · W_O\nr = x + ctx',
  },
  {
    id: 'ffn',
    title: 'Feed-Forward Network',
    shortTitle: 'FFN',
    description: 'The vector is normalized, expanded to 16 hidden features through GELU, contracted back to the model width, and added to its input again.',
    readoutTitle: 'Hidden layer pulse',
    note: 'f = r + GELU(LN₂(r)·W₁ + b₁)·W₂ + b₂\nGELU: tanh approximation · 8 → 16 → 8',
  },
  {
    id: 'unembedding',
    title: 'Vocabulary Projection',
    shortTitle: 'Vocab',
    description: 'The last token’s final vector is normalized and compared with each candidate’s embedding row (the output layer reuses the embeddings), producing logits.',
    readoutTitle: 'Candidate logits',
    note: 'logits = LN_f(f) · E⊤',
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
export const HEAD_COUNT = 3;
// Toy choice: d_k · h = 12 ≠ d. Real models usually use d_k = d / h, so the concatenated heads
// are exactly d wide; here W_O maps 12 → 8 so each head keeps a readable 4-wide lane.
export const HEAD_DIMENSIONS = 4;
export const FFN_DIMENSIONS = 16;
const LAYER_NORM_EPSILON = 1e-5;

// Chosen by a one-off deterministic search (the first seed, counting up from 1, where "mat" leads
// the next candidate by at least 0.4 logits) so that the default sentence "The cat sat on the"
// predicts "mat" clearly. The logits themselves are not special-cased.
export const MODEL_SEED = 11;

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

const mulberry32 = (seed: number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

type Matrix = number[][];

export type LayerNormParams = { gamma: number[]; beta: number[] };

export type ModelParams = {
  W_Q: Matrix[];
  W_K: Matrix[];
  W_V: Matrix[];
  W_O: Matrix;
  ln1: LayerNormParams;
  ln2: LayerNormParams;
  lnf: LayerNormParams;
  W1: Matrix;
  b1: number[];
  W2: Matrix;
  b2: number[];
};

// Uniform in ±√(3 / fan_in), i.e. variance 1 / fan_in, so every projection keeps values in
// roughly the same range as its input (about −3 to 3 on the sheet). Each tensor draws from its
// own stream, keyed by the global seed and the tensor's name.
const seededMatrix = (seed: number, name: string, rows: number, cols: number): Matrix => {
  const next = mulberry32(seed ^ hashString(name));
  const limit = Math.sqrt(3 / rows);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (next() * 2 - 1) * limit));
};

const seededBias = (seed: number, name: string, length: number) => {
  const next = mulberry32(seed ^ hashString(name));
  return Array.from({ length }, () => (next() * 2 - 1) * 0.1);
};

const identityLayerNorm = (): LayerNormParams => ({
  gamma: Array.from({ length: DIMENSIONS }, () => 1),
  beta: Array.from({ length: DIMENSIONS }, () => 0),
});

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
};

export const createParams = (seed: number): ModelParams =>
  deepFreeze({
    W_Q: Array.from({ length: HEAD_COUNT }, (_, h) => seededMatrix(seed, `W_Q.${h}`, DIMENSIONS, HEAD_DIMENSIONS)),
    W_K: Array.from({ length: HEAD_COUNT }, (_, h) => seededMatrix(seed, `W_K.${h}`, DIMENSIONS, HEAD_DIMENSIONS)),
    W_V: Array.from({ length: HEAD_COUNT }, (_, h) => seededMatrix(seed, `W_V.${h}`, DIMENSIONS, HEAD_DIMENSIONS)),
    W_O: seededMatrix(seed, 'W_O', HEAD_COUNT * HEAD_DIMENSIONS, DIMENSIONS),
    ln1: identityLayerNorm(),
    ln2: identityLayerNorm(),
    lnf: identityLayerNorm(),
    W1: seededMatrix(seed, 'W1', DIMENSIONS, FFN_DIMENSIONS),
    b1: seededBias(seed, 'b1', FFN_DIMENSIONS),
    W2: seededMatrix(seed, 'W2', FFN_DIMENSIONS, DIMENSIONS),
    b2: seededBias(seed, 'b2', DIMENSIONS),
  });

export const PARAMS = createParams(MODEL_SEED);

const dot = (left: number[], right: number[]) => left.reduce((total, value, index) => total + value * right[index], 0);

const add = (left: number[], right: number[]) => left.map((value, index) => value + right[index]);

// Row vector times matrix: (1 × rows) · (rows × cols) → (1 × cols).
const project = (vector: number[], matrix: Matrix) =>
  matrix[0].map((_, col) => vector.reduce((total, value, row) => total + value * matrix[row][col], 0));

const layerNorm = (vector: number[], { gamma, beta }: LayerNormParams) => {
  const mean = vector.reduce((total, value) => total + value, 0) / vector.length;
  const variance = vector.reduce((total, value) => total + (value - mean) ** 2, 0) / vector.length;
  const scale = Math.sqrt(variance + LAYER_NORM_EPSILON);
  return vector.map((value, index) => ((value - mean) / scale) * gamma[index] + beta[index]);
};

// GELU, tanh approximation (as in GPT-2).
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

const softmax = (values: number[], temperature = 1) => {
  const safeT = Math.max(temperature, 0.05);
  const scaled = values.map((value) => value / safeT);
  const max = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
};

export const positionalVector = (tokenIndex: number) =>
  Array.from({ length: DIMENSIONS }, (_, dim) => {
    const wave = dim % 2 === 0 ? Math.sin(tokenIndex / (dim + 1)) : Math.cos(tokenIndex / (dim + 1));
    return Number((wave * 0.85).toFixed(2));
  });

const addPosition = (embedding: number[], positional: number[]) =>
  embedding.map((value, index) => Number((value + positional[index] * 0.35).toFixed(2)));

// The token embedding table E, hashed from the token text. The output layer is tied to it, as in
// GPT-2, so vocabulary candidates are scored against these same rows.
export const embed = (token: string) => generateVector(11 + (hashString(token.toLowerCase()) % 997) * 1.7, DIMENSIONS);

const DEFAULT_VOCAB = ['rug', 'mat', 'floor', 'sofa', 'nap'];

const CANDIDATE_POOL = [
  'time', 'way', 'day', 'world', 'life', 'hand', 'part', 'eye', 'place', 'work',
  'week', 'house', 'room', 'door', 'light', 'water', 'story', 'fact', 'idea', 'moon',
  'tree', 'road', 'song', 'rain', 'fire', 'bird', 'stone', 'dream', 'wind', 'book',
  'night', 'city', 'star', 'sea', 'dog', 'floor', 'chair', 'table', 'garden', 'river',
];

const pickCandidates = (tokens: string[]) => {
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
  return candidates;
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

export type SimulationOptions = {
  params?: ModelParams;
  positional?: (tokenIndex: number) => number[];
};

// One Pre-LN decoder block (GPT-2 layout), run for every query position t with causal access to
// positions 0…t. Each stage reads only the previous stage's output.
const runPosition = (tokenData: TokenDatum[], t: number, params: ModelParams): PositionTrace => {
  const heads = HEAD_CONFIG.map((head, h) => {
    const query = tokenData[t].qkv[h].query;
    const allowed = tokenData.slice(0, t + 1);
    const rawScores = allowed.map((token, j) => {
      const scaled = dot(query, token.qkv[h].key) / Math.sqrt(HEAD_DIMENSIONS);
      return scaled + (TEACHING_BIAS ? head.bias(t, j) : 0);
    });
    const weights = softmax(rawScores);
    const output = Array.from({ length: HEAD_DIMENSIONS }, (_, dim) =>
      allowed.reduce((total, token, j) => total + weights[j] * token.qkv[h].value[dim], 0),
    );

    return {
      id: head.id,
      label: head.label,
      color: head.color,
      description: TEACHING_BIAS ? head.description : '',
      attention: allowed.map((token, tokenIndex) => ({
        token: token.text,
        tokenIndex,
        rawScore: rawScores[tokenIndex],
        weight: weights[tokenIndex],
      })),
      output,
    };
  });

  const concat = heads.flatMap((head) => head.output);
  const context = project(concat, params.W_O);
  const residual = add(tokenData[t].combined, context);
  const ffnInput = layerNorm(residual, params.ln2);
  const ffnHidden = add(project(ffnInput, params.W1), params.b1).map(gelu);
  const ffnOutput = add(project(ffnHidden, params.W2), params.b2);
  const blockOutput = add(residual, ffnOutput);
  const final = layerNorm(blockOutput, params.lnf);

  return { heads, concat, context, residual, ffnInput, ffnHidden, ffnOutput, blockOutput, final };
};

export const buildSimulation = (sentence: string, options: SimulationOptions = {}): Simulation => {
  const params = options.params ?? PARAMS;
  const getPositional = options.positional ?? positionalVector;
  const tokens = sentence.trim().split(/\s+/).filter(Boolean).slice(0, MAX_TOKENS);

  const tokenData: TokenDatum[] = tokens.map((text, index) => {
    const embedding = embed(text);
    const positional = getPositional(index);
    const combined = addPosition(embedding, positional);
    const normed = layerNorm(combined, params.ln1);

    return {
      text,
      embedding,
      positional,
      combined,
      normed,
      qkv: params.W_Q.map((_, h) => ({
        query: project(normed, params.W_Q[h]),
        key: project(normed, params.W_K[h]),
        value: project(normed, params.W_V[h]),
      })),
    };
  });

  const positions = tokenData.map((_, t) => runPosition(tokenData, t, params));
  const last = positions[positions.length - 1];
  const vocabLogits = pickCandidates(tokens).map((token) => ({ token, logit: last ? dot(last.final, embed(token)) : 0 }));
  const nextToken = vocabLogits.reduce((best, item) => (item.logit > best.logit ? item : best), vocabLogits[0]).token;

  return { sentence: tokens.join(' '), tokens, tokenData, positions, vocabLogits, nextToken };
};

export const getAttentionHeads = (sim: Simulation, activeTokenIndex: number): AttentionHeadDatum[] =>
  sim.positions[activeTokenIndex].heads;

export const getHeadIndex = (headId: string | null) => Math.max(0, HEAD_CONFIG.findIndex((head) => head.id === headId));

export const getVocabulary = (sim: Simulation, temperature = 1): VocabularyDatum[] => {
  const probabilities = softmax(sim.vocabLogits.map((item) => item.logit), temperature);

  return sim.vocabLogits
    .map((item, index) => ({ ...item, probability: probabilities[index] }))
    .sort((a, b) => b.logit - a.logit);
};

const formatVector = (values: number[], maxItems = 4) =>
  `[${values.slice(0, maxItems).map((value) => value.toFixed(2)).join(', ')}${values.length > maxItems ? ', ...' : ''}]`;

const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

export const getReadoutRows = (sim: Simulation, stepIndex: number, activeTokenIndex: number, temperature = 1, headId: string | null = null) => {
  const step = STEPS[stepIndex];
  const activeToken = sim.tokenData[activeTokenIndex];
  const trace = sim.positions[activeTokenIndex];
  const heads = trace.heads;
  const headIndex = getHeadIndex(headId);
  const lastIndex = sim.tokens.length - 1;

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
        { label: 'LN₁(x)', value: formatVector(activeToken.normed) },
        { label: `${heads[headIndex].label} query`, value: formatVector(activeToken.qkv[headIndex].query) },
        { label: `${heads[headIndex].label} key`, value: formatVector(activeToken.qkv[headIndex].key) },
        { label: `${heads[headIndex].label} value`, value: formatVector(activeToken.qkv[headIndex].value) },
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
        { label: 'Concat heads', value: `${heads.length} × ${HEAD_DIMENSIONS} = ${trace.concat.length}` },
        { label: 'ctx = concat · W_O', value: formatVector(trace.context) },
        { label: 'r = x + ctx', value: formatVector(trace.residual) },
        { label: 'Top focus', value: heads.map((head) => `${head.label}: ${head.attention.reduce((best, item) => (item.weight > best.weight ? item : best), head.attention[0]).token}`).join(' / ') },
      ];
    case 'ffn':
      return [
        { label: 'LN₂(r)', value: formatVector(trace.ffnInput) },
        { label: `GELU hidden (${trace.ffnHidden.length})`, value: formatVector(trace.ffnHidden) },
        { label: 'f = r + MLP', value: formatVector(trace.blockOutput) },
      ];
    case 'unembedding':
      return [
        { label: `LN_f(f) at "${sim.tokens[lastIndex]}"`, value: formatVector(sim.positions[lastIndex].final) },
        ...getVocabulary(sim).map((item) => ({
          label: item.token,
          value: `logit ${item.logit.toFixed(2)}`,
        })),
      ];
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
