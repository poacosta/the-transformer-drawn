import { describe, expect, it } from 'vitest';
import {
  DEFAULT_SENTENCE,
  type ModelParams,
  PARAMS,
  buildSimulation,
  getVocabulary,
  positionalVector,
} from './simulationData';

const SENTENCES = [DEFAULT_SENTENCE, 'A dog runs to the river', 'hello', 'the the the'];

const mutableParams = (): ModelParams => structuredClone(PARAMS);

const logitsOf = (sentence: string, params: ModelParams) =>
  buildSimulation(sentence, { params }).vocabLogits.map((item) => item.logit);

const logitShift = (params: ModelParams) => {
  const base = logitsOf(DEFAULT_SENTENCE, PARAMS);
  return Math.max(...logitsOf(DEFAULT_SENTENCE, params).map((value, i) => Math.abs(value - base[i])));
};

const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);

describe('determinism', () => {
  it.each(SENTENCES)('builds the same sheet twice for "%s"', (sentence) => {
    expect(buildSimulation(sentence)).toEqual(buildSimulation(sentence));
  });
});

describe('shapes', () => {
  const sim = buildSimulation(DEFAULT_SENTENCE);

  it('has 4-wide Q/K/V per head', () => {
    for (const token of sim.tokenData) {
      expect(token.qkv).toHaveLength(3);
      for (const lanes of token.qkv) {
        expect(lanes.query).toHaveLength(4);
        expect(lanes.key).toHaveLength(4);
        expect(lanes.value).toHaveLength(4);
      }
    }
  });

  it('concatenates to 12, projects to 8, expands the FFN to 16, and scores 5 candidates', () => {
    for (const trace of sim.positions) {
      expect(trace.concat).toHaveLength(12);
      expect(trace.context).toHaveLength(8);
      expect(trace.ffnHidden).toHaveLength(16);
      expect(trace.blockOutput).toHaveLength(8);
    }
    expect(sim.vocabLogits).toHaveLength(5);
  });
});

describe('softmax', () => {
  it.each(SENTENCES)('each head’s weights sum to 1 for "%s"', (sentence) => {
    const sim = buildSimulation(sentence);
    for (const trace of sim.positions) {
      for (const head of trace.heads) {
        expect(sum(head.attention.map((item) => item.weight))).toBeCloseTo(1, 6);
      }
    }
  });

  it.each([0.2, 0.5, 1, 2, 3])('vocabulary probabilities sum to 1 at T = %s', (temperature) => {
    const probabilities = getVocabulary(buildSimulation(DEFAULT_SENTENCE), temperature).map((item) => item.probability);
    expect(sum(probabilities)).toBeCloseTo(1, 6);
  });
});

describe('causal mask', () => {
  const sim = buildSimulation(DEFAULT_SENTENCE);

  it('lets the first token attend only to itself', () => {
    for (const head of sim.positions[0].heads) {
      expect(head.attention).toHaveLength(1);
      expect(head.attention[0].tokenIndex).toBe(0);
      expect(head.attention[0].weight).toBeCloseTo(1, 12);
    }
  });

  it('gives token t exactly t + 1 weights', () => {
    sim.positions.forEach((trace, t) => {
      for (const head of trace.heads) {
        expect(head.attention.map((item) => item.tokenIndex)).toEqual(Array.from({ length: t + 1 }, (_, j) => j));
      }
    });
  });
});

describe('chaining', () => {
  it('feeds the positional vector into that position’s Q/K/V and the attention weights', () => {
    const base = buildSimulation(DEFAULT_SENTENCE);
    // Perturb one dimension only: a uniform shift across all dimensions would be removed by
    // LayerNorm's mean subtraction and change nothing downstream.
    const shifted = buildSimulation(DEFAULT_SENTENCE, {
      positional: (index) =>
        index === 1 ? positionalVector(index).map((value, dim) => (dim === 0 ? value + 2 : value)) : positionalVector(index),
    });
    const maxDiff = (left: number[], right: number[]) => Math.max(...left.map((value, i) => Math.abs(value - right[i])));
    const flatQkv = (sim: typeof base, i: number) => sim.tokenData[i].qkv.flatMap((lanes) => [...lanes.query, ...lanes.key, ...lanes.value]);
    const weights = (sim: typeof base) => sim.positions[2].heads.flatMap((head) => head.attention.map((item) => item.weight));

    expect(maxDiff(flatQkv(shifted, 1), flatQkv(base, 1))).toBeGreaterThan(1e-2);
    expect(flatQkv(shifted, 0)).toEqual(flatQkv(base, 0));
    expect(maxDiff(weights(shifted), weights(base))).toBeGreaterThan(1e-3);
  });

  it('changes the logits when W_O is perturbed', () => {
    const params = mutableParams();
    params.W_O[0][0] += 0.5;
    expect(logitShift(params)).toBeGreaterThan(1e-3);
  });

  it('changes the logits when an attention projection is perturbed', () => {
    const params = mutableParams();
    params.W_V[1][2][3] += 0.5;
    expect(logitShift(params)).toBeGreaterThan(1e-3);
  });

  it('changes the logits when W1 is perturbed', () => {
    const params = mutableParams();
    params.W1[3][5] += 0.5;
    expect(logitShift(params)).toBeGreaterThan(1e-3);
  });
});

describe('default sentence', () => {
  it('predicts "mat"', () => {
    const sim = buildSimulation(DEFAULT_SENTENCE);
    expect(sim.nextToken).toBe('mat');
    expect(getVocabulary(sim)[0].token).toBe('mat');
  });
});

describe('temperature', () => {
  it('sharpens the distribution at T = 0.2 compared with T = 3.0', () => {
    const sim = buildSimulation(DEFAULT_SENTENCE);
    expect(getVocabulary(sim, 0.2)[0].probability).toBeGreaterThan(getVocabulary(sim, 3)[0].probability);
  });
});
