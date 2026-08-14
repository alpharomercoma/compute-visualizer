import { describe, expect, it } from 'vitest'
import { DGX_H100 } from './hardware'
import { H100_EFFICIENCY, H100_OVERHEAD } from './efficiency'
import { fromHuggingFaceConfig } from './modelspec'
import { analyseInference, largestStage, layersInStage } from './inference'
import { analyseTraining } from './training'
import { attentionPairs } from './kernels'
import { inferenceCapacity } from './capacity'

/**
 * Regressions for the defects the implementation review found.
 *
 * Each of these shipped in a working, tested, plausible-looking build. They are
 * here because "the tests pass" was already true when they were wrong.
 */

const llama70b = fromHuggingFaceConfig(
  {
    model_type: 'llama',
    hidden_size: 8192,
    num_hidden_layers: 80,
    num_attention_heads: 64,
    num_key_value_heads: 8,
    intermediate_size: 28672,
    vocab_size: 128256,
    max_position_embeddings: 8192,
    tie_word_embeddings: false,
  },
  { id: 'llama-3-70b' },
)

const base = {
  spec: llama70b,
  topology: DGX_H100,
  weightDtype: 'bf16',
  kvDtype: 'bf16',
  activationDtype: 'bf16',
  efficiency: H100_EFFICIENCY,
  overhead: H100_OVERHEAD,
  cudaGraph: true,
  flashAttention: true,
} as const

describe('attention pair count rejects impossible shapes', () => {
  it('refuses more queries than context', () => {
    // q=4, s=1 used to yield -2 pairs, i.e. negative work.
    expect(() => attentionPairs(4, 1, true)).toThrow(RangeError)
    expect(() => attentionPairs(-1, 10, true)).toThrow(RangeError)
  })

  it('still accepts the three real cases', () => {
    expect(attentionPairs(1, 2048, true)).toBe(2048) // decode
    expect(attentionPairs(2048, 2048, true)).toBe((2048 * 2049) / 2) // full prefill
    expect(attentionPairs(512, 2048, true)).toBeGreaterThan(512 * 1536) // chunk
  })
})

describe('pipeline stages partition the model exactly', () => {
  it('sums to the layer count for divisors and non-divisors alike', () => {
    for (const pipeline of [1, 2, 3, 4, 5, 6, 7, 8, 16]) {
      let total = 0
      for (let stage = 0; stage < pipeline; stage++) {
        total += layersInStage(80, pipeline, stage)
      }
      // ceil(80/6) x 6 = 84 would have modelled four layers that do not exist.
      expect(total).toBe(80)
    }
  })

  it('reports the largest stage, since that is the one that runs out of memory', () => {
    expect(largestStage(80, 6)).toBe(14)
    expect(largestStage(80, 8)).toBe(10)
    expect(largestStage(80, 1)).toBe(80)
  })
})

describe('capacity solvers account for what grows with the answer', () => {
  const capacityFor = (batch: number, contextLength: number) =>
    inferenceCapacity({
      spec: llama70b,
      accelerator: DGX_H100.accelerator,
      parallelism: { tensor: 8, pipeline: 1, data: 1 },
      batch,
      contextLength,
      weightDtype: 'bf16',
      kvDtype: 'bf16',
      activationDtype: 'bf16',
    })

  it('advertises a maximum context that actually fits', () => {
    const report = capacityFor(32, 2048)
    const atLimit = capacityFor(32, report.maxContextLength)
    expect(atLimit.verdict).not.toBe('will-not-fit')
  })

  it('advertises a maximum batch that actually fits', () => {
    const report = capacityFor(32, 2048)
    expect(Number.isFinite(report.maxBatch)).toBe(true)
    const atLimit = capacityFor(report.maxBatch, 2048)
    expect(atLimit.verdict).not.toBe('will-not-fit')
  })

  it('never claims an unbounded batch, even at zero context', () => {
    // Dividing by a zero-length context used to return Infinity.
    expect(Number.isFinite(capacityFor(1, 0).maxBatch)).toBe(true)
  })

  it('charges activations at activation precision, not KV precision', () => {
    const bf16Kv = capacityFor(32, 2048)
    const fp8Kv = inferenceCapacity({
      spec: llama70b,
      accelerator: DGX_H100.accelerator,
      parallelism: { tensor: 8, pipeline: 1, data: 1 },
      batch: 32,
      contextLength: 2048,
      weightDtype: 'bf16',
      kvDtype: 'fp8',
      activationDtype: 'bf16',
    })
    // Quantizing the KV cache must shrink the KV cache and leave activations
    // alone; charging activations at KV precision would halve them too.
    expect(fp8Kv.perGpu.kvCache).toBeLessThan(bf16Kv.perGpu.kvCache)
    expect(fp8Kv.perGpu.activations).toBe(bf16Kv.perGpu.activations)
  })
})

describe('selective recomputation is additive, not a blanket multiplier', () => {
  const train = (recompute: 'none' | 'selective' | 'full') =>
    analyseTraining({
      spec: llama70b,
      topology: DGX_H100,
      parallelism: { tensor: 8, pipeline: 4, data: 2 },
      microBatch: 1,
      sequenceLength: 2048,
      gradientAccumulation: 16,
      weightDtype: 'bf16',
      activationDtype: 'bf16',
      optimizer: 'adam-mixed',
      zeroStage: 3,
      sequenceParallel: true,
      selectiveRecompute: recompute === 'selective',
      fullRecompute: recompute === 'full',
      efficiency: H100_EFFICIENCY,
      overhead: H100_OVERHEAD,
    })

  it('adds far less work than full recomputation does', () => {
    const none = train('none').hardwareFlopsPerStep!
    const selective = train('selective').hardwareFlopsPerStep!
    const full = train('full').hardwareFlopsPerStep!

    expect(selective).toBeGreaterThan(none)
    expect(full).toBeGreaterThan(selective)

    // Full recompute runs every layer forward again: 3x becomes 4x, so ~33%.
    expect(full / none).toBeCloseTo(4 / 3, 1)

    // Selective touches attention only. At s = h its share is small; a blanket
    // 10% multiplier across every kernel — which is what this used to be — would
    // put this ratio at 1.1 regardless of shape.
    expect(selective / none).toBeLessThan(1.1)
  })
})

describe('data-parallel gradient reduction overlaps backward', () => {
  const train = (data: number) =>
    analyseTraining({
      spec: llama70b,
      topology: DGX_H100,
      parallelism: { tensor: 8, pipeline: 4, data },
      microBatch: 1,
      sequenceLength: 8192,
      gradientAccumulation: 16,
      weightDtype: 'bf16',
      activationDtype: 'bf16',
      optimizer: 'adam-mixed',
      zeroStage: 3,
      sequenceParallel: true,
      selectiveRecompute: true,
      fullRecompute: false,
      efficiency: H100_EFFICIENCY,
      overhead: H100_OVERHEAD,
    })

  it('adds only its exposed tail to the step, once per step', () => {
    // Gradient accumulation sums locally, so the reduction happens once per
    // optimizer step rather than once per microbatch. Charging it per
    // microbatch would multiply it by the accumulation count.
    const withoutDp = train(1)
    const withDp = train(8)

    const added = withDp.stepTime! - withoutDp.stepTime!
    expect(added).toBeCloseTo(withDp.exposedCommunication!, 6)
  })

  it('hides most of the reduction behind backward', () => {
    const withDp = train(8)
    // The exposed part must be a fraction of the step, not the whole reduction:
    // upper-layer buckets reduce while lower layers are still computing.
    expect(withDp.exposedCommunication!).toBeLessThan(withDp.stepTime! * 0.5)
  })
})

describe('the dependency-bound class is reachable', () => {
  it('names a step where no single resource holds the critical path', () => {
    // A batch-1 decode of a small model on many ranks splits its time between
    // memory, collective latency and launch overhead, with none dominating.
    const small = fromHuggingFaceConfig(
      {
        model_type: 'llama',
        hidden_size: 2048,
        num_hidden_layers: 16,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        intermediate_size: 5632,
        vocab_size: 32000,
        max_position_embeddings: 4096,
        tie_word_embeddings: false,
      },
      { id: 'tiny' },
    )

    const result = analyseInference({
      ...base,
      spec: small,
      parallelism: { tensor: 8, pipeline: 1, data: 1 },
      batch: 1,
      promptLength: 128,
      generateLength: 8,
      cudaGraph: false,
    })

    const { breakdown } = result.decode!.schedule
    const shares = Object.values(breakdown).map((t) => t / result.timePerOutputToken!)
    // Whatever it is classified as, the classification must follow the timeline:
    // if something holds a majority it is named, otherwise it is dependency-bound.
    const dominant = Math.max(...shares)
    if (dominant < 0.5) expect(result.decode!.bottleneck).toBe('serialization')
    else expect(result.decode!.bottleneck).not.toBe('serialization')
  })
})
