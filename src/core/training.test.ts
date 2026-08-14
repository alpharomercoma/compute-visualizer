import { describe, expect, it } from 'vitest'
import { DGX_H100 } from './hardware'
import { H100_EFFICIENCY, H100_OVERHEAD } from './efficiency'
import { fromHuggingFaceConfig } from './modelspec'
import { analyseTraining, sixNDFlops, timeToTrain } from './training'
import { nonEmbeddingParameters } from './params'

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
  microBatch: 1,
  sequenceLength: 8192,
  gradientAccumulation: 16,
  weightDtype: 'bf16',
  activationDtype: 'bf16',
  optimizer: 'adam-mixed',
  sequenceParallel: true,
  efficiency: H100_EFFICIENCY,
  overhead: H100_OVERHEAD,
} as const

const run = (over: Partial<Parameters<typeof analyseTraining>[0]> = {}) =>
  analyseTraining({
    ...base,
    parallelism: { tensor: 8, pipeline: 4, data: 2 },
    zeroStage: 3,
    selectiveRecompute: true,
    fullRecompute: false,
    ...over,
  })

describe('MFU versus HFU', () => {
  it('never reports hardware FLOPs below model FLOPs', () => {
    const result = run()
    expect(result.hfu!).toBeGreaterThanOrEqual(result.mfu!)
    expect(result.hardwareFlopsPerStep!).toBeGreaterThanOrEqual(result.modelFlopsPerStep!)
  })

  it('lands in the band real H100 training runs report', () => {
    // Llama 3 405B on 16,384 H100s: 400 TFLOP/s per GPU, BF16 MFU 38-43%. A
    // model predicting far above that for a comparable setup is wrong.
    const result = run()
    expect(result.mfu!).toBeGreaterThan(0.25)
    expect(result.mfu!).toBeLessThan(0.55)
  })

  it('shows full recomputation raising HFU while lowering MFU', () => {
    // The signature behaviour PaLM used to argue MFU is the throughput-honest
    // metric: recomputation makes the hardware busier and the run slower. A
    // model that moved both in the same direction would be wrong.
    const selective = run({ selectiveRecompute: true, fullRecompute: false })
    const full = run({ selectiveRecompute: false, fullRecompute: true })

    expect(full.mfu!).toBeLessThan(selective.mfu!)
    expect(full.hfu!).toBeGreaterThan(selective.hfu!)
    expect(full.stepTime!).toBeGreaterThan(selective.stepTime!)
  })
})

describe('memory', () => {
  it('makes 70B untrainable on a single node, which is the truth', () => {
    // 70B with Adam is ~16 bytes per parameter. Over 8 ranks that is 131 GiB of
    // optimizer state alone against 80 GiB of HBM.
    const single = run({
      parallelism: { tensor: 8, pipeline: 1, data: 1 },
      zeroStage: 1,
      selectiveRecompute: false,
      fullRecompute: false,
    })
    expect(single.feasible).toBe(false)
    expect(single.stepTime).toBeUndefined()
  })

  it('makes full recomputation dramatically cheaper in activation memory', () => {
    const selective = run({ selectiveRecompute: true, fullRecompute: false })
    const full = run({ selectiveRecompute: false, fullRecompute: true })

    // Selective keeps 34*s*b*h*L/t; full keeps only each layer's input.
    expect(full.capacity.perGpu.activations).toBeLessThan(
      selective.capacity.perGpu.activations / 10,
    )
  })

  it('shards optimizer state across data-parallel ranks under ZeRO', () => {
    const stage1 = run({ zeroStage: 1 })
    const stage3 = run({ zeroStage: 3 })

    // Stage 1 partitions optimizer state only; stage 3 partitions weights and
    // gradients too, so its total must be strictly smaller.
    expect(stage3.capacity.perGpu.total).toBeLessThan(stage1.capacity.perGpu.total)
    expect(stage3.capacity.perGpu.weights).toBeLessThan(stage1.capacity.perGpu.weights)
  })
})

describe('the pipeline bubble', () => {
  it('shrinks as microbatches per step grow', () => {
    const few = run({ gradientAccumulation: 4 })
    const many = run({ gradientAccumulation: 64 })
    expect(many.bubbleFraction!).toBeLessThan(few.bubbleFraction!)
    // (p-1)/m: 3/4 against 3/64.
    expect(few.bubbleFraction!).toBeCloseTo(0.75, 5)
    expect(many.bubbleFraction!).toBeCloseTo(3 / 64, 5)
  })

  it('vanishes without pipeline parallelism', () => {
    const result = run({ parallelism: { tensor: 8, pipeline: 1, data: 8 } })
    expect(result.bubbleFraction).toBe(0)
  })
})

describe('6ND as a cross-check', () => {
  it('agrees with the structural model to within the attention term', () => {
    const nonEmbedding = nonEmbeddingParameters(llama70b)
    const tokens = 8192 * 16 * 2 // sequence x accumulation x replicas
    const estimate = sixNDFlops(nonEmbedding, tokens)
    const structural = run().modelFlopsPerStep!

    // 6ND omits the quadratic attention term, whose share is about s/(6h) --
    // roughly 17% at s = h = 8192. So the structural figure should be modestly
    // larger, never smaller, and never by a factor.
    expect(structural).toBeGreaterThan(estimate)
    expect(structural / estimate).toBeLessThan(1.5)
  })
})

describe('time to train', () => {
  it('converts throughput into days', () => {
    expect(timeToTrain(1e6, 15e12)).toBeCloseTo(173.6, 0)
  })

  it('reports an unreachable target rather than zero', () => {
    expect(timeToTrain(0, 15e12)).toBe(Number.POSITIVE_INFINITY)
  })
})
