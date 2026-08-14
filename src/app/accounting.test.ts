/**
 * The only claim worth testing here is the one the interface makes: every share
 * on the page is a fraction of the same step, and the slices account for all of
 * it. Both of the configurations below were reachable in two clicks and both
 * reported a share that was wrong by an integer factor.
 */

import { describe, expect, test } from 'vitest'
import { accountInference, accountTraining } from './accounting'
import { DGX_H100 } from '../core/hardware'
import { H100_EFFICIENCY, H100_OVERHEAD } from '../core/efficiency'
import { analyseInference } from '../core/inference'
import { analyseTraining } from '../core/training'
import { fromHuggingFaceConfig } from '../core/modelspec'
import { PRESETS } from '../data/presets'

const preset = PRESETS.find((p) => p.id.includes('8B'))!
const spec = fromHuggingFaceConfig(preset.config, { id: preset.id, name: preset.label })

const inferenceBase = {
  spec,
  topology: DGX_H100,
  batch: 32,
  promptLength: 2048,
  generateLength: 256,
  weightDtype: 'bf16' as const,
  kvDtype: 'bf16' as const,
  activationDtype: 'bf16' as const,
  efficiency: H100_EFFICIENCY,
  overhead: H100_OVERHEAD,
  cudaGraph: true,
  flashAttention: true,
}

const sum = (xs: readonly { seconds: number }[]) => xs.reduce((a, b) => a + b.seconds, 0)

describe('inference accounting', () => {
  test.each([
    ['no pipeline', { tensor: 8, pipeline: 1, data: 1 }],
    ['pipelined', { tensor: 1, pipeline: 8, data: 1 }],
    ['mixed', { tensor: 2, pipeline: 2, data: 2 }],
  ])('slices sum to the reported step time — %s', (_name, parallelism) => {
    const result = analyseInference({ ...inferenceBase, parallelism })
    const account = accountInference(result, parallelism.pipeline)!

    expect(account.totalSeconds).toBe(result.timePerOutputToken)
    expect(sum(account.slices) / account.totalSeconds).toBeCloseTo(1, 6)
  })

  test('the winner is a whole class, not the largest single slice', () => {
    // Pipelining splits network time across two slices — the per-layer
    // collectives and the stage-to-stage hops. Ranking slices rather than
    // classes let a smaller compute slice beat a larger network total.
    const parallelism = { tensor: 2, pipeline: 4, data: 1 }
    const result = analyseInference({ ...inferenceBase, parallelism })
    const account = accountInference(result, parallelism.pipeline)!

    const CLASS_OF: Record<string, string> = {
      compute: 'compute',
      memory: 'memory-bandwidth',
      'network-bandwidth': 'network-bandwidth',
      'network-latency': 'network-latency',
      overhead: 'overhead',
      stall: 'serialization',
      bubble: 'serialization',
      hops: 'network-bandwidth',
      'grad-allreduce': 'network-bandwidth',
    }

    const ownClass = account.slices.filter(
      (s) => CLASS_OF[s.key] === account.dominant.bottleneck,
    )
    expect(account.dominant.seconds).toBeCloseTo(sum(ownClass), 12)

    // And it really is the largest class, not merely a class.
    const byClass = new Map<string, number>()
    for (const slice of account.slices) {
      const key = CLASS_OF[slice.key]!
      byClass.set(key, (byClass.get(key) ?? 0) + slice.seconds)
    }
    expect(account.dominant.seconds).toBe(Math.max(...byClass.values()))
  })

  test('the pipeline factor is applied, not ignored', () => {
    // The bug this replaces: the stage breakdown was divided by the whole
    // traversal, so eight stages reported an eighth of the real share.
    const parallelism = { tensor: 1, pipeline: 8, data: 1 }
    const result = analyseInference({ ...inferenceBase, parallelism })
    const account = accountInference(result, 8)!

    const naive = result.decode!.schedule.breakdown.memory / result.timePerOutputToken!
    const honest = account.slices.find((s) => s.key === 'memory')!.seconds / account.totalSeconds

    expect(honest / naive).toBeCloseTo(8, 3)
    expect(honest).toBeGreaterThan(0.5)
  })
})

describe('training accounting', () => {
  const trainingBase = {
    spec,
    topology: DGX_H100,
    microBatch: 1,
    sequenceLength: 4096,
    weightDtype: 'bf16' as const,
    activationDtype: 'bf16' as const,
    optimizer: 'adam-mixed' as const,
    zeroStage: 3 as const,
    sequenceParallel: true,
    selectiveRecompute: true,
    fullRecompute: false,
    efficiency: H100_EFFICIENCY,
    overhead: H100_OVERHEAD,
  }

  test.each([
    ['data parallel', { tensor: 1, pipeline: 1, data: 8 }, 16],
    ['tensor parallel', { tensor: 8, pipeline: 1, data: 1 }, 16],
    ['pipelined', { tensor: 1, pipeline: 4, data: 2 }, 8],
  ])('slices sum to the reported step time — %s', (_name, parallelism, accumulation) => {
    const result = analyseTraining({
      ...trainingBase,
      parallelism,
      gradientAccumulation: accumulation,
    })
    if (!result.feasible) return

    const account = accountTraining(result, accumulation)!
    expect(account.totalSeconds).toBe(result.stepTime)
    expect(sum(account.slices) / account.totalSeconds).toBeCloseTo(1, 6)
  })

  test('the bubble is charged to the step, and can win it', () => {
    // Sixteen stages against two microbatches is nearly all bubble. The
    // microbatch scheduler cannot see that: it only ever schedules work.
    const result = analyseTraining({
      ...trainingBase,
      parallelism: { tensor: 1, pipeline: 8, data: 1 },
      gradientAccumulation: 1,
    })
    if (!result.feasible) return

    const account = accountTraining(result, 1)!
    const bubble = account.slices.find((s) => s.key === 'bubble')
    expect(bubble).toBeDefined()
    expect(bubble!.seconds).toBeGreaterThan(0)
    expect(sum(account.slices) / account.totalSeconds).toBeCloseTo(1, 6)
  })
})
