/**
 * The rescue makes one promise — press this and it runs — so that is what gets
 * tested. Both scenarios below are two clicks from the front page, and both
 * used to produce a dead end with no controls on it.
 */

import { describe, expect, test } from 'vitest'
import { nearestFeasible } from './rescue'
import { DEFAULT_STATE, type AppState } from './state'
import { DGX_H100, SINGLE_H100 } from '../core/hardware'
import { H100_EFFICIENCY, H100_OVERHEAD } from '../core/efficiency'
import { analyseInference } from '../core/inference'
import { analyseTraining } from '../core/training'
import { fromHuggingFaceConfig } from '../core/modelspec'
import { PRESETS } from '../data/presets'

const preset = PRESETS.find((p) => p.id.includes('70B'))!
const spec = fromHuggingFaceConfig(preset.config, { id: preset.id, name: preset.label })

/** Runs the offered configuration exactly as the page would. */
function fits(state: AppState, topology = DGX_H100): boolean {
  const parallelism = {
    tensor: state.tensorParallel,
    pipeline: state.pipelineParallel,
    data: state.dataParallel,
  }
  const result =
    state.phase === 'inference'
      ? analyseInference({
          spec,
          topology,
          parallelism,
          batch: state.batch,
          promptLength: state.promptLength,
          generateLength: state.generateLength,
          weightDtype: state.weightDtype,
          kvDtype: state.kvDtype,
          activationDtype: 'bf16',
          efficiency: H100_EFFICIENCY,
          overhead: H100_OVERHEAD,
          cudaGraph: state.cudaGraph,
          flashAttention: state.flashAttention,
        })
      : analyseTraining({
          spec,
          topology,
          parallelism,
          microBatch: 1,
          sequenceLength: state.sequenceLength,
          gradientAccumulation: state.gradientAccumulation,
          weightDtype: state.weightDtype,
          activationDtype: 'bf16',
          optimizer: state.optimizer,
          zeroStage: state.zeroStage,
          sequenceParallel: state.sequenceParallel,
          selectiveRecompute: state.recompute === 'selective',
          fullRecompute: state.recompute === 'full',
          efficiency: H100_EFFICIENCY,
          overhead: H100_OVERHEAD,
        })
  return result.feasible
}

describe('rescue', () => {
  test('a 70B on one H100 gets an offer that actually fits', () => {
    const stuck: AppState = {
      ...DEFAULT_STATE,
      topologyId: 'h100-x1',
      tensorParallel: 1,
    }
    expect(fits(stuck, SINGLE_H100)).toBe(false)

    const rescues = nearestFeasible(stuck, SINGLE_H100, spec)
    expect(rescues.length).toBeGreaterThan(0)
    for (const rescue of rescues) {
      expect(fits({ ...stuck, ...rescue.patch }, SINGLE_H100)).toBe(true)
      expect(rescue.usedBytes).toBeLessThanOrEqual(rescue.availableBytes)
    }
  })

  test('training a 70B on one node falls back to naming a model that fits', () => {
    // No configuration rescues this one. Mixed-precision Adam is 16 bytes per
    // parameter — 141 GiB per GPU across eight however it is sharded, and still
    // 84 GiB with an 8-bit optimizer and full recomputation against 79.6
    // available. This is the case that dead-ended the front page's own Training
    // button, and the only honest fix is a different model.
    const stuck: AppState = { ...DEFAULT_STATE, phase: 'training' }
    expect(fits(stuck)).toBe(false)

    const rescues = nearestFeasible(stuck, DGX_H100, spec)
    expect(rescues.length).toBeGreaterThan(0)
    expect(rescues.every((r) => r.patch.modelId !== undefined)).toBe(true)
    for (const rescue of rescues) {
      expect(rescue.usedBytes).toBeLessThanOrEqual(rescue.availableBytes)
    }
  })

  test('a rescue is never a superset of a cheaper one already offered', () => {
    const stuck: AppState = { ...DEFAULT_STATE, batch: 256, promptLength: 16384 }
    const rescues = nearestFeasible(stuck, DGX_H100, spec)
    expect(rescues.length).toBeGreaterThan(1)

    for (let i = 1; i < rescues.length; i++) {
      for (let j = 0; j < i; j++) {
        const cheaper = Object.entries(rescues[j]!.patch)
        const covered = cheaper.every(
          ([key, value]) => rescues[i]!.patch[key as keyof AppState] === value,
        )
        expect(covered).toBe(false)
      }
    }
  })
})
