/**
 * A link is untrusted input. Every field below is a divisor somewhere in the
 * model, and the parallelism decides how many ways the weights are cut — so a
 * hash that names a split the hardware cannot run does not produce an error, it
 * produces a confident wrong answer.
 */

import { describe, expect, test } from 'vitest'
import { DEFAULT_STATE, decodeState, encodeState, normalise } from './state'

describe('decoding a link', () => {
  test('out-of-range numbers fall back to the default', () => {
    const state = decodeState('#b=0&in=1e12&out=-4&ga=0')
    expect(state.batch).toBe(DEFAULT_STATE.batch)
    expect(state.promptLength).toBe(DEFAULT_STATE.promptLength)
    expect(state.generateLength).toBe(DEFAULT_STATE.generateLength)
    expect(state.gradientAccumulation).toBe(DEFAULT_STATE.gradientAccumulation)
  })

  test('garbage numbers do not become NaN', () => {
    const state = decodeState('#b=abc&in=&tp=null')
    expect(Number.isFinite(state.batch)).toBe(true)
    expect(Number.isFinite(state.promptLength)).toBe(true)
    expect(Number.isFinite(state.tensorParallel)).toBe(true)
  })

  test('a round trip is stable', () => {
    const original = { ...DEFAULT_STATE, batch: 64, weightDtype: 'fp8' as const }
    expect(decodeState(`#${encodeState(original)}`)).toEqual(original)
  })
})

describe('normalising the split', () => {
  test('one GPU cannot be shared eight ways', () => {
    // `#hw=h100-x1` alone used to carry the default tensor parallelism of 8 into
    // a single-GPU topology, and the capacity model divided the weights by eight
    // and reported that a 70B fits.
    const state = normalise({ ...DEFAULT_STATE, topologyId: 'h100-x1' })
    expect(state.tensorParallel).toBe(1)
    expect(state.pipelineParallel).toBe(1)
    expect(state.dataParallel).toBe(1)
  })

  test('a decoded link is normalised too', () => {
    const state = decodeState('#hw=h100-x1&tp=8&pp=4&dp=2')
    expect(state.tensorParallel * state.pipelineParallel * state.dataParallel).toBe(1)
  })

  test('shrinking to one GPU and back restores tensor parallelism', () => {
    // The round trip must not land on TP 1 · DP 8. Data parallelism replicates
    // the model, so that split is the one that does NOT fit a large model, and
    // arriving there by changing the hardware twice would look like a bug in
    // the memory calculation rather than in the split.
    const shrunk = normalise({ ...DEFAULT_STATE, topologyId: 'h100-x1' })
    const restored = normalise({ ...shrunk, topologyId: 'h100-x8' })
    expect(restored.tensorParallel).toBe(8)
    expect(restored.dataParallel).toBe(1)
  })

  test('unknown enum values in a link are ignored', () => {
    const state = decodeState('#ph=nope&wd=float9&opt=sideways&rc=maybe&hw=h100-x99')
    expect(state.phase).toBe(DEFAULT_STATE.phase)
    expect(state.weightDtype).toBe(DEFAULT_STATE.weightDtype)
    expect(state.optimizer).toBe(DEFAULT_STATE.optimizer)
    expect(state.recompute).toBe(DEFAULT_STATE.recompute)
    expect(state.topologyId).toBe(DEFAULT_STATE.topologyId)
  })

  test('the split always accounts for every GPU', () => {
    for (const [tensor, pipeline, data] of [
      [1, 1, 1],
      [8, 8, 8],
      [3, 1, 1],
      [1, 3, 1],
      [2, 2, 1],
    ]) {
      const state = normalise({
        ...DEFAULT_STATE,
        topologyId: 'h100-x8',
        tensorParallel: tensor!,
        pipelineParallel: pipeline!,
        dataParallel: data!,
      })
      expect(state.tensorParallel * state.pipelineParallel * state.dataParallel).toBe(8)
    }
  })
})
