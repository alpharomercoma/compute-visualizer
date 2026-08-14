/**
 * The answer as a function, not a point.
 *
 * Every other calculator in this space answers for one configuration. The model
 * here is a pure function of its inputs and costs about two milliseconds to
 * evaluate, so it can be run across a whole axis and the shape of the answer
 * shown directly: where the limit changes class, and where the configuration
 * stops fitting at all.
 *
 * That shape is the thing worth knowing. "Memory-bandwidth-bound at batch 32" is
 * a fact about today. "Memory-bandwidth-bound at every batch that fits, because
 * the crossover is at 296 and memory runs out at 44" is a fact about the
 * hardware, and it closes off a whole line of wasted effort.
 *
 * `core/search.ts` already binary-searches for a single crossover. This
 * generalises it: the sweep finds every boundary on the axis, including the
 * capacity wall, which a search over the bottleneck class alone cannot see.
 */

import type { AppState } from './state'
import type { Topology } from '../core/hardware'
import type { TransformerSpec } from '../core/modelspec'
import { H100_EFFICIENCY, H100_OVERHEAD } from '../core/efficiency'
import { analyseInference } from '../core/inference'
import { analyseTraining } from '../core/training'
import type { BottleneckClass } from '../core/schedule'
import type { CollectiveProfile } from '../core/collectives'
import { accountInference, accountTraining } from './accounting'

export type AxisKey = 'batch' | 'promptLength' | 'sequenceLength' | 'gradientAccumulation'

export interface Axis {
  readonly key: AxisKey
  /** The control's own name, not the field's. */
  readonly label: string
  readonly min: number
  readonly max: number
  /** Values must land on this grid to be legal configurations. */
  readonly step: number
  readonly format: (value: number) => string
}

export const AXES: Record<AxisKey, Axis> = {
  batch: {
    key: 'batch',
    label: 'Batch size',
    min: 1,
    max: 512,
    step: 1,
    format: (v) => String(v),
  },
  promptLength: {
    key: 'promptLength',
    label: 'Context length',
    min: 128,
    max: 32768,
    step: 128,
    format: (v) => (v >= 1024 ? `${Math.round(v / 1024)}k` : String(v)),
  },
  sequenceLength: {
    key: 'sequenceLength',
    label: 'Sequence length',
    min: 512,
    max: 32768,
    step: 512,
    format: (v) => (v >= 1024 ? `${Math.round(v / 1024)}k` : String(v)),
  },
  gradientAccumulation: {
    key: 'gradientAccumulation',
    label: 'Microbatches',
    min: 1,
    max: 256,
    step: 1,
    format: (v) => String(v),
  },
}

export const AXES_FOR = (phase: AppState['phase']): readonly Axis[] =>
  phase === 'inference'
    ? [AXES.batch, AXES.promptLength]
    : [AXES.sequenceLength, AXES.gradientAccumulation]

/**
 * How the answer reads at one point on the axis.
 *
 * 'balanced' is its own state rather than a shade of the dominant class,
 * because it is the state everything else on the page is trying to reach: the
 * largest span is real work and still under half the step, so there is nothing
 * left that is obviously worth fixing. Folding it into the dominant class made
 * the strip a single flat colour and hid the only transition on the axis.
 */
export type SweepState = BottleneckClass | 'balanced' | 'infeasible'

export interface SweepPoint {
  readonly value: number
  readonly feasible: boolean
  readonly state: SweepState
  /** The dominant class even where nothing dominates, for the fill and tooltip. */
  readonly bottleneck?: BottleneckClass
  /** Share of the step held by the largest span, 0..1. */
  readonly share?: number
  readonly tokensPerSecond?: number
}

export interface Boundary {
  /** The first value on the axis where the answer reads differently. */
  readonly at: number
  readonly from: SweepState
  readonly to: SweepState
}

export interface SweepResult {
  readonly axis: Axis
  readonly points: readonly SweepPoint[]
  readonly boundaries: readonly Boundary[]
  /** The largest value on the axis that still fits, if the wall is on-axis. */
  readonly capacityLimit?: number
}

/**
 * Log-spaced sample values, snapped to the axis grid and deduplicated.
 *
 * Log spacing because the interesting structure in every one of these axes is
 * multiplicative — the crossover sits at a ratio of machine constants, not at an
 * offset — and a linear sweep of batch 1..512 spends most of its samples in the
 * region where nothing changes.
 */
export function sampleValues(axis: Axis, count = 40): number[] {
  const lo = Math.log(axis.min)
  const hi = Math.log(axis.max)
  const seen = new Set<number>()

  for (let i = 0; i < count; i++) {
    const raw = Math.exp(lo + ((hi - lo) * i) / (count - 1))
    const snapped = Math.max(axis.min, Math.round(raw / axis.step) * axis.step)
    if (snapped <= axis.max) seen.add(snapped)
  }

  return [...seen].sort((a, b) => a - b)
}

function evaluate(
  state: AppState,
  spec: TransformerSpec,
  topology: Topology,
  profiles?: readonly CollectiveProfile[],
): SweepPoint | undefined {
  const parallelism = {
    tensor: state.tensorParallel,
    pipeline: state.pipelineParallel,
    data: state.dataParallel,
  }

  try {
    if (state.phase === 'inference') {
      const result = analyseInference({
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
        ...(profiles ? { collectiveProfiles: profiles } : {}),
      })
      if (!result.feasible) return { value: 0, feasible: false, state: 'infeasible' }

      const account = accountInference(result, parallelism.pipeline)
      if (!account) return { value: 0, feasible: false, state: 'infeasible' }
      return {
        value: 0,
        feasible: true,
        state: account.balanced ? 'balanced' : account.bottleneck,
        bottleneck: account.bottleneck,
        share: account.dominant.seconds / account.totalSeconds,
        tokensPerSecond: result.aggregateTokensPerSecond ?? 0,
      }
    }

    const result = analyseTraining({
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
      ...(profiles ? { collectiveProfiles: profiles } : {}),
    })
    if (!result.feasible) return { value: 0, feasible: false, state: 'infeasible' }

    const account = accountTraining(result, state.gradientAccumulation)
    if (!account) return { value: 0, feasible: false, state: 'infeasible' }
    return {
      value: 0,
      feasible: true,
      state: account.balanced ? 'balanced' : account.bottleneck,
      bottleneck: account.bottleneck,
      share: account.dominant.seconds / account.totalSeconds,
      tokensPerSecond: result.tokensPerSecond ?? 0,
    }
  } catch {
    return undefined
  }
}

export function sweep(
  state: AppState,
  spec: TransformerSpec,
  topology: Topology,
  axis: Axis,
  /**
   * Measured collective profiles, when the user has supplied them.
   *
   * Omitting these made the strip answer a different question from the verdict
   * above it: calibrating against your own nccl-tests output moved the headline
   * and left every threshold on the sweep computed from the shipped priors.
   */
  profiles?: readonly CollectiveProfile[],
): SweepResult {
  const points: SweepPoint[] = []

  for (const value of sampleValues(axis)) {
    const point = evaluate({ ...state, [axis.key]: value }, spec, topology, profiles)
    if (point) points.push({ ...point, value })
  }

  /**
   * Boundaries bisected, not merely sampled.
   *
   * Forty log-spaced samples over 1..512 sit about 17% apart, so reporting the
   * first CHANGED sample as the threshold can be a long way out — and the
   * verdict quotes that number as something to go and set. Bisecting between the
   * bracketing samples costs a handful of extra evaluations and makes "raise the
   * batch to 296" mean 296.
   */
  const boundaries: Boundary[] = []
  for (let i = 1; i < points.length; i++) {
    const before = points[i - 1]!
    const after = points[i]!
    if (before.state === after.state) continue

    let low = before.value
    let high = after.value
    // Tracked rather than assumed: a third state can sit between the two
    // samples, and the boundary belongs to whatever is actually on the far side
    // of it.
    let highState = after.state

    while (high - low > axis.step) {
      const mid = Math.round((low + high) / 2 / axis.step) * axis.step
      if (mid <= low || mid >= high) break
      const probe = evaluate({ ...state, [axis.key]: mid }, spec, topology, profiles)
      if (probe && probe.state === before.state) {
        low = mid
      } else {
        high = mid
        if (probe) highState = probe.state
      }
    }

    boundaries.push({ at: high, from: before.state, to: highState })
  }

  // Where the wall is, if it is on this axis at all. Reported separately from
  // the class boundaries because "you run out of memory first" is the answer to
  // a different question than "what is the limit". Taken from the bisected
  // boundary, so it is the last value that fits and not the last one sampled.
  const wall = boundaries.find((b) => b.to === 'infeasible')
  const capacityLimit = wall ? Math.max(axis.min, wall.at - axis.step) : undefined

  return { axis, points, boundaries, ...(capacityLimit !== undefined ? { capacityLimit } : {}) }
}

/**
 * A signature that ignores the swept field.
 *
 * Dragging the marker changes only that field, so keying the memo on this lets
 * the sweep survive the drag untouched — the curve does not depend on where you
 * are standing on it.
 */
export function sweepKey(state: AppState, axis: Axis, modelId: string, topologyId: string): string {
  const { [axis.key]: _swept, ...rest } = state as unknown as Record<string, unknown>
  return JSON.stringify([rest, axis.key, modelId, topologyId])
}
