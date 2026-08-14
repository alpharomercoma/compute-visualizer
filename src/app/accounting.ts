/**
 * One step, accounted for exactly once.
 *
 * The interface previously used two different denominators on the same screen.
 * The verdict divided `schedule.breakdown` by the reported step time; the
 * stacked bar divided the same breakdown by its own sum. Those agree in exactly
 * one configuration — inference with no pipeline parallelism — and silently
 * disagree everywhere else, because `schedule` is not a step:
 *
 *   - in inference it covers ONE PIPELINE STAGE, and a token crosses `p` of them
 *     plus `p-1` hops, so with PP 8 the verdict reported an eighth of the truth;
 *   - in training it covers ONE MICROBATCH, and a step runs `m` of them plus the
 *     pipeline bubble and the exposed gradient reduction, so with the default 16
 *     microbatches the verdict reported "memory bandwidth is the limit" beside a
 *     share of 5%.
 *
 * A verdict that contradicts its own evidence is worse than no verdict. So the
 * scaling happens once, here, and everything downstream — the headline share,
 * the bar, the timeline, the sweep — reads the same slices.
 *
 * The extra terms are derived as RESIDUALS against the step time the core
 * already reported, never by re-deriving core's formulae. That makes the
 * reconciliation exact by construction rather than by agreement between two
 * copies of the same algebra, which is the kind of thing that drifts.
 */

import type { InferenceResult } from '../core/inference'
import type { BottleneckClass, LimitedBy, ScheduleResult } from '../core/schedule'
import type { TrainingResult } from '../core/training'

/** A named span of the step. Slices sum to `totalSeconds`, always. */
export interface Slice {
  readonly key: string
  readonly label: string
  readonly color: string
  readonly seconds: number
  /** Which of the five limits this span is charged to, where it has one. */
  readonly limit?: LimitedBy | 'stall'
}

/** Slices sharing one limit, added up. What the verdict is actually about. */
export interface ClassTotal {
  readonly bottleneck: BottleneckClass
  readonly label: string
  readonly seconds: number
}

export interface StepAccounting {
  /** The step time actually reported to the user. */
  readonly totalSeconds: number
  /** Every slice, largest first. Sums to `totalSeconds` to within rounding. */
  readonly slices: readonly Slice[]
  /** The limiting CLASS, not the largest single slice. */
  readonly dominant: ClassTotal
  /** The runner-up class, which is what makes a balanced step legible. */
  readonly runnerUp?: ClassTotal
  /** The class named by the verdict, refined by anything the scheduler cannot see. */
  readonly bottleneck: BottleneckClass
  /**
   * True when the largest span is real work and still holds under half the step.
   *
   * This is a genuinely different situation from any of the six classes, and
   * it is the one people are trying to reach: nothing is starving, so there is
   * no single thing to fix. Reporting it as "the schedule is the limit" — which
   * the classifier's under-half rule does, because that rule was written for a
   * critical path where waiting was the only way to get there — sends someone
   * off to restructure a step that is already balanced.
   */
  readonly balanced: boolean
  /** What one unit of the repeated work is, for the timeline's caption. */
  readonly repeat: { readonly count: number; readonly unit: string; readonly seconds: number }
}

/**
 * What each class is called in a sentence.
 *
 * One vocabulary, used by the verdict, the sweep and every tooltip, so the same
 * limit is never called three different things on one page.
 */
export const BOTTLENECK_NAME: Record<BottleneckClass, string> = {
  compute: 'the tensor cores',
  'memory-bandwidth': 'memory bandwidth',
  'network-bandwidth': 'interconnect bandwidth',
  'network-latency': 'collective startup',
  overhead: 'kernel launch',
  serialization: 'the schedule',
}

const KERNEL_SLICES: ReadonlyArray<{ key: LimitedBy; label: string; color: string }> = [
  { key: 'compute', label: 'Compute', color: 'var(--c-compute)' },
  { key: 'memory', label: 'Memory', color: 'var(--c-memory)' },
  { key: 'network-bandwidth', label: 'Collectives', color: 'var(--c-net-bw)' },
  { key: 'network-latency', label: 'Collective startup', color: 'var(--c-net-lat)' },
  { key: 'overhead', label: 'Kernel launch', color: 'var(--c-overhead)' },
]

/**
 * The class a slice implies.
 *
 * `schedule.classify` names a class from the kernel-level breakdown alone,
 * which is right for the work it can see and blind to the terms added above it:
 * a pipeline bubble is idle time nobody scheduled, and it can be the largest
 * thing in a training step. Naming it 'serialization' — everything waiting its
 * turn — is what that class already means.
 */
const CLASS_OF: Record<string, BottleneckClass> = {
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

function scaled(schedule: ScheduleResult, factor: number): Slice[] {
  const out: Slice[] = KERNEL_SLICES.map((s) => ({
    key: s.key,
    label: s.label,
    color: s.color,
    seconds: schedule.breakdown[s.key] * factor,
    limit: s.key,
  }))

  if (schedule.stallTime > 0) {
    out.push({
      key: 'stall',
      label: 'Waiting',
      color: 'var(--c-serial)',
      seconds: schedule.stallTime * factor,
      limit: 'stall',
    })
  }

  return out
}

/** What a whole class of spans is called when it wins. */
const CLASS_LABEL: Record<BottleneckClass, string> = {
  compute: 'Compute',
  'memory-bandwidth': 'Memory',
  'network-bandwidth': 'Interconnect',
  'network-latency': 'Collective startup',
  overhead: 'Kernel launch',
  serialization: 'Waiting',
}

function finish(
  slices: Slice[],
  totalSeconds: number,
  repeat: StepAccounting['repeat'],
): StepAccounting {
  const kept = slices.filter((s) => s.seconds > 0).sort((a, b) => b.seconds - a.seconds)

  /**
   * Totalled by CLASS before anything is called the winner.
   *
   * Ranking the individual slices instead reports the largest of them, which is
   * not the same question. A step that is 30% collectives, 30% stage-to-stage
   * transfer and 40% arithmetic would have been announced as compute-bound at
   * 40% while the interconnect held 60% of it — the residual terms this module
   * exists to account for were being counted separately and then compared as
   * though they were rivals.
   */
  const totals = new Map<BottleneckClass, number>()
  for (const slice of kept) {
    const bottleneck = CLASS_OF[slice.key] ?? 'serialization'
    totals.set(bottleneck, (totals.get(bottleneck) ?? 0) + slice.seconds)
  }

  const ranked: ClassTotal[] = [...totals.entries()]
    .map(([bottleneck, seconds]) => ({ bottleneck, label: CLASS_LABEL[bottleneck], seconds }))
    .sort((a, b) => b.seconds - a.seconds)

  const dominant = ranked[0] ?? {
    bottleneck: 'serialization' as const,
    label: CLASS_LABEL.serialization,
    seconds: 0,
  }
  const runnerUp = ranked[1]

  const share = totalSeconds > 0 ? dominant.seconds / totalSeconds : 0
  const waiting = dominant.bottleneck === 'serialization'

  return {
    totalSeconds,
    slices: kept,
    dominant,
    ...(runnerUp ? { runnerUp } : {}),
    bottleneck: dominant.bottleneck,
    balanced: !waiting && share < 0.5,
    repeat,
  }
}

/**
 * Inference: the schedule covers one pipeline stage of one token.
 *
 * `timePerOutputToken` is `stage x p + hop x (p-1)`, so scaling the stage by `p`
 * and calling the remainder "pipeline transfer" reconciles exactly without
 * restating the hop formula.
 */
export function accountInference(
  result: InferenceResult,
  pipelineParallel: number,
): StepAccounting | undefined {
  const decode = result.decode
  const total = result.timePerOutputToken
  if (!decode || total === undefined) return undefined

  const slices = scaled(decode.schedule, pipelineParallel)

  const hops = total - decode.schedule.makespan * pipelineParallel
  if (hops > 0) {
    slices.push({
      key: 'hops',
      label: 'Stage-to-stage transfer',
      color: 'var(--c-net-bw)',
      seconds: hops,
      limit: 'network-bandwidth',
    })
  }

  return finish(slices, total, {
    count: pipelineParallel,
    unit: pipelineParallel > 1 ? 'pipeline stages' : 'pass',
    seconds: decode.schedule.makespan,
  })
}

/**
 * Training: the schedule covers one microbatch's forward and backward.
 *
 * A step is `m` of those, plus the pipeline bubble, plus whatever of the
 * gradient all-reduce outlived the last backward pass. Both extras are already
 * reported by the core, and the bubble is taken as the residual so the slices
 * cannot drift from the step time the user is shown.
 */
export function accountTraining(
  result: TrainingResult,
  microbatches: number,
): StepAccounting | undefined {
  const total = result.stepTime
  if (total === undefined || !result.schedule) return undefined

  const slices = scaled(result.schedule, microbatches)

  const exposed = result.exposedCommunication ?? 0
  if (exposed > 0) {
    slices.push({
      key: 'grad-allreduce',
      label: 'Exposed gradient all-reduce',
      color: 'var(--c-net-bw)',
      seconds: exposed,
      limit: 'network-bandwidth',
    })
  }

  const bubble = total - result.schedule.makespan * microbatches - exposed
  if (bubble > 1e-12) {
    slices.push({
      key: 'bubble',
      label: 'Pipeline bubble',
      color: 'var(--c-serial)',
      seconds: bubble,
      limit: 'stall',
    })
  }

  return finish(slices, total, {
    count: microbatches,
    unit: microbatches > 1 ? 'microbatches' : 'microbatch',
    seconds: result.schedule.makespan,
  })
}
