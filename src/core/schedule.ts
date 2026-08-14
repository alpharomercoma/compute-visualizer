/**
 * A list scheduler over a task DAG.
 *
 * No global rule — "step time is max", "step time is sum" — is valid, because
 * whether work overlaps depends on dependency edges and on which resource each
 * task occupies. Rather than hardcode a policy per phase, the model states the
 * dependencies and lets the schedule fall out:
 *
 *   - TP decode: every all-reduce feeds the next GEMM and there is no
 *     independent work to hide it, so collectives land on the critical path and
 *     the answer is effectively a sum.
 *   - DP training: a gradient bucket's all-reduce depends only on its own
 *     backward kernel, so it overlaps the backward of lower layers and only the
 *     tail is exposed.
 *
 * Same scheduler, opposite outcomes, neither hardcoded.
 */

import { type Seconds, seconds } from './units'

/**
 * What a task occupies while it runs.
 *
 * Compute and memory are one resource, not two: within a single kernel they are
 * genuinely overlapped by the hardware, which is what makes `max(compute,
 * memory)` the right duration for a kernel and the wrong aggregation across
 * kernels.
 */
export type Resource = 'gpu' | 'network'

/** Why a task took as long as it did — the input to bottleneck classification. */
export type LimitedBy = 'compute' | 'memory' | 'network-bandwidth' | 'network-latency' | 'overhead'

export interface Task {
  readonly id: string
  readonly label: string
  readonly deps: readonly string[]
  readonly resource: Resource
  readonly duration: Seconds
  readonly limitedBy: LimitedBy
}

export interface ScheduledTask extends Task {
  readonly start: Seconds
  readonly finish: Seconds
  /** True when this task sat waiting on a busy resource rather than on its deps. */
  readonly stalled: boolean
}

export type BottleneckClass =
  | 'compute'
  | 'memory-bandwidth'
  | 'network-bandwidth'
  | 'network-latency'
  | 'overhead'
  | 'serialization'

export interface ScheduleResult {
  readonly makespan: Seconds
  readonly tasks: readonly ScheduledTask[]
  /** Task ids on the critical path, in order. */
  readonly criticalPath: readonly string[]
  /** Time on the critical path attributable to each limit. */
  readonly breakdown: Readonly<Record<LimitedBy, Seconds>>
  /** Critical-path time spent waiting rather than working. */
  readonly stallTime: Seconds
  readonly bottleneck: BottleneckClass
}

export interface ScheduleOptions {
  /**
   * Slowdown applied to a kernel that runs while a collective is in flight.
   *
   * NCCL reductions consume SM and HBM, so crediting a kernel and a concurrent
   * collective both their full standalone rates overstates overlap. The honest
   * default is 0 — not because there is no contention, but because we have no
   * calibrated figure for it, and an invented one is worse than none. Exposed so
   * a user who has measured it can supply theirs.
   */
  readonly contentionPenalty?: number
  /** Internal: network-busy windows discovered by a first pass. */
  readonly _networkWindows?: ReadonlyArray<readonly [number, number]>
}

/**
 * Schedules with contention accounted for, by looking twice.
 *
 * A single pass cannot see a collective that is emitted after the kernel it
 * would slow down, so whether the penalty fired depended on the order tasks
 * happened to be listed in. The first pass finds where the network is busy; the
 * second charges kernels that overlap those windows. With no penalty configured
 * — the default, because we have no calibrated figure — the second pass is
 * skipped entirely.
 */
export function scheduleWithContention(
  tasks: readonly Task[],
  options: ScheduleOptions = {},
): ScheduleResult {
  const penalty = options.contentionPenalty ?? 0
  const first = schedule(tasks, { ...options, contentionPenalty: 0 })
  if (penalty <= 0) return first

  const windows = first.tasks
    .filter((t) => t.resource === 'network')
    .map((t) => [t.start, t.finish] as const)

  return schedule(tasks, { ...options, _networkWindows: windows })
}

/**
 * Schedules tasks at the earliest time their dependencies have finished and
 * their resource is free, then reports the makespan and what drove it.
 *
 * Tasks must be given in an order where dependencies precede dependents.
 */
export function schedule(
  tasks: readonly Task[],
  options: ScheduleOptions = {},
): ScheduleResult {
  const penalty = options.contentionPenalty ?? 0

  const finishTimes = new Map<string, number>()
  const resourceFree: Record<Resource, number> = { gpu: 0, network: 0 }
  /** The task currently holding each resource, so a stall can be attributed. */
  const resourceHeldBy: Record<Resource, string | undefined> = {
    gpu: undefined,
    network: undefined,
  }
  /** For each task, the predecessor whose finish determined its start. */
  const determinedBy = new Map<string, string | undefined>()
  const scheduled: ScheduledTask[] = []

  // Network windows from a previous pass, so a kernel can see a collective that
  // is emitted after it. Populated on the second pass only.
  const networkBusy: ReadonlyArray<readonly [number, number]> = options._networkWindows ?? []

  for (const task of tasks) {
    let depsReady = 0
    let critical: string | undefined

    for (const dep of task.deps) {
      const finish = finishTimes.get(dep)
      if (finish === undefined) {
        throw new Error(`task ${task.id} depends on ${dep}, which has not been scheduled`)
      }
      if (finish > depsReady) {
        depsReady = finish
        critical = dep
      }
    }

    const free = resourceFree[task.resource]
    const start = Math.max(depsReady, free)
    const stalled = free > depsReady

    let duration = task.duration
    if (penalty > 0 && task.resource === 'gpu') {
      const overlaps = networkBusy.some(([from, to]) => start < to && from < start + duration)
      if (overlaps) duration = (duration * (1 + penalty)) as Seconds
    }

    const finish = seconds(start + duration)
    finishTimes.set(task.id, finish)

    // When a task waited on a busy resource, the task that held that resource is
    // its true predecessor on the critical path. Ending the trace there instead
    // would charge all the preceding work to "stall" and misreport a fully
    // utilised pipeline as dependency-bound.
    determinedBy.set(task.id, stalled ? resourceHeldBy[task.resource] : critical)

    resourceFree[task.resource] = finish
    resourceHeldBy[task.resource] = task.id

    scheduled.push({
      ...task,
      duration,
      start: seconds(start),
      finish,
      stalled,
    })
  }

  const makespan = seconds(
    scheduled.reduce((latest, t) => (t.finish > latest ? t.finish : latest), 0),
  )

  const criticalPath = tracePath(scheduled, determinedBy, makespan)
  const onPath = new Set(criticalPath)

  const breakdown: Record<LimitedBy, Seconds> = {
    compute: seconds(0),
    memory: seconds(0),
    'network-bandwidth': seconds(0),
    'network-latency': seconds(0),
    overhead: seconds(0),
  }

  let working = 0
  for (const task of scheduled) {
    if (!onPath.has(task.id)) continue
    breakdown[task.limitedBy] = seconds(breakdown[task.limitedBy] + task.duration)
    working += task.duration
  }
  const stallTime = seconds(Math.max(0, makespan - working))

  return {
    makespan,
    tasks: scheduled,
    criticalPath,
    breakdown,
    stallTime,
    bottleneck: classify(breakdown, stallTime, makespan),
  }
}

/**
 * Walks back from the last task to finish, following whichever predecessor
 * determined each start. Where a task was stalled on a busy resource rather than
 * on a dependency, the chain ends: that gap is stall time, not work.
 */
function tracePath(
  scheduled: readonly ScheduledTask[],
  determinedBy: Map<string, string | undefined>,
  makespan: number,
): string[] {
  const last = scheduled.find((t) => t.finish === makespan)
  if (!last) return []

  const path: string[] = []
  let cursor: string | undefined = last.id
  const guard = new Set<string>()

  while (cursor !== undefined && !guard.has(cursor)) {
    guard.add(cursor)
    path.push(cursor)
    cursor = determinedBy.get(cursor)
  }

  return path.reverse()
}

/**
 * Names the bottleneck.
 *
 * The distinction that earns its keep is network-bandwidth versus
 * network-latency. Collapsing both into "communication-bound" tells a user with
 * a batch-1 tensor-parallel decode to buy faster interconnect, when their
 * 16 KiB messages spend microseconds in startup and nanoseconds on the wire.
 * The fix there is fewer collectives — less tensor parallelism, CUDA graphs —
 * and no amount of bandwidth touches it.
 */
function classify(
  breakdown: Readonly<Record<LimitedBy, Seconds>>,
  _stallTime: Seconds,
  makespan: Seconds,
): BottleneckClass {
  const ranked = (Object.entries(breakdown) as Array<[LimitedBy, Seconds]>).sort(
    (a, b) => b[1] - a[1],
  )
  const top = ranked[0]
  if (!top || top[1] === 0) return 'serialization'

  /**
   * Dependency-bound means nothing is saturated because everything is waiting
   * its turn — so no single resource holds a majority of the critical path.
   *
   * An earlier version tried to detect this from idle time, which an
   * earliest-start scheduler cannot produce: the traced path tiles the makespan
   * by construction, so that test was always false and the class unreachable.
   * A review caught it. This test is about the SHAPE of the critical path
   * instead, which is what the class actually describes.
   */
  const dominantShare = makespan > 0 ? top[1] / makespan : 0
  if (dominantShare < 0.5) return 'serialization'

  switch (top[0]) {
    case 'compute':
      return 'compute'
    case 'memory':
      return 'memory-bandwidth'
    case 'network-bandwidth':
      return 'network-bandwidth'
    case 'network-latency':
      return 'network-latency'
    case 'overhead':
      return 'overhead'
  }
}

/** Human-readable name, and what actually helps. */
export const BOTTLENECK_REMEDY: Record<BottleneckClass, { label: string; remedy: string }> = {
  compute: {
    label: 'Compute-bound',
    remedy: 'Lower precision, better kernels, or more FLOPs.',
  },
  'memory-bandwidth': {
    label: 'Memory-bandwidth-bound',
    remedy: 'Quantize weights or KV, raise batch size, or improve cache reuse.',
  },
  'network-bandwidth': {
    label: 'Network-bandwidth-bound',
    remedy: 'Reduce tensor parallelism, reshard, or move to a faster fabric.',
  },
  'network-latency': {
    label: 'Network-latency-bound',
    remedy:
      'Faster interconnect will not help. Issue fewer collectives: less tensor parallelism, CUDA graph capture, larger batches.',
  },
  overhead: {
    label: 'Launch-overhead-bound',
    remedy: 'Capture the step in a CUDA graph, or fuse kernels.',
  },
  serialization: {
    label: 'Dependency-bound',
    remedy: 'The critical path is idle waiting on ordering. Restructure the schedule.',
  },
}
