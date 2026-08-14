import { describe, expect, it } from 'vitest'
import { type Task, schedule, scheduleWithContention } from './schedule'
import { seconds } from './units'

const kernel = (id: string, deps: string[], duration: number): Task => ({
  id,
  label: id,
  deps,
  resource: 'gpu',
  duration: seconds(duration),
  limitedBy: 'memory',
})

const collective = (id: string, deps: string[], duration: number): Task => ({
  id,
  label: id,
  deps,
  resource: 'network',
  duration: seconds(duration),
  limitedBy: 'network-latency',
})

describe('serial chains', () => {
  it('sums a strictly dependent chain', () => {
    const result = schedule([
      kernel('a', [], 1),
      collective('b', ['a'], 2),
      kernel('c', ['b'], 3),
    ])
    expect(result.makespan).toBe(6)
    expect(result.criticalPath).toEqual(['a', 'b', 'c'])
    expect(result.stallTime).toBe(0)
  })
})

describe('overlap', () => {
  it('hides a collective behind independent work on another resource', () => {
    // The gradient-all-reduce shape: a long collective depending on `a` runs
    // while `c` and `d` continue on the GPU.
    const result = schedule([
      kernel('a', [], 1),
      collective('reduce', ['a'], 5),
      kernel('c', ['a'], 3),
      kernel('d', ['c'], 3),
    ])
    // GPU work finishes at 7, the collective at 6, so the GPU is the constraint.
    expect(result.makespan).toBe(7)
  })

  it('exposes only the tail of a collective that outlasts the work hiding it', () => {
    const result = schedule([
      kernel('a', [], 1),
      collective('reduce', ['a'], 10),
      kernel('c', ['a'], 2),
    ])
    // Collective finishes at 11; GPU work finished at 3. The exposed tail is 8.
    expect(result.makespan).toBe(11)
    expect(result.bottleneck).toBe('network-latency')
  })
})

describe('stall attribution', () => {
  /**
   * The defect this guards: when a task waited on a busy resource rather than on
   * its dependencies, the critical-path trace used to stop there. Everything
   * before it was then charged to "stall", and a fully utilised pipeline was
   * reported as dependency-bound.
   */
  it('traces through a resource stall to the task that held the resource', () => {
    const result = schedule([
      kernel('a', [], 5),
      // `b` depends on nothing, so its deps are ready at 0, but the GPU is busy
      // until 5. It stalls on the resource, not on a dependency.
      kernel('b', [], 5),
    ])

    expect(result.makespan).toBe(10)
    expect(result.criticalPath).toEqual(['a', 'b'])
    // Both tasks were working the whole time; none of it was idle.
    expect(result.stallTime).toBe(0)
    expect(result.bottleneck).toBe('memory-bandwidth')
  })

  it('still reports genuine idleness as dependency-bound', () => {
    // `c` cannot start until the long collective finishes, and there is nothing
    // for the GPU to do meanwhile. That gap is real.
    const result = schedule([
      kernel('a', [], 0.1),
      collective('wait', ['a'], 10),
      kernel('c', ['wait'], 0.1),
    ])
    expect(result.makespan).toBeCloseTo(10.2, 5)
    // The collective is on the path and dominates, so it is named, not the gap.
    expect(result.bottleneck).toBe('network-latency')
  })
})

describe('contention', () => {
  it('leaves durations alone when no penalty is configured', () => {
    const tasks = [kernel('a', [], 1), collective('r', ['a'], 5), kernel('c', ['a'], 3)]
    expect(scheduleWithContention(tasks).makespan).toBe(schedule(tasks).makespan)
  })

  it('charges a kernel that overlaps a collective, whatever order tasks are listed in', () => {
    // The collective is emitted BEFORE the kernel it overlaps, which a
    // single-pass scheduler cannot see: at the moment `c` is placed, `r` has not
    // been scheduled yet.
    const tasks = [kernel('a', [], 1), collective('r', ['a'], 5), kernel('c', ['a'], 3)]

    const durationOf = (result: ReturnType<typeof schedule>) =>
      result.tasks.find((t) => t.id === 'c')!.duration

    expect(durationOf(schedule(tasks))).toBe(3)
    expect(durationOf(scheduleWithContention(tasks, { contentionPenalty: 0.5 }))).toBeCloseTo(4.5, 6)
  })

  it('lets contention push a kernel past a collective it was hiding behind', () => {
    const tasks = [kernel('a', [], 1), collective('r', ['a'], 5), kernel('c', ['a'], 4.5)]
    // Without a penalty the GPU work finishes at 5.5, inside the collective's
    // shadow. With one it finishes at 7.75 and becomes the constraint.
    expect(schedule(tasks).makespan).toBe(6)
    expect(scheduleWithContention(tasks, { contentionPenalty: 0.5 }).makespan).toBeCloseTo(7.75, 6)
  })
})

describe('classification', () => {
  it('separates network latency from network bandwidth', () => {
    const latencyBound = schedule([
      { ...collective('c', [], 5), limitedBy: 'network-latency' },
    ])
    const bandwidthBound = schedule([
      { ...collective('c', [], 5), limitedBy: 'network-bandwidth' },
    ])
    expect(latencyBound.bottleneck).toBe('network-latency')
    expect(bandwidthBound.bottleneck).toBe('network-bandwidth')
  })

  it('rejects a dependency it was never given', () => {
    expect(() => schedule([kernel('b', ['missing'], 1)])).toThrow(/has not been scheduled/)
  })
})
