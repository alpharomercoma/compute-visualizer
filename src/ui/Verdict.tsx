import type { BottleneckClass, ScheduleResult } from '../core/schedule'

/**
 * The answer, in words, owning the page.
 *
 * Two things make this different from the heading it replaces. It states a
 * CAUSE and a LEVER, not just a classification — "memory-bandwidth-bound" tells
 * you nothing to do. And the number beside it is the limiting class's share of
 * the step, which is what proves the diagnosis and tells you whether acting on
 * it is worth anything. Throughput and cost are outcomes; they come after.
 */

export interface VerdictProps {
  readonly bottleneck: BottleneckClass
  readonly schedule: ScheduleResult
  readonly stepSeconds: number
  readonly phase: 'inference' | 'training'
}

interface Reading {
  readonly headline: string
  readonly cause: (share: number) => string
  readonly lever: string
}

/**
 * Plain language, and a lever that names an action.
 *
 * The distinction the whole tool exists for is the two network rows: a
 * bandwidth-bound collective wants a faster fabric, a latency-bound one will
 * not notice the fabric at all and wants fewer collectives. Collapsing them
 * into "communication-bound" sends half these users shopping for the wrong
 * hardware.
 */
const READINGS: Record<BottleneckClass, Reading> = {
  'memory-bandwidth': {
    headline: 'Memory bandwidth is the limit',
    cause: (s) => `${s}% of each step goes to reading weights and KV cache, not to arithmetic.`,
    lever: 'Raise the batch size, quantize the weights, or shrink the KV cache with a smaller dtype.',
  },
  compute: {
    headline: 'The tensor cores are the limit',
    cause: (s) => `${s}% of each step is arithmetic. The GPUs are doing what you bought them for.`,
    lever: 'Drop to a lower precision, or accept this — being compute-bound is the good case.',
  },
  'network-bandwidth': {
    headline: 'Interconnect bandwidth is the limit',
    cause: (s) => `${s}% of each step is collectives moving real bytes between GPUs.`,
    lever: 'Use less tensor parallelism, reshard toward data parallelism, or move to a faster fabric.',
  },
  'network-latency': {
    headline: 'Collective latency is the limit',
    cause: (s) =>
      `${s}% of each step is collective startup cost. The messages are tiny; almost nothing is on the wire.`,
    lever:
      'A faster fabric will not help. Issue fewer collectives: less tensor parallelism, CUDA graph capture, or a larger batch.',
  },
  overhead: {
    headline: 'Kernel launch overhead is the limit',
    cause: (s) => `${s}% of each step is spent launching work rather than doing it.`,
    lever: 'Capture the step in a CUDA graph, or fuse the small kernels.',
  },
  serialization: {
    headline: 'The schedule is the limit',
    cause: (s) =>
      `No single resource holds the step — the largest is only ${s}%. Everything is waiting its turn.`,
    lever: 'Restructure the schedule so independent work can overlap.',
  },
}

const SHARE_KEY: Record<BottleneckClass, keyof ScheduleResult['breakdown'] | undefined> = {
  compute: 'compute',
  'memory-bandwidth': 'memory',
  'network-bandwidth': 'network-bandwidth',
  'network-latency': 'network-latency',
  overhead: 'overhead',
  serialization: undefined,
}

export function Verdict({ bottleneck, schedule, stepSeconds, phase }: VerdictProps) {
  const reading = READINGS[bottleneck]
  const key = SHARE_KEY[bottleneck]

  const seconds = key ? schedule.breakdown[key] : Math.max(...Object.values(schedule.breakdown))
  const share = stepSeconds > 0 ? Math.round((seconds / stepSeconds) * 100) : 0

  return (
    <section className="panel verdict">
      <div>
        <p className="eyebrow" style={{ margin: '0 0 0.5rem' }}>
          {phase === 'inference' ? 'Decode step' : 'Training step'}
        </p>
        <h2>{reading.headline}</h2>
        <p className="cause">{reading.cause(share)}</p>
        <p className="lever">
          <strong>What helps:</strong> {reading.lever}
        </p>
      </div>

      <div className="share">
        <b>{share}%</b>
        <span>of the step</span>
      </div>
    </section>
  )
}
