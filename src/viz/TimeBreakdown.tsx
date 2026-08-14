/**
 * The stacked time breakdown.
 *
 * This, not the roofline, is what answers the user's question. The roofline
 * tells you where a kernel sits relative to a machine balance; this tells you
 * where the step time actually went, and it can represent communication
 * honestly, which a two-axis roofline cannot.
 *
 * Five classes, not three. "Communication-bound" collapses two different
 * problems with two different fixes: a bandwidth-bound collective wants a faster
 * fabric, a latency-bound one wants fewer collectives and will not notice the
 * fabric at all.
 */

import type { LimitedBy, ScheduleResult } from '../core/schedule'
import { toMilliseconds } from '../core/units'

const SEGMENTS: ReadonlyArray<{ key: LimitedBy; label: string; color: string }> = [
  { key: 'compute', label: 'Compute', color: 'var(--c-compute)' },
  { key: 'memory', label: 'HBM', color: 'var(--c-memory)' },
  { key: 'network-bandwidth', label: 'Net bandwidth', color: 'var(--c-net-bw)' },
  { key: 'network-latency', label: 'Net latency', color: 'var(--c-net-lat)' },
  { key: 'overhead', label: 'Launch', color: 'var(--c-overhead)' },
]

export interface TimeBreakdownProps {
  readonly schedule: ScheduleResult
  readonly totalSeconds: number
}

export function TimeBreakdown({ schedule, totalSeconds }: TimeBreakdownProps) {
  const parts = SEGMENTS.map((segment) => ({
    ...segment,
    seconds: schedule.breakdown[segment.key],
  })).filter((part) => part.seconds > 0)

  const stalled = schedule.stallTime
  const all = stalled > 0 ? [...parts, { key: 'stall' as const, label: 'Waiting', color: 'var(--c-serial)', seconds: stalled }] : parts
  const total = all.reduce((sum, part) => sum + part.seconds, 0) || totalSeconds || 1

  // preserveAspectRatio="none" plus a fixed CSS height: the bar is a length
  // encoding, so horizontal stretch is exactly right and vertical letterboxing
  // — which rendered this at 57% of its height on desktop and 36% on a phone —
  // is exactly wrong.
  const height = 28
  const width = 1000
  // A 2px surface gap keeps adjacent fills from reading as one block.
  const gap = 3

  const dominant = all.reduce((a, b) => (b.seconds > a.seconds ? b : a), all[0]!)
  let cursor = 0

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        style={{ height: 28 }}
        role="img"
        aria-label={`Step time breakdown: ${all
          .map((p) => `${p.label} ${((p.seconds / total) * 100).toFixed(0)}%`)
          .join(', ')}`}
      >
        <title>Where the step time went</title>
        {all.map((part) => {
          const w = Math.max(0, (part.seconds / total) * width - gap)
          const x = cursor
          cursor += (part.seconds / total) * width
          return (
            <rect key={part.key} x={x} y={0} width={w} height={height} rx={2} fill={part.color}>
              <title>
                {part.label}: {toMilliseconds(part.seconds).toFixed(2)} ms (
                {((part.seconds / total) * 100).toFixed(1)}%)
              </title>
            </rect>
          )
        })}
      </svg>

      <p className="small muted" style={{ margin: '0.5rem 0 0' }}>
        <b style={{ color: 'var(--ink)' }}>
          {dominant.label} is {((dominant.seconds / total) * 100).toFixed(0)}%
        </b>{' '}
        of the step.
      </p>

      {/* Identity is never colour alone: every segment is named, and the share
          is spelled out for anyone who cannot separate two hues. */}
      <div className="legend">
        {all.map((part) => (
          <span key={part.key}>
            <i style={{ background: part.color }} aria-hidden="true" />
            {part.label} {((part.seconds / total) * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  )
}
