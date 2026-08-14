/**
 * The stacked time breakdown — the summary above the timeline.
 *
 * This, not the roofline, answers the user's question. The roofline says where a
 * kernel sits relative to a machine balance; this says where the step time went,
 * and it can represent communication honestly, which a two-axis roofline cannot.
 *
 * Five classes, not three. "Communication-bound" collapses two different
 * problems with two different fixes: a bandwidth-bound collective wants a faster
 * fabric, a latency-bound one wants fewer collectives and will not notice the
 * fabric at all.
 *
 * The slices arrive already reconciled to the reported step time. This component
 * used to normalise them against their own sum, which quietly made the bar show
 * 100% of a microbatch while the verdict above it showed a share of the step —
 * two denominators, one screen.
 */

import type { Slice } from '../app/accounting'

export interface TimeBreakdownProps {
  readonly slices: readonly Slice[]
  readonly totalSeconds: number
}

export function TimeBreakdown({ slices, totalSeconds }: TimeBreakdownProps) {
  const total = totalSeconds || 1

  // preserveAspectRatio="none" plus a fixed CSS height: the bar is a length
  // encoding, so horizontal stretch is exactly right and vertical letterboxing —
  // which rendered this at 57% of its height on desktop and 36% on a phone — is
  // exactly wrong. Safe here only because the bar carries no text.
  const height = 28
  const width = 1000
  // A 2px surface gap keeps adjacent fills from reading as one block.
  const gap = 3

  let cursor = 0

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        style={{ height }}
        role="img"
        aria-label={`Step time: ${slices
          .map((s) => `${s.label} ${((s.seconds / total) * 100).toFixed(0)}%`)
          .join(', ')}`}
      >
        <title>Where the step time went</title>
        {slices.map((slice) => {
          const w = Math.max(0, (slice.seconds / total) * width - gap)
          const x = cursor
          cursor += (slice.seconds / total) * width
          return (
            <rect key={slice.key} x={x} y={0} width={w} height={height} rx={2} fill={slice.color}>
              <title>
                {slice.label}: {(slice.seconds * 1e3).toFixed(3)} ms (
                {((slice.seconds / total) * 100).toFixed(1)}%)
              </title>
            </rect>
          )
        })}
      </svg>

      {/* Identity is never colour alone: every slice is named, and its share is
          spelled out for anyone who cannot separate two hues. */}
      <div className="legend">
        {slices.map((slice) => (
          <span key={slice.key}>
            <i style={{ background: slice.color }} aria-hidden="true" />
            {slice.label} {((slice.seconds / total) * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  )
}
