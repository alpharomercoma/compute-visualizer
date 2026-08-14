/**
 * The hardware, flat.
 *
 * This replaces a 235 KB three.js scene as the default view, and it says more:
 * every tile is labelled with which GPU it is and how full its memory is, which
 * the 3D board never did. A bottleneck diagnosis has no spatial component, so
 * depth was never carrying meaning — it was carrying atmosphere.
 *
 * Pure SVG: no dependency, instant paint, legible at 390px, and it themes by
 * inheriting currentColor and the CSS custom properties rather than needing a
 * second set of materials.
 */

export interface HardwareBoardProps {
  readonly gpus: number
  readonly accelerator: string
  /** Fraction of each GPU's memory in use, 0..1. */
  readonly occupancy: number
  readonly usedBytesLabel: string
  readonly totalBytesLabel: string
  /** Share of the step spent in collectives, 0..1. Drives link emphasis only. */
  readonly linkActivity: number
  readonly accent: string
  readonly infeasible: boolean
}

export function HardwareBoard({
  gpus,
  accelerator,
  occupancy,
  usedBytesLabel,
  totalBytesLabel,
  linkActivity,
  accent,
  infeasible,
}: HardwareBoardProps) {
  const columns = gpus === 1 ? 1 : Math.ceil(gpus / 2)
  const rows = gpus === 1 ? 1 : 2

  const tile = { w: 92, h: 62 }
  const gapX = 14
  const gapY = 44 // room for the switch spine between the rows
  const pad = { x: 8, y: 22 }

  const width = pad.x * 2 + columns * tile.w + (columns - 1) * gapX
  const height = pad.y * 2 + rows * tile.h + (rows - 1) * gapY

  const x = (i: number) => pad.x + (i % columns) * (tile.w + gapX)
  const y = (i: number) => pad.y + Math.floor(i / columns) * (tile.h + gapY)

  const clamped = Math.max(0, Math.min(1, occupancy))
  const fillHeight = Math.max(clamped > 0 ? 2 : 0, clamped * (tile.h - 20))
  const tight = clamped > 0.9 || infeasible

  const spineY = rows > 1 ? pad.y + tile.h + gapY / 2 : height - pad.y / 2

  return (
    <figure style={{ margin: 0 }}>
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${gpus} ${accelerator}, each using ${usedBytesLabel} of ${totalBytesLabel} memory. Collectives take ${Math.round(
          linkActivity * 100,
        )} percent of the step.`}
        style={{ maxHeight: 260 }}
      >
        {/* The switch spine: one line, not 28 crossing all-to-all edges. Every
            GPU reaching every other through the switch is the property that
            matters, and a hairball obscured it. */}
        {gpus > 1 && (
          <>
            <line
              x1={pad.x + tile.w / 2}
              y1={spineY}
              x2={width - pad.x - tile.w / 2}
              y2={spineY}
              stroke={accent}
              strokeWidth={1 + linkActivity * 5}
              strokeLinecap="round"
              opacity={0.25 + linkActivity * 0.6}
            />
            {Array.from({ length: gpus }, (_, i) => (
              <line
                key={`drop${i}`}
                x1={x(i) + tile.w / 2}
                y1={i < columns ? y(i) + tile.h : y(i)}
                x2={x(i) + tile.w / 2}
                y2={spineY}
                stroke={accent}
                strokeWidth={1 + linkActivity * 2.5}
                opacity={0.25 + linkActivity * 0.5}
              />
            ))}
            {/* Just the fabric. A share-of-step figure is a timing metric and
                does not belong in a capacity diagram. */}
            <text
              x={width - pad.x}
              y={spineY - 6}
              textAnchor="end"
              style={{ fill: 'var(--ink-2)' }}
            >
              NVSwitch
            </text>
          </>
        )}

        {Array.from({ length: gpus }, (_, i) => (
          <g key={i}>
            <rect
              x={x(i)}
              y={y(i)}
              width={tile.w}
              height={tile.h}
              rx={5}
              fill="var(--panel-sunk)"
              stroke={tight ? 'var(--c-net-lat)' : 'var(--control-border)'}
            />
            {/* Memory fill, growing from the base like a gauge. */}
            {fillHeight > 0 && (
              <rect
                x={x(i) + 6}
                y={y(i) + tile.h - 7 - fillHeight}
                width={tile.w - 12}
                height={fillHeight}
                rx={2}
                fill={accent}
                opacity={0.85}
              />
            )}
            <text x={x(i) + 7} y={y(i) + 15} style={{ fill: 'var(--ink-2)' }}>
              GPU {i}
            </text>
            <text
              x={x(i) + tile.w - 7}
              y={y(i) + 15}
              textAnchor="end"
              style={{ fill: 'var(--ink)', fontFamily: 'var(--mono)' }}
            >
              {Math.round(clamped * 100)}%
            </text>
          </g>
        ))}
      </svg>

      <figcaption className="small muted" style={{ marginTop: '0.5rem' }}>
        {gpus === 1 ? 'One' : gpus} × {accelerator}. Each holds{' '}
        <b className="num" style={{ color: 'var(--ink)' }}>
          {usedBytesLabel}
        </b>{' '}
        of {totalBytesLabel}.
        {tight && ' At this occupancy allocator fragmentation may still defeat it.'}
      </figcaption>
    </figure>
  )
}
