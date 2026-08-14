/**
 * The roofline, hand-rolled.
 *
 * No charting library. On log-log axes every ceiling is a straight line, ticks
 * are decades, and the scale is one subtraction — which makes a dependency cost
 * more than it saves. Recharts alone would be 148 KB gzipped, heavier than a
 * minimal three.js scene, and would still need hand-written geometry for
 * annotated ceilings.
 *
 * Two roofs only: compute and HBM. There is deliberately no network ceiling
 * here. Drawn at FLOPs/t_net it is not a roof at all, because adding dummy FLOPs
 * to a kernel would make it rise. Communication is reported by the time
 * breakdown beside this chart, where it can be stated honestly.
 */

import type { Roofline as RooflineModel, OperatingPoint } from '../core/roofline'
import { attainable, decadeTicks, logPosition } from '../core/roofline'

const CLASS_COLOR: Record<string, string> = {
  'qkv-projection': 'var(--c-compute)',
  'output-projection': 'var(--c-compute)',
  mlp: 'var(--c-compute)',
  attention: 'var(--c-memory)',
  norm: 'var(--c-overhead)',
  'lm-head': 'var(--c-net-bw)',
}

export interface RooflineProps {
  readonly model: RooflineModel
  readonly points: readonly OperatingPoint[]
  /** Closed-form annotations: where the decode points sit and why. */
  readonly asymptotes?: {
    readonly linear: number
    readonly attention: number
  }
  readonly showTheoretical: boolean
  readonly height?: number
}

const PAD = { top: 16, right: 130, bottom: 34, left: 62 }

/**
 * Nudges labels apart so they can all be read.
 *
 * The projection kernels genuinely sit almost on top of each other — they have
 * near-identical intensity — so without this three labels overprint into an
 * illegible smear. Dropping labels instead would hide the fact that several
 * classes share a position, which is itself informative.
 */
interface Placed {
  readonly point: OperatingPoint
  readonly x: number
  readonly y: number
  readonly labelY: number
}

function placeLabels(
  items: ReadonlyArray<{ point: OperatingPoint; x: number; y: number }>,
  minGap = 11,
): Placed[] {
  const byHeight = [...items].sort((a, b) => a.y - b.y)

  let previous = Number.NEGATIVE_INFINITY
  const placed: Placed[] = []
  for (const item of byHeight) {
    const labelY = Math.max(item.y + 3, previous + minGap)
    placed.push({ ...item, labelY })
    previous = labelY
  }
  return placed
}

export function Roofline({
  model,
  points,
  asymptotes,
  showTheoretical,
  height = 250,
}: RooflineProps) {
  // A wide viewBox keeps the rendered height near this figure at the widths this
  // panel actually gets, so the chart does not grow to half the viewport.
  const width = 940

  const intensities = points.map((p) => p.arithmeticIntensity).filter((i) => i > 0 && isFinite(i))
  const xMin = 10 ** Math.floor(Math.log10(Math.min(0.5, ...intensities)))
  const xMax = 10 ** Math.ceil(Math.log10(Math.max(model.ridgePoint * 8, ...intensities, 1000)))
  const yMax = 10 ** Math.ceil(Math.log10(model.computeRoof))
  const yMin = yMax / 1e5

  const plotWidth = width - PAD.left - PAD.right
  const plotHeight = height - PAD.top - PAD.bottom

  const px = (i: number) => PAD.left + logPosition(i, xMin, xMax) * plotWidth
  const py = (f: number) => PAD.top + (1 - logPosition(f, yMin, yMax)) * plotHeight

  const roofPath = (derated: boolean) => {
    const ridge = derated ? model.achievableRidgePoint : model.ridgePoint
    const ceiling = derated ? model.achievableComputeRoof : model.computeRoof
    return [
      `M ${px(xMin)} ${py(attainable(model, xMin, derated))}`,
      `L ${px(ridge)} ${py(ceiling)}`,
      `L ${px(xMax)} ${py(ceiling)}`,
    ].join(' ')
  }

  const formatFlops = (f: number) =>
    f >= 1e15 ? `${f / 1e15} PF` : f >= 1e12 ? `${f / 1e12} TF` : `${f / 1e9} GF`

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Roofline. Ridge point ${model.ridgePoint.toFixed(0)} FLOP per byte. ${points
        .map(
          (p) =>
            `${p.label} at intensity ${p.arithmeticIntensity.toFixed(1)}, ${
              p.memoryBound ? 'memory-bound' : 'compute-bound'
            }`,
        )
        .join('. ')}`}
    >
      <title>Attainable performance against arithmetic intensity</title>

      {decadeTicks(xMin, xMax).map((tick) => (
        <g key={`x${tick}`}>
          <line className="grid" x1={px(tick)} y1={PAD.top} x2={px(tick)} y2={height - PAD.bottom} />
          <text x={px(tick)} y={height - PAD.bottom + 12} textAnchor="middle">
            {tick >= 1000 ? `${tick / 1000}k` : tick}
          </text>
        </g>
      ))}

      {decadeTicks(yMin, yMax).map((tick) => (
        <g key={`y${tick}`}>
          <line className="grid" x1={PAD.left} y1={py(tick)} x2={width - PAD.right} y2={py(tick)} />
          <text x={PAD.left - 6} y={py(tick) + 3} textAnchor="end">
            {formatFlops(tick)}
          </text>
        </g>
      ))}

      {showTheoretical && <path className="roof" d={roofPath(false)} />}
      <path className={showTheoretical ? 'roof-derated' : 'roof'} d={roofPath(true)} />

      {/* The ridge is the single most explanatory number on the chart: below it
          nothing can be compute-bound, whatever the hardware costs. */}
      <line
        x1={px(model.ridgePoint)}
        y1={PAD.top}
        x2={px(model.ridgePoint)}
        y2={height - PAD.bottom}
        stroke="var(--rule-strong)"
        strokeDasharray="2 4"
      />
      <text x={px(model.ridgePoint) + 4} y={PAD.top + 9} fill="var(--ink-2)">
        RIDGE {model.ridgePoint.toFixed(0)}
      </text>

      {asymptotes && (
        <g>
          <line
            x1={px(asymptotes.attention)}
            y1={height - PAD.bottom}
            x2={px(asymptotes.attention)}
            y2={height - PAD.bottom - 8}
            stroke="var(--c-memory)"
            strokeWidth={2}
          />
          <text
            x={px(asymptotes.attention)}
            y={height - PAD.bottom - 12}
            textAnchor="middle"
            fill="var(--c-memory)"
          >
            ATTN
          </text>
        </g>
      )}

      {placeLabels(
        points.map((point) => ({
          point,
          x: px(point.arithmeticIntensity),
          y: py(point.achieved),
        })),
      ).map(({ point, x, y, labelY }) => {
        const color = CLASS_COLOR[point.kernelClass] ?? 'var(--ink-2)'
        if (!isFinite(x) || !isFinite(y)) return null
        return (
          <g key={point.label}>
            {/* Leader line, for labels nudged clear of a neighbour. */}
            {Math.abs(labelY - y) > 2 && (
              <line
                x1={x + 5}
                y1={y}
                x2={x + 9}
                y2={labelY - 3}
                stroke="var(--rule-strong)"
                strokeWidth={1}
              />
            )}
            {/* A surface ring keeps overlapping marks separable. */}
            <circle cx={x} cy={y} r={5.5} fill="var(--surface)" />
            <circle cx={x} cy={y} r={4} fill={color}>
              <title>
                {point.label}: {point.arithmeticIntensity.toFixed(1)} FLOP/byte,{' '}
                {point.memoryBound ? 'memory-bound' : 'compute-bound'}
              </title>
            </circle>
            <text x={x + 10} y={labelY} fill="var(--ink-2)">
              {point.label.toUpperCase()}
            </text>
          </g>
        )
      })}

      <text className="axis-title" x={PAD.left} y={height - 4}>
        ARITHMETIC INTENSITY — FLOP / BYTE OF HBM TRAFFIC
      </text>
    </svg>
  )
}
