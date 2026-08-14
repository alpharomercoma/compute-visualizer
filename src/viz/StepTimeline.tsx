/**
 * One layer, as time.
 *
 * This is the figure the rest of the tool exists to earn. Every other
 * calculator in this space adds up independent terms and reports a total; this
 * model builds the step as a task DAG and runs a list scheduler over two
 * resources, so it knows when each kernel starts, when each collective starts,
 * and which of them determined the finish. None of that survives being
 * summarised into a percentage — so here it is drawn instead.
 *
 * What the picture shows that the stacked bar cannot:
 *
 *   - the ORDER. An all-reduce is not "13% of the step" floating somewhere; it
 *     sits between the output projection and the MLP, and the next GEMM cannot
 *     start until it lands. Seeing that gap is what makes "tensor parallelism
 *     serialises decode" obvious rather than asserted.
 *   - the SHAPE. Wide blocks on the GPU lane with slivers below is a different
 *     problem from a lane that is mostly empty.
 *   - the SCALE. A 5 µs collective startup beside a 40 µs GEMM is a ratio you can
 *     see, and every table in this field hides it.
 *
 * One layer rather than the whole step, because a step is that layer repeated
 * eighty times: drawing all of it would be eighty copies of one pattern, which
 * is a texture, not a diagram.
 */

import type { ScheduleResult } from '../core/schedule'
import { useWidth } from './useWidth'

const COLOR: Record<string, string> = {
  compute: 'var(--c-compute)',
  memory: 'var(--c-memory)',
  'network-bandwidth': 'var(--c-net-bw)',
  'network-latency': 'var(--c-net-lat)',
  overhead: 'var(--c-overhead)',
}

export interface StepTimelineProps {
  readonly schedule: ScheduleResult
  /** Sentence under the figure, tying one layer to the whole step. */
  readonly caption: React.ReactNode
}

interface Block {
  readonly id: string
  readonly label: string
  readonly lane: 0 | 1
  readonly start: number
  readonly duration: number
  readonly limitedBy: string
  readonly onPath: boolean
}

/**
 * The tasks belonging to a single layer.
 *
 * Ids are `L{layer}.{kernel}`, so the grouping is already in the data. Layer 1
 * rather than layer 0 wherever it exists: nothing treats the first layer
 * specially today, but a model that did — a dense first block in an MoE, a
 * hybrid attention schedule — would make layer 0 the unrepresentative one.
 */
function oneLayer(schedule: ScheduleResult): { blocks: Block[]; span: number } | undefined {
  const layers = new Set<number>()
  for (const task of schedule.tasks) {
    const match = /^L(\d+)\./.exec(task.id)
    if (match) layers.add(Number(match[1]))
  }
  if (layers.size === 0) return undefined

  const sorted = [...layers].sort((a, b) => a - b)
  const chosen = sorted.length > 1 ? sorted[1]! : sorted[0]!
  const prefix = `L${chosen}.`

  const tasks = schedule.tasks.filter((t) => t.id.startsWith(prefix))
  if (tasks.length === 0) return undefined

  const t0 = Math.min(...tasks.map((t) => t.start))
  const end = Math.max(...tasks.map((t) => t.finish))
  const onPath = new Set(schedule.criticalPath)

  const blocks = tasks.map(
    (task): Block => ({
      id: task.id,
      label: task.label.replace(/^layer \d+ /, ''),
      lane: task.resource === 'network' ? 1 : 0,
      start: task.start - t0,
      duration: task.duration,
      limitedBy: task.limitedBy,
      onPath: onPath.has(task.id),
    }),
  )

  return { blocks, span: end - t0 }
}

const us = (s: number) => (s >= 1e-3 ? `${(s * 1e3).toFixed(2)} ms` : `${(s * 1e6).toFixed(1)} µs`)

export function StepTimeline({ schedule, caption }: StepTimelineProps) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const layer = oneLayer(schedule)

  const laneHeight = 30
  const laneGap = 8
  const top = 14
  const axis = 26

  return (
    <figure style={{ margin: 0 }}>
      <div ref={ref}>
        {layer && layer.span > 0 && width > 0
          ? (() => {
              const { blocks, span } = layer
              const hasNetwork = blocks.some((b) => b.lane === 1)
              const narrow = width < 520
              const gutter = narrow ? 40 : 100
              const track = Math.max(20, width - gutter)
              const lanes = hasNetwork ? 2 : 1
              const height = top + laneHeight * lanes + laneGap * (lanes - 1) + axis

              const x = (t: number) => gutter + (t / span) * track
              const w = (d: number) => Math.max(1.5, (d / span) * track)
              const y = (lane: 0 | 1) => top + lane * (laneHeight + laneGap)

              const description = blocks
                .filter((b) => b.duration / span > 0.02)
                .map((b) => `${b.label}, ${us(b.duration)}`)
                .join('; ')

              return (
                <svg
                  className="chart"
                  viewBox={`0 0 ${width} ${height}`}
                  width={width}
                  height={height}
                  role="img"
                  aria-label={`One layer takes ${us(span)}. In order: ${description}.`}
                >
                  {/* Lane beds, so an idle interconnect reads as "nothing crossed
                      it" rather than as a row that failed to render. */}
                  {Array.from({ length: lanes }, (_, lane) => (
                    <rect
                      key={lane}
                      x={gutter}
                      y={y(lane as 0 | 1)}
                      width={track}
                      height={laneHeight}
                      rx={3}
                      fill="var(--panel-sunk)"
                      stroke="var(--divider)"
                    />
                  ))}

                  <text x={gutter - 8} y={y(0) + laneHeight / 2 + 4} textAnchor="end">
                    GPU
                  </text>
                  {hasNetwork && (
                    <text x={gutter - 8} y={y(1) + laneHeight / 2 + 4} textAnchor="end">
                      {narrow ? 'Net' : 'Interconnect'}
                    </text>
                  )}

                  {blocks.map((block) => {
                    const bw = w(block.duration)
                    const bx = x(block.start)
                    return (
                      <g key={block.id}>
                        <rect
                          x={bx}
                          y={y(block.lane) + 3}
                          width={bw}
                          height={laneHeight - 6}
                          rx={2}
                          fill={COLOR[block.limitedBy] ?? 'var(--c-serial)'}
                          opacity={block.onPath ? 0.92 : 0.4}
                        >
                          <title>
                            {block.label} — {us(block.duration)}, {block.limitedBy}
                          </title>
                        </rect>
                        {/* Labelled only where the label fits inside its own
                            block. A leader line to a three-pixel sliver is
                            decoration, and the tooltip already carries it. */}
                        {bw > 9 * block.label.length && (
                          <text
                            className="on-fill"
                            x={bx + 7}
                            y={y(block.lane) + laneHeight / 2 + 4}
                            style={{ fontWeight: 600 }}
                          >
                            {block.label}
                          </text>
                        )}
                      </g>
                    )
                  })}

                  <line
                    x1={gutter}
                    y1={height - axis + 6}
                    x2={width}
                    y2={height - axis + 6}
                    stroke="var(--divider)"
                  />
                  <text x={gutter} y={height - 5}>
                    0
                  </text>
                  <text x={width} y={height - 5} textAnchor="end">
                    {us(span)}
                  </text>
                </svg>
              )
            })()
          : null}
      </div>

      <figcaption className="small muted" style={{ marginTop: '0.5rem' }}>
        {caption}
      </figcaption>
    </figure>
  )
}
