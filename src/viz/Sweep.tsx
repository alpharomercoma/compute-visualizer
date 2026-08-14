/**
 * Where the answer changes.
 *
 * A single verdict tells you about today's configuration. This tells you about
 * the machine: run the model across a whole axis and the boundaries fall out —
 * the batch size where memory bandwidth stops holding the step, the context
 * length where the KV cache stops fitting. Those thresholds are what someone
 * remembers a week later, and they are the reason to build the model as a pure
 * function rather than a spreadsheet.
 *
 * It is also the honest way to present a lever. "Raise the batch size" is only
 * advice if raising it reaches something; when the threshold sits past the point
 * where capacity gives out, the strip shows the wall before the boundary and the
 * suggestion is visibly futile.
 *
 * The strip is a control, not a readout: click or drag anywhere on it to move
 * there, so reading the shape and acting on it are the same gesture.
 */

import type { SweepResult, SweepState } from '../app/sweep'
import { BOTTLENECK_NAME } from '../app/accounting'
import { useWidth } from './useWidth'

const COLOR: Record<string, string> = {
  compute: 'var(--c-compute)',
  'memory-bandwidth': 'var(--c-memory)',
  'network-bandwidth': 'var(--c-net-bw)',
  'network-latency': 'var(--c-net-lat)',
  overhead: 'var(--c-overhead)',
  serialization: 'var(--c-serial)',
}

const stateName = (state: SweepState): string =>
  state === 'infeasible'
    ? 'will not fit'
    : state === 'balanced'
      ? 'no single limit'
      : BOTTLENECK_NAME[state]

export interface SweepProps {
  readonly result: SweepResult
  readonly current: number
  /**
   * The state at exactly the current value, from the page's own analysis.
   *
   * Not inferred from the nearest sample: near a threshold the nearest sample
   * can read differently from the verdict above, and a chart that contradicts
   * the headline is worse than one that says less.
   */
  readonly currentState: SweepState
  readonly onChange: (value: number) => void
}

export function Sweep({ result, current, currentState, onChange }: SweepProps) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { axis, points, boundaries } = result

  const gutter = 6
  const track = Math.max(20, width - gutter * 2)
  const barTop = 20
  const barHeight = 34
  const height = barTop + barHeight + 34

  const lo = Math.log(axis.min)
  const hi = Math.log(axis.max)
  const x = (value: number) => gutter + ((Math.log(value) - lo) / (hi - lo)) * track
  const valueAt = (px: number) => {
    const t = Math.min(1, Math.max(0, (px - gutter) / track))
    const raw = Math.exp(lo + t * (hi - lo))
    const snapped = Math.round(raw / axis.step) * axis.step
    return Math.min(axis.max, Math.max(axis.min, snapped))
  }

  const drag = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    onChange(valueAt(event.clientX - box.left))
  }

  // Labels are dropped rather than overlapped: two thresholds twenty pixels
  // apart are unreadable as text, and the strip still shows both edges.
  const labelled: Array<{ at: number; text: string }> = []
  for (const boundary of boundaries) {
    const text =
      boundary.to === 'infeasible'
        ? `stops fitting at ${axis.format(boundary.at)}`
        : boundary.from === 'infeasible'
          ? `fits from ${axis.format(boundary.at)}`
          : `${stateName(boundary.to)} from ${axis.format(boundary.at)}`
    const last = labelled[labelled.length - 1]
    if (last && Math.abs(x(boundary.at) - x(last.at)) < 110) continue
    labelled.push({ at: boundary.at, text })
  }

  const summary = describe(result, current)

  return (
    <figure style={{ margin: 0 }}>
      <div ref={ref}>
        {width > 0 && (
          <svg
            className="chart sweep"
            viewBox={`0 0 ${width} ${height}`}
            width={width}
            height={height}
            role="slider"
            tabIndex={0}
            aria-label={`${axis.label}. ${summary}`}
            aria-valuemin={axis.min}
            aria-valuemax={axis.max}
            aria-valuenow={current}
            aria-valuetext={`${axis.format(current)} — ${stateName(currentState)}`}
            style={{ touchAction: 'none', cursor: 'pointer' }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              drag(e)
            }}
            onPointerMove={(e) => {
              if (e.currentTarget.hasPointerCapture(e.pointerId)) drag(e)
            }}
            onKeyDown={(e) => {
              const step = e.shiftKey ? axis.step * 10 : axis.step
              if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
                e.preventDefault()
                onChange(Math.min(axis.max, current + step))
              } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
                e.preventDefault()
                onChange(Math.max(axis.min, current - step))
              }
            }}
          >
            <defs>
              {/* Infeasible is drawn as absence, not as another colour: it is not
                  a kind of bottleneck, it is the region with no answer. */}
              <pattern
                id="sweep-void"
                width={6}
                height={6}
                patternTransform="rotate(45)"
                patternUnits="userSpaceOnUse"
              >
                <rect width={6} height={6} fill="var(--panel-sunk)" />
                <line x1={0} y1={0} x2={0} y2={6} stroke="var(--control-border)" strokeWidth={1.5} />
              </pattern>
            </defs>

            {/* Merged into runs rather than drawn per sample: forty adjacent
                rects of one colour show forty hairline seams where the edges
                anti-alias, which reads as structure that is not there. */}
            {runs(points).map((run) => {
              const from = x(run.from)
              const to = run.toIsEnd ? gutter + track : x(run.to)
              return (
                <rect
                  key={`${run.from}-${run.key}`}
                  x={from}
                  y={barTop}
                  width={Math.max(1, to - from)}
                  height={barHeight}
                  fill={
                    run.state === 'infeasible'
                      ? 'url(#sweep-void)'
                      : (COLOR[run.fill] ?? 'var(--c-serial)')
                  }
                  // Balanced keeps the dominant hue but drops out of full
                  // strength, because "still mostly memory, but nothing is
                  // starving" is a weaker claim and should look like one.
                  opacity={run.state === 'infeasible' ? 1 : run.state === 'balanced' ? 0.4 : 0.85}
                >
                  <title>
                    {axis.label} {axis.format(run.from)} to {axis.format(run.to)}:{' '}
                    {stateName(run.state)}
                  </title>
                </rect>
              )
            })}

            <rect
              x={gutter}
              y={barTop}
              width={track}
              height={barHeight}
              rx={3}
              fill="none"
              stroke="var(--divider)"
            />

            {/* The ends of the axis, so the strip has a scale and not just a
                shape. On the baseline, where no threshold label can sit — and
                dropped when the marker has parked on top of one, rather than
                printing the same number twice. */}
            {x(current) - gutter > 28 && (
              <text x={gutter} y={barTop + barHeight + 13}>
                {axis.format(axis.min)}
              </text>
            )}
            {gutter + track - x(current) > 28 && (
              <text x={gutter + track} y={barTop + barHeight + 13} textAnchor="end">
                {axis.format(axis.max)}
              </text>
            )}

            {labelled.map((label) => (
              <g key={label.at}>
                <line
                  x1={x(label.at)}
                  y1={barTop}
                  x2={x(label.at)}
                  y2={barTop + barHeight}
                  stroke="var(--panel)"
                  strokeWidth={2}
                />
                <text
                  x={Math.min(width - 2, Math.max(2, x(label.at)))}
                  y={height - 2}
                  textAnchor={
                    x(label.at) > width - 70 ? 'end' : x(label.at) < 70 ? 'start' : 'middle'
                  }
                >
                  {label.text}
                </text>
              </g>
            ))}

            {/* The marker: a full-height rule plus a value above, so where you
                are stays legible against any fill underneath it. */}
            <line
              x1={x(current)}
              y1={barTop - 6}
              x2={x(current)}
              y2={barTop + barHeight + 4}
              stroke="var(--ink)"
              strokeWidth={2}
            />
            <text
              x={Math.min(width - 4, Math.max(4, x(current)))}
              y={12}
              textAnchor={x(current) > width - 60 ? 'end' : x(current) < 40 ? 'start' : 'middle'}
              style={{ fill: 'var(--ink)', fontWeight: 700 }}
            >
              {axis.format(current)}
            </text>
          </svg>
        )}
      </div>

      <figcaption className="small muted" style={{ marginTop: '0.5rem' }}>
        {summary}
      </figcaption>
    </figure>
  )
}

interface Run {
  readonly state: SweepState
  /** The class whose hue fills the run, which balance does not change. */
  readonly fill: string
  readonly key: string
  readonly from: number
  readonly to: number
  readonly toIsEnd: boolean
}

/** Contiguous stretches of the axis that read the same way. */
function runs(points: SweepResult['points']): Run[] {
  const out: Run[] = []

  for (const point of points) {
    // Balanced runs are split by their dominant class as well as their state,
    // so a single run never has to be two colours at once.
    const key = point.state === 'balanced' ? `balanced:${point.bottleneck}` : point.state
    const last = out[out.length - 1]
    if (last && last.key === key) {
      out[out.length - 1] = { ...last, to: point.value }
    } else {
      out.push({
        state: point.state,
        fill: point.bottleneck ?? point.state,
        key,
        from: point.value,
        to: point.value,
        toIsEnd: false,
      })
    }
  }

  // Each run reaches the next one's first sample, otherwise the gap between the
  // last sample of one and the first of the next paints as unfilled track. The
  // final run owns the rest: its last sample is the last one taken, not the last
  // value the axis admits.
  return out.map((run, i) => {
    const next = out[i + 1]
    return next ? { ...run, to: next.from } : { ...run, toIsEnd: true }
  })
}

/**
 * The shape of the answer, in a sentence.
 *
 * The case worth naming explicitly is a lever that cannot reach: when the
 * threshold sits past the capacity wall, "raise the batch size" is not a smaller
 * version of the right advice, it is the wrong advice.
 */
function describe(result: SweepResult, current: number): string {
  const { axis, points, boundaries, capacityLimit } = result
  const unit = axis.label.toLowerCase()
  const feasible = points.filter((p) => p.feasible)

  if (feasible.length === 0) return 'Nothing on this axis fits.'

  const states = new Set(feasible.map((p) => p.state))
  const next = boundaries.find((b) => b.at > current && b.to !== 'infeasible')

  if (states.size === 1) {
    const only = stateName([...states][0]!)
    return capacityLimit !== undefined
      ? `${sentence(only)} at every ${unit} that fits — the last one is ${axis.format(
          capacityLimit,
        )}. Moving along this axis will not change the answer.`
      : `${sentence(only)} across the whole range. This axis is not the one that matters here.`
  }

  if (next) {
    const reachable =
      capacityLimit === undefined || next.at <= capacityLimit
        ? 'It still fits there.'
        : `Out of reach: it stops fitting at ${axis.format(capacityLimit)}, before that.`
    return `At ${axis.format(next.at)} the answer becomes "${stateName(next.to)}". ${reachable}`
  }

  // Past every boundary. Saying so alone — which this used to do — tells you
  // nothing you can act on; what is useful is which way the answer lies and how
  // far back it is.
  const previous = [...boundaries].reverse().find((b) => b.at <= current)
  if (previous) {
    return `Nothing changes above this ${unit}. Below ${axis.format(
      previous.at,
    )} the answer goes back to "${stateName(previous.from)}".`
  }

  return `Nothing on this axis changes the answer from here.`
}

const sentence = (name: string) =>
  name === 'no single limit' ? 'No single limit' : `Limited by ${name}`
