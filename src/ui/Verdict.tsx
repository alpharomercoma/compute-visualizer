import type { BottleneckClass } from '../core/schedule'

/**
 * The answer, in words, owning the page.
 *
 * Three things separate this from the heading it replaces.
 *
 * It states a CAUSE and a LEVER, because "memory-bandwidth-bound" names a
 * category and gives you nothing to do about it.
 *
 * The lever carries the tool's own numbers. The previous version said "raise the
 * batch size" while the roofline two screens below said the crossover was at
 * batch 296 and the capacity table said memory gives out at 44 — the tool knew
 * the advice was unreachable and offered it anyway. Generic advice from a
 * calculator that has the specifics is the failure this page exists to avoid.
 *
 * And the share is stated ONCE. It used to appear at hero size, again inside the
 * cause sentence, and a third time in the bar below, which reads as padding.
 */

export interface Crossover {
  /** The axis it moves along, named as the control is named. */
  readonly axis: string
  /** Where the class changes, already formatted. */
  readonly at: string
  /** What becomes the limit there. Never assumed — the sweep reports it. */
  readonly becomes: string
  /** True when what it crosses into is "nothing dominates", not another class. */
  readonly intoBalance: boolean
  /** False when the configuration stops fitting before it gets there. */
  readonly reachable: boolean
}

export interface VerdictProps {
  readonly bottleneck: BottleneckClass
  /** Share of the step held by the limiting class, 0..1. */
  readonly share: number
  readonly phase: 'inference' | 'training'
  /** True when the largest span is real work and still under half the step. */
  readonly balanced?: boolean
  readonly runnerUp?: { readonly label: string; readonly share: number }
  readonly crossover?: Crossover
  /** A sharding that would beat the current one, if the search found one. */
  readonly betterSplit?: { readonly label: string; readonly gain: number; readonly apply: () => void }
}

interface Reading {
  readonly headline: string
  readonly cause: string
  readonly lever: (crossover?: Crossover) => string
}

/**
 * Plain language, and a lever that names an action.
 *
 * The distinction the whole tool exists for is the two network rows: a
 * bandwidth-bound collective wants a faster fabric, a latency-bound one will not
 * notice the fabric at all and wants fewer collectives. Collapsing them into
 * "communication-bound" sends half these users shopping for the wrong hardware.
 */
const READINGS: Record<BottleneckClass, Reading> = {
  'memory-bandwidth': {
    headline: 'Memory bandwidth is the limit',
    cause: 'The step is spent reading weights and KV cache out of HBM, not doing arithmetic.',
    lever: (crossover) =>
      crossover && !crossover.reachable
        ? `Raising ${crossover.axis.toLowerCase()} will not fix this: it stops fitting before ${crossover.at}, where the limit would have changed. Quantize the weights, or shrink the KV cache with a smaller dtype.`
        : crossover?.intoBalance
          ? `Raise ${crossover.axis.toLowerCase()} to ${crossover.at} and nothing dominates any more — that is as balanced as this gets. Quantizing the weights brings that threshold closer.`
          : crossover
            ? `Raise ${crossover.axis.toLowerCase()} to ${crossover.at} and ${crossover.becomes} becomes the limit instead. Quantizing the weights brings that threshold closer.`
            : 'Quantize the weights, shrink the KV cache with a smaller dtype, or raise the batch size.',
  },
  compute: {
    headline: 'The tensor cores are the limit',
    cause: 'The step is arithmetic. The GPUs are doing what you bought them for.',
    lever: () => 'This is the good case. A lower precision is what buys more of it.',
  },
  'network-bandwidth': {
    headline: 'Interconnect bandwidth is the limit',
    cause: 'The step is collectives moving real bytes between GPUs.',
    lever: () =>
      'Use less tensor parallelism, reshard toward data parallelism, or move to a faster fabric.',
  },
  'network-latency': {
    headline: 'Collective startup is the limit',
    cause:
      'The step is spent starting collectives. The messages are tiny — almost nothing is on the wire.',
    // CUDA graph capture belongs on this list in the real world, and is left off
    // it here: in this model the graph flag changes kernel launch cost only, and
    // it neither removes a collective nor shortens one. Naming a lever the tool
    // cannot then demonstrate is how a calculator loses its credibility.
    lever: () =>
      'A faster fabric will not help — the messages are too small to be on it long. Issue fewer collectives: less tensor parallelism, or a larger batch so each one carries more.',
  },
  overhead: {
    headline: 'Kernel launch overhead is the limit',
    cause: 'The step is spent launching work rather than doing it.',
    lever: () => 'Capture the step in a CUDA graph, or fuse the small kernels.',
  },
  serialization: {
    headline: 'The schedule is the limit',
    cause: 'No single resource holds the step. Everything is waiting its turn.',
    lever: () =>
      'Overlap what can overlap: more microbatches to fill the pipeline, or gradient buckets that reduce while the backward pass is still running.',
  },
}

export function Verdict({
  bottleneck,
  share,
  phase,
  balanced,
  runnerUp,
  crossover,
  betterSplit,
}: VerdictProps) {
  const reading = READINGS[bottleneck]
  const percent = Math.round(share * 100)

  /**
   * A balanced step gets its own reading rather than the dominant class's.
   *
   * Saying "memory bandwidth is the limit — 46% of the step" beside a compute
   * slice of 41% invites someone to spend a week on the wrong half. The useful
   * statement is that there is no single thing to fix, which is also the state
   * everything else on this page is trying to move you toward.
   */
  const headline = balanced ? 'Nothing dominates this step' : reading.headline
  const cause =
    balanced && runnerUp
      ? `${reading.cause} But only just: ${runnerUp.label.toLowerCase()} is close behind at ${Math.round(
          runnerUp.share * 100,
        )}%, so no single resource is starving the others.`
      : reading.cause
  const lever = balanced
    ? 'Nothing here is the obvious win. Bigger gains now come from a different split or a different model, not from tuning this one.'
    : reading.lever(crossover)

  /**
   * A large second place gets named.
   *
   * "The tensor cores are the limit — nothing else here is worth touching" was
   * printed above a bar showing 30% collectives, which is emphatically worth
   * touching. One class holding a majority does not make the rest noise.
   */
  const secondary =
    !balanced && runnerUp && runnerUp.share >= 0.2
      ? `${runnerUp.label} is ${Math.round(runnerUp.share * 100)}% and is the next thing worth attacking.`
      : undefined

  return (
    <section className="panel verdict">
      <div>
        <p className="eyebrow" style={{ margin: '0 0 0.5rem' }}>
          {phase === 'inference' ? 'Decode step' : 'Training step'}
        </p>
        <h2>{headline}</h2>
        <p className="cause">{cause}</p>
        <p className="lever">
          <strong>What helps:</strong> {lever}
          {secondary && <span className="muted"> {secondary}</span>}
        </p>

        {betterSplit && (
          <p className="lever" style={{ marginTop: '0.75rem' }}>
            <button type="button" className="ghost" onClick={betterSplit.apply}>
              Try {betterSplit.label}
            </button>{' '}
            <span className="muted">
              — {Math.round(betterSplit.gain * 100)}% faster on the same hardware.
            </span>
          </p>
        )}
      </div>

      <div className="share">
        <b>{percent}%</b>
        <span>of the step</span>
      </div>
    </section>
  )
}
