import {
  algorithmicMessage,
  chooseCollective,
  defaultProfiles,
  latencyBandwidthKnee,
} from '../core/collectives'
import { bytes, KiB, MiB } from '../core/units'
import type { Topology } from '../core/hardware'

/**
 * What the interconnect is actually doing.
 *
 * A verdict of "network-latency-bound" is unconvincing without this: it is the
 * evidence that a 16 KiB message spends microseconds in startup and nanoseconds
 * on the wire, and therefore that a faster fabric would change nothing.
 */

export interface CollectivesProps {
  readonly topology: Topology
  readonly tensorParallel: number
  readonly messageBytes: number
  readonly collectivesPerStep: number
}

const bytesLabel = (value: number) =>
  value >= MiB(1) ? `${(value / MiB(1)).toFixed(1)} MiB` : `${(value / KiB(1)).toFixed(0)} KiB`

const us = (seconds: number) => `${(seconds * 1e6).toFixed(1)} µs`

export function Collectives({
  topology,
  tensorParallel,
  messageBytes,
  collectivesPerStep,
}: CollectivesProps) {
  if (tensorParallel <= 1) {
    return (
      <p className="small muted">
        No collectives: a single rank holds the whole model, so nothing crosses the interconnect.
      </p>
    )
  }

  const profiles = defaultProfiles(tensorParallel, topology.fabric === 'nvswitch')
  const choice = chooseCollective(algorithmicMessage(bytes(messageBytes)), profiles)
  const knee = latencyBandwidthKnee(choice.profile)
  const transfer = messageBytes / choice.profile.algbw

  return (
    <div>
      <dl className="rows">
        <Row label="Algorithm" value={choice.profile.algorithm.toUpperCase()} />
        <Row label="Message" value={bytesLabel(messageBytes)} />
        <Row label="Per step" value={`${collectivesPerStep}`} />
        <Row label="Startup" value={us(choice.profile.alpha)} />
        <Row label="On the wire" value={us(transfer)} />
      </dl>

      <p className="small muted" style={{ marginTop: '0.75rem' }}>
        {choice.latencyBound ? (
          <>
            Latency-bound. Transfer is {(choice.profile.alpha / transfer).toFixed(0)}× faster than
            the startup cost, and the bandwidth–latency knee sits at {bytesLabel(knee)} — about{' '}
            {(knee / messageBytes).toFixed(0)}× this message. A faster fabric would not move this;
            fewer collectives would.
          </>
        ) : (
          <>
            Bandwidth-bound. The message clears the {bytesLabel(knee)} knee, so the wire, not the
            startup cost, sets the pace.
          </>
        )}
      </p>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}
