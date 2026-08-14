/**
 * Collective communication cost.
 *
 * This module exists in the shape it does to make one specific bug
 * unrepresentable. `nccl-tests` reports two bandwidths:
 *
 *   algbw = S / t
 *   busbw = algbw * 2(n-1)/n        (for all-reduce)
 *
 * and its own documentation notes the conversion is applied "independent of the
 * algorithm used (ring, tree, or other)". So a published busbw ALREADY contains
 * the 1.75 factor at n=8, whatever actually ran. Divide by it without
 * re-applying the factor and you underpredict by 1.75x; apply the factor to an
 * algbw figure and you overpredict by the same amount. Both reviewers of this
 * model independently named this as the likeliest place for it to go wrong.
 *
 * The fix is structural rather than careful: this module stores `algbw` only,
 * times collectives as `T = alpha + M/algbw`, and offers exactly one function
 * for converting a published busbw at the point of data entry. There is no
 * ring factor anywhere in the timing path, so it cannot be applied twice.
 */

import { type Bytes, type BytesPerSecond, type Seconds, bytesPerSecond, seconds } from './units'

declare const collectiveBrand: unique symbol

/**
 * Algorithm bandwidth: buffer size over elapsed time. What timing needs.
 *
 * Distinct from bus bandwidth in the TYPE, not merely by convention. A review
 * pointed out that sharing `BytesPerSecond` between the two meant the
 * double-count was only avoided by the current call sites happening to be
 * correct — which is not the same as being impossible. It is now impossible:
 * a busbw figure will not compile where an algbw is required.
 */
export type Algbw = BytesPerSecond & { readonly [collectiveBrand]: 'algbw' }

/** Bus bandwidth as `nccl-tests` reports it, ring factor already folded in. */
export type Busbw = BytesPerSecond & { readonly [collectiveBrand]: 'busbw' }

/** The user's buffer — the `S` in `algbw = S/t`. What a collective is timed on. */
export type AlgorithmicMessage = Bytes & { readonly [collectiveBrand]: 'algorithmic' }

/** Physical bytes crossing a port. For reporting only; never for timing. */
export type WireTraffic = Bytes & { readonly [collectiveBrand]: 'wire' }

export const algbw = (value: BytesPerSecond): Algbw => value as Algbw
export const busbw = (value: BytesPerSecond): Busbw => value as Busbw
export const algorithmicMessage = (value: Bytes): AlgorithmicMessage => value as AlgorithmicMessage

export type CollectiveOp = 'all-reduce' | 'all-gather' | 'reduce-scatter' | 'broadcast' | 'all-to-all'

export type CollectiveAlgorithm = 'ring' | 'tree' | 'nvls'

/**
 * The factor `nccl-tests` multiplies into algbw to report busbw.
 *
 * Present for ONE purpose: converting published measurements at the catalog
 * boundary. It is never used when timing.
 */
export function busbwFactor(op: CollectiveOp, ranks: number): number {
  if (ranks <= 1) return 1
  switch (op) {
    case 'all-reduce':
      return (2 * (ranks - 1)) / ranks
    case 'all-gather':
    case 'reduce-scatter':
    case 'all-to-all':
      return (ranks - 1) / ranks
    case 'broadcast':
      return 1
  }
}

/**
 * Converts a published nccl-tests busbw into the algbw this model stores.
 *
 * Use this once, at the point a benchmark number enters the system, and never
 * again. If a figure has already been through here, it is algbw and must not be
 * scaled a second time.
 */
export function busbwToAlgbw(measured: Busbw, op: CollectiveOp, ranks: number): Algbw {
  return algbw(bytesPerSecond(measured / busbwFactor(op, ranks)))
}

export interface CollectiveProfile {
  readonly algorithm: CollectiveAlgorithm
  /** Large-message algorithm bandwidth. Typed so a busbw cannot be passed. */
  readonly algbw: Algbw
  /** Startup cost, independent of message size. [prior] */
  readonly alpha: Seconds
  readonly alphaRange: readonly [Seconds, Seconds]
  /** Whether the topology and software stack can select this algorithm at all. */
  readonly eligible: boolean
  readonly source: string
}

const us = (n: number) => seconds(n * 1e-6)
const GBps = (n: number) => bytesPerSecond(n * 1e9)
const measuredBusbw = (n: number) => busbw(GBps(n))

/**
 * Defaults for an 8x H100 NVSwitch node.
 *
 * Every figure is a FITTED PRIOR, not a vendor specification. No credible
 * published all_reduce_perf table exists for a single 8xH100 node together with
 * the NCCL version, command line and NVLS state needed to fit one, and none is
 * invented here. The calibration path exists precisely so a user with the
 * hardware can replace these.
 *
 * Ring:  busbw ~370 GB/s measured  ->  algbw ~211 GB/s
 * NVLS:  busbw ~480 GB/s reported  ->  algbw ~274 GB/s
 *
 * The NVLS busbw of 480 GB/s exceeds the 450 GB/s NVLink line rate, which is
 * itself proof the conversion is wrong for in-network reduction: nccl-tests
 * applies the ring formula regardless, and PR #239 proposes (n-1)/(n+1) for
 * NVLS. It is a benchmark metric, not wire traffic.
 */
export function defaultProfiles(ranks: number, hasNvSwitch: boolean): CollectiveProfile[] {
  return [
    {
      algorithm: 'ring',
      algbw: busbwToAlgbw(measuredBusbw(370), 'all-reduce', ranks),
      alpha: us(5),
      alphaRange: [us(3), us(10)],
      eligible: true,
      source: 'fitted prior: ~370 GB/s busbw on 8x H100, nccl-tests',
    },
    {
      algorithm: 'tree',
      algbw: busbwToAlgbw(measuredBusbw(250), 'all-reduce', ranks),
      alpha: us(4),
      alphaRange: [us(2.5), us(8)],
      eligible: true,
      source: 'fitted prior: tree is latency-favoured, bandwidth-poorer than ring',
    },
    {
      algorithm: 'nvls',
      // NVLS reduces in the switch, so the ring factor genuinely vanishes from
      // the physical traffic. The measured large-message algbw is what belongs
      // here regardless -- not the 450 GB/s line rate, which no algorithm
      // reaches.
      algbw: algbw(GBps(274)),
      alpha: us(5),
      alphaRange: [us(3), us(10)],
      // NVLS needs NVSwitch multicast support and NCCL >= 2.17.
      eligible: hasNvSwitch && ranks >= 4,
      source: 'fitted prior: ~274 GB/s algbw (480 GB/s reported busbw) on 8x H100',
    },
  ]
}

/**
 * Time for one collective.
 *
 *   T = alpha + M / algbw
 *
 * `M` is the ALGORITHMIC message size — the size of the user's buffer, the same
 * `S` that nccl-tests divides by elapsed time. No ring factor. See the module
 * comment for why that is stated three times.
 */
export function collectiveTime(
  messageBytes: AlgorithmicMessage,
  profile: CollectiveProfile,
): Seconds {
  if (!profile.eligible) return seconds(Number.POSITIVE_INFINITY)
  if (messageBytes === 0) return profile.alpha
  return seconds(profile.alpha + messageBytes / profile.algbw)
}

export interface CollectiveChoice {
  readonly profile: CollectiveProfile
  readonly time: Seconds
  /** Whether the startup cost dominates the transfer. */
  readonly latencyBound: boolean
}

/**
 * Picks the algorithm NCCL would, by taking the fastest eligible one.
 *
 * This is the only defensible rule available in a browser: NCCL's real selection
 * consults a tuning model built from the installed topology, and hardcoding
 * thresholds would be inventing specificity we do not have.
 */
export function chooseCollective(
  messageBytes: AlgorithmicMessage,
  profiles: readonly CollectiveProfile[],
): CollectiveChoice {
  let best: CollectiveChoice | undefined

  for (const profile of profiles) {
    if (!profile.eligible) continue
    const time = collectiveTime(messageBytes, profile)
    if (best === undefined || time < best.time) {
      const transfer = messageBytes / profile.algbw
      best = { profile, time, latencyBound: profile.alpha > transfer }
    }
  }

  if (best === undefined) {
    throw new Error('no eligible collective algorithm for this topology')
  }
  return best
}

/**
 * Message size at which transfer time overtakes startup cost.
 *
 *   M_knee = alpha * algbw
 *
 * Below it a collective is latency-bound and faster interconnect buys nothing;
 * above it, bandwidth is what matters. At alpha = 5 us and algbw = 274 GB/s the
 * knee is about 1.4 MB. A batch-1 TP=8 decode all-reduce is 16 KiB — nearly two
 * orders of magnitude below, which is why that workload is latency-bound and why
 * calling it merely "communication-bound" would send a user to buy hardware that
 * changes nothing.
 */
export function latencyBandwidthKnee(profile: CollectiveProfile): Bytes {
  return (profile.alpha * profile.algbw) as Bytes
}

/**
 * Physical bytes crossing one rank's port. Reporting only — never timing.
 *
 * Ring moves 2(p-1)/p of the payload in each direction. NVLS sends the payload
 * up to the switch once and receives the reduced result once, so the ring factor
 * genuinely vanishes.
 */
export function wireBytes(
  messageBytes: AlgorithmicMessage,
  op: CollectiveOp,
  ranks: number,
  algorithm: CollectiveAlgorithm,
): WireTraffic {
  if (algorithm === 'nvls') return messageBytes as unknown as WireTraffic
  return (messageBytes * busbwFactor(op, ranks)) as WireTraffic
}
