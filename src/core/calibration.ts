/**
 * Replacing fitted priors with measurements.
 *
 * Every collective constant this model ships is a guess with a citation for its
 * plausibility and nothing more. No credible published `all_reduce_perf` table
 * exists for a single 8xH100 node together with the NCCL version, command line
 * and NVLS state needed to fit one, and inventing that table would have been
 * worse than admitting the gap.
 *
 * So the gap is admitted, and this is the way out of it: paste the output of
 * nccl-tests from your own machine and the model refits alpha and algbw against
 * it, then labels every number downstream as calibrated rather than prior.
 *
 *   NCCL_DEBUG=INFO ./build/all_reduce_perf -b 1K -e 1G -f 2 -g 8 -n 100
 */

import type { CollectiveProfile } from './collectives'
import { algbw, busbwToAlgbw } from './collectives'
import { bytesPerSecond, seconds } from './units'

export interface Measurement {
  readonly sizeBytes: number
  readonly timeSeconds: number
  readonly algbw: number
  readonly busbw: number
}

export interface CalibrationFit {
  readonly measurements: readonly Measurement[]
  /** Small-message intercept, in seconds. */
  readonly alpha: number
  /** Large-message asymptotic algorithm bandwidth, in bytes/second. */
  readonly algbw: number
  /** Where the fit crosses from latency-dominated to bandwidth-dominated. */
  readonly kneeBytes: number
  readonly ranks: number | undefined
}

/**
 * Parses nccl-tests output.
 *
 * The table columns are: size(B), count, type, redop, root, then in-place and
 * out-of-place triples of time(us), algbw(GB/s), busbw(GB/s). Header and debug
 * lines start with '#', so anything that does not begin with a number is
 * skipped rather than guessed at.
 */
export function parseNcclTests(text: string): { measurements: Measurement[]; ranks?: number } {
  const measurements: Measurement[] = []
  let ranks: number | undefined

  for (const line of text.split('\n')) {
    const trimmed = line.trim()

    // "# nThread 1 nGpus 1 minBytes ..." and "# Rank 0 ... on host" lines carry
    // the rank count, which is needed to convert busbw and to sanity-check.
    const rankMatch = /nRanks\s+(\d+)/.exec(trimmed) ?? /nGpus\s+(\d+)/.exec(trimmed)
    if (rankMatch) ranks = Number(rankMatch[1])

    if (!trimmed || trimmed.startsWith('#')) continue

    const columns = trimmed.split(/\s+/)
    if (columns.length < 8) continue

    const sizeBytes = Number(columns[0])
    // Out-of-place timing sits at columns 5..7 in the standard layout.
    const timeUs = Number(columns[5])
    const algbw = Number(columns[6])
    const busbw = Number(columns[7])

    if (
      !Number.isFinite(sizeBytes) ||
      !Number.isFinite(timeUs) ||
      !Number.isFinite(algbw) ||
      sizeBytes <= 0 ||
      timeUs <= 0
    ) {
      continue
    }

    measurements.push({
      sizeBytes,
      timeSeconds: timeUs * 1e-6,
      algbw: algbw * 1e9,
      busbw: busbw * 1e9,
    })
  }

  return ranks !== undefined ? { measurements, ranks } : { measurements }
}

/**
 * Fits `T = alpha + M / algbw` to the measurements.
 *
 * Deliberately not a least-squares fit over the whole range. The two ends of the
 * curve measure different things — the small end is startup, the large end is
 * bandwidth — and a single regression across both would blend them into a line
 * that describes neither. So alpha comes from the smallest messages and algbw
 * from the largest, which is how one reads this table by hand anyway.
 */
export function fitCollective(
  measurements: readonly Measurement[],
  ranks?: number,
): CalibrationFit | undefined {
  if (measurements.length < 2) return undefined

  const sorted = [...measurements].sort((a, b) => a.sizeBytes - b.sizeBytes)

  // Startup: the floor the smallest messages cannot get below.
  const smallest = sorted.slice(0, Math.max(1, Math.floor(sorted.length / 4)))
  const alpha = Math.min(...smallest.map((m) => m.timeSeconds))

  // Bandwidth: the asymptote the largest messages approach.
  const largest = sorted.slice(-Math.max(1, Math.floor(sorted.length / 4)))
  const algbw = Math.max(...largest.map((m) => m.sizeBytes / Math.max(m.timeSeconds - alpha, 1e-12)))

  return {
    measurements: sorted,
    alpha,
    algbw,
    kneeBytes: alpha * algbw,
    ranks,
  }
}

/** Builds a profile from a fit, so calibrated numbers replace the priors. */
export function profileFromFit(fit: CalibrationFit): CollectiveProfile {
  return {
    algorithm: 'ring',
    algbw: algbw(bytesPerSecond(fit.algbw)),
    alpha: seconds(fit.alpha),
    alphaRange: [seconds(fit.alpha), seconds(fit.alpha)],
    eligible: true,
    source: `calibrated from ${fit.measurements.length} nccl-tests measurements`,
  }
}

/**
 * How far the shipped priors were from this machine.
 *
 * Worth showing rather than silently swapping: a user who learns the default was
 * 40% off has learned something about every other number in the tool.
 */
export function compareToPrior(
  fit: CalibrationFit,
  prior: CollectiveProfile,
): { alphaRatio: number; algbwRatio: number } {
  return {
    alphaRatio: fit.alpha / prior.alpha,
    algbwRatio: fit.algbw / prior.algbw,
  }
}

/** Converts a published busbw figure to the algbw this model stores. */
export const busbwEntryToAlgbw = busbwToAlgbw
