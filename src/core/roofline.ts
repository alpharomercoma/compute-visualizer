/**
 * Plot geometry for the roofline and its scoped companion.
 *
 * Pure numbers: no SVG, no React, no pixels. The chart component maps these to
 * screen space, so the geometry stays testable and the physics stays out of the
 * view layer.
 *
 * Two deliberate absences:
 *
 * 1. There is no network ceiling on the classic roofline. Drawn at
 *    FLOPs_total/t_net it is not a roof at all — adding dummy FLOPs to a kernel
 *    makes the "ceiling" rise, because communication consumes neither the FLOPs
 *    nor the DRAM bytes the axes measure.
 * 2. There is no whole-step operating point. A step's aggregate intensity is a
 *    descriptive statistic, not a latency predictor: by Jensen's inequality it
 *    parks near the ridge and implies a utilisation that never happens. Points
 *    are per kernel class.
 */

import type { Accelerator, TensorPrecision } from './hardware'
import type { Kernel } from './kernels'
import { type EfficiencyModel } from './efficiency'
import { intensity } from './units'

export interface Roofline {
  /** Horizontal roof: peak compute, FLOP/s. */
  readonly computeRoof: number
  /** Same, derated to what large GEMMs actually reach. */
  readonly achievableComputeRoof: number
  /** Slope of the diagonal roof: bytes/s. */
  readonly memoryRoof: number
  readonly achievableMemoryRoof: number
  /** Where the two meet, in FLOP/byte. */
  readonly ridgePoint: number
  readonly achievableRidgePoint: number
}

export function roofline(
  accelerator: Accelerator,
  precision: TensorPrecision,
  efficiency: EfficiencyModel,
): Roofline {
  const compute = accelerator.peak[precision]
  const memory = accelerator.memoryBandwidth
  const achievableCompute = compute * efficiency.gemmLarge.value
  const achievableMemory = memory * efficiency.hbmStreaming.value

  return {
    computeRoof: compute,
    achievableComputeRoof: achievableCompute,
    memoryRoof: memory,
    achievableMemoryRoof: achievableMemory,
    ridgePoint: compute / memory,
    achievableRidgePoint: achievableCompute / achievableMemory,
  }
}

/** Attainable performance at a given intensity: min(peak, I x bandwidth). */
export function attainable(roof: Roofline, arithmeticIntensity: number, derated = false): number {
  const compute = derated ? roof.achievableComputeRoof : roof.computeRoof
  const memory = derated ? roof.achievableMemoryRoof : roof.memoryRoof
  return Math.min(compute, arithmeticIntensity * memory)
}

export interface OperatingPoint {
  readonly label: string
  readonly kernelClass: string
  /** FLOP/byte of HBM traffic. */
  readonly arithmeticIntensity: number
  /** FLOP/s this kernel actually achieved. */
  readonly achieved: number
  /** Fraction of what the roofline says was possible at this intensity. */
  readonly efficiencyAgainstRoof: number
  readonly memoryBound: boolean
}

export function operatingPoint(
  kernel: Kernel,
  elapsedSeconds: number,
  roof: Roofline,
  label = kernel.name,
): OperatingPoint | undefined {
  // A kernel with no traffic and no work has no position. Plotting it at the
  // origin would be inventing a data point.
  if (kernel.hbmBytes === 0 && kernel.flops === 0) return undefined
  if (elapsedSeconds <= 0) return undefined

  const i = intensity(kernel.flops, kernel.hbmBytes)
  const achieved = kernel.flops / elapsedSeconds
  const ceiling = attainable(roof, i)

  return {
    label,
    kernelClass: kernel.kernelClass,
    arithmeticIntensity: i,
    achieved,
    efficiencyAgainstRoof: ceiling > 0 ? achieved / ceiling : 0,
    memoryBound: i < roof.ridgePoint,
  }
}

/**
 * The two closed-form decode points, for annotation rather than computation.
 *
 * Both are asymptotes that hold while activation traffic is small next to weight
 * and KV traffic — which is the whole of the decode regime. The scheduler uses
 * exact per-projection traffic; these exist because they are what a reader can
 * hold in their head.
 */
export interface DecodeAsymptotes {
  /** Linear layers: 2b/w_weight. Walks right as batch grows. */
  readonly linear: number
  /** Attention: 2*n_q/(n_kv*w_kv). Does not move with batch, ever. */
  readonly attention: number
  /** Batch at which the linear point crosses the ridge. */
  readonly crossoverBatch: number
}

export function decodeAsymptotes(
  batch: number,
  weightBytes: number,
  queryHeads: number,
  kvHeads: number,
  kvBytes: number,
  roof: Roofline,
): DecodeAsymptotes {
  return {
    linear: (2 * batch) / weightBytes,
    // Independent of batch: each sequence owns its KV cache, so batching adds
    // proportionally more traffic alongside the extra work. This is why long
    // context decode never becomes compute-bound at any batch size, and why
    // GQA, MLA and KV quantization are the only levers that move it.
    attention: (2 * queryHeads) / (kvHeads * kvBytes),
    crossoverBatch: (roof.ridgePoint * weightBytes) / 2,
  }
}

/**
 * Ridgeline (arXiv:2209.01368), plotted per kernel class and clearly scoped.
 *
 * Its axes make all three bounds machine constants, which is what a network
 * ceiling on the classic roofline fails to do:
 *
 *   x = memory bytes / network bytes      vertical boundary   at B_mem/B_net
 *   y = FLOPs / memory bytes              horizontal boundary at C/B_mem
 *   x*y = FLOPs / network bytes           diagonal boundary   at C/B_net, slope -1
 *
 * all three meeting at (B_mem/B_net, C/B_mem). Adding dummy FLOPs now moves the
 * workload point up rather than moving a ceiling, which is physically right.
 *
 * It is secondary, not primary, for two reasons stated in docs/MODEL.md: it
 * assumes T_net = bytes/bandwidth and so silently misplaces latency-bound
 * points, and both axes are intensities so attainable performance is not
 * readable off the plot. Callers must surface the scope note.
 */
export interface RidgelinePlane {
  readonly memoryVsNetwork: number
  readonly computeVsMemory: number
  readonly computeVsNetwork: number
  readonly scopeNote: string
}

export function ridgelinePlane(
  peakCompute: number,
  memoryBandwidth: number,
  networkBandwidth: number,
): RidgelinePlane {
  return {
    memoryVsNetwork: memoryBandwidth / networkBandwidth,
    computeVsMemory: peakCompute / memoryBandwidth,
    computeVsNetwork: peakCompute / networkBandwidth,
    scopeNote:
      'Assumes bandwidth-dominated communication. Excludes collective startup latency, ' +
      'serial ordering and launch overhead, so latency-bound points are misplaced.',
  }
}

export type RidgelineRegion = 'compute' | 'memory' | 'network'

export interface RidgelinePoint {
  readonly label: string
  readonly memoryIntensity: number
  readonly arithmeticIntensity: number
  readonly networkIntensity: number
  readonly region: RidgelineRegion
  /** True when the point's own collectives are latency-dominated, so it lies. */
  readonly unreliable: boolean
}

export function ridgelinePoint(
  label: string,
  flopsCount: number,
  memoryBytes: number,
  networkBytes: number,
  plane: RidgelinePlane,
  latencyDominated: boolean,
): RidgelinePoint | undefined {
  // A kernel with no network traffic sits at x = infinity; a bare collective
  // sits at y = 0. Neither has a position on this plane, and inventing one is
  // exactly the degeneracy that keeps Ridgeline secondary.
  if (memoryBytes === 0 || networkBytes === 0) return undefined

  const x = memoryBytes / networkBytes
  const y = flopsCount / memoryBytes
  const xy = flopsCount / networkBytes

  const region: RidgelineRegion =
    y < plane.computeVsMemory && x > plane.memoryVsNetwork
      ? 'memory'
      : xy < plane.computeVsNetwork && x < plane.memoryVsNetwork
        ? 'network'
        : 'compute'

  return {
    label,
    memoryIntensity: x,
    arithmeticIntensity: y,
    networkIntensity: xy,
    region,
    unreliable: latencyDominated,
  }
}

/** Decade ticks spanning a range, for a log axis. */
export function decadeTicks(min: number, max: number): number[] {
  if (!(min > 0) || !(max > min)) return []
  const ticks: number[] = []
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    ticks.push(10 ** e)
  }
  return ticks
}

/** Position on a log axis, normalised to [0, 1]. */
export function logPosition(value: number, min: number, max: number): number {
  if (!(value > 0) || !(min > 0) || !(max > min)) return 0
  return (Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min))
}
