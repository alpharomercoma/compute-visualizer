/**
 * Achievable fractions of peak.
 *
 * Peak is never reached, and a calculator that pretends otherwise overstates
 * throughput by 25-45% while looking authoritative. So every result is computed
 * twice: once against vendor peak (the theoretical roof) and once against these
 * derated rates (what a real kernel gets).
 *
 * Nothing in this file is a vendor specification. Every entry carries where it
 * came from and a plausible range, and every one is meant to be overridden by
 * the calibration panel when a user has measurements from their own machine.
 * Constants in `hardware.ts` are NVIDIA's; constants here are ours.
 */

export interface EfficiencyFactor {
  /** Fraction of peak, in (0, 1]. */
  readonly value: number
  /** Plausible span, for the sensitivity slider. */
  readonly range: readonly [number, number]
  readonly source: string
}

const factor = (
  value: number,
  range: readonly [number, number],
  source: string,
): EfficiencyFactor => {
  if (!(value > 0 && value <= 1)) {
    throw new RangeError(`efficiency must be in (0, 1], got ${value}`)
  }
  return { value, range, source }
}

export interface EfficiencyModel {
  /** Large, well-shaped GEMMs — the best case tensor cores actually reach. */
  readonly gemmLarge: EfficiencyFactor
  /**
   * Skinny GEMMs. Decode is a batch-sized-by-hidden matrix against a weight
   * matrix: `m` is tiny, tiles are poorly filled, and tail waves dominate. This
   * is the single most important derating for inference and it is the one most
   * calculators omit.
   */
  readonly gemmSkinny: EfficiencyFactor
  /** Contiguous weight streaming, the friendliest HBM access pattern there is. */
  readonly hbmStreaming: EfficiencyFactor
  /**
   * KV-cache reads. A paged, per-sequence gather, not a stream, so it lands well
   * below the STREAM figure that vendor-adjacent material tends to quote.
   */
  readonly hbmGather: EfficiencyFactor
  /** End-to-end training, as an independent sanity bound on the whole model. */
  readonly trainingEndToEnd: EfficiencyFactor
}

/**
 * Defaults for H100-class hardware.
 *
 * The training figure is the anchor and deserves its provenance stated plainly:
 * Meta trained Llama 3 405B on 16,384 H100s at 400 TFLOP/s per GPU, BF16 MFU
 * 38-43%. That is a real number from a real run at scale, including every
 * collective, every pipeline bubble and every optimizer step. If this model
 * predicts a training MFU far above 43% for a comparable configuration, this
 * model is wrong.
 */
export const H100_EFFICIENCY: EfficiencyModel = {
  gemmLarge: factor(0.72, [0.65, 0.8], 'measured cuBLAS BF16 4096^3 on H100 SXM (~713 TFLOP/s)'),
  gemmSkinny: factor(0.57, [0.45, 0.7], 'measured cuBLAS on LLaMA FFN shapes (~502-561 TFLOP/s)'),
  hbmStreaming: factor(0.87, [0.8, 0.92], 'STREAM / BabelStream on H100 (~2.9-3.1 TB/s)'),
  hbmGather: factor(0.68, [0.6, 0.8], 'paged KV gather; less contiguous than STREAM'),
  trainingEndToEnd: factor(0.4, [0.34, 0.54], 'Llama 3 405B, 16,384x H100: 400 TFLOP/s/GPU, MFU 38-43%'),
}

/** Everything at 100%: the theoretical roof, for the dual-curve display. */
export const IDEAL_EFFICIENCY: EfficiencyModel = {
  gemmLarge: factor(1, [1, 1], 'theoretical peak'),
  gemmSkinny: factor(1, [1, 1], 'theoretical peak'),
  hbmStreaming: factor(1, [1, 1], 'theoretical peak'),
  hbmGather: factor(1, [1, 1], 'theoretical peak'),
  trainingEndToEnd: factor(1, [1, 1], 'theoretical peak'),
}

/**
 * How confident we are in a reported number.
 *
 * Surfaced in the UI next to every result, because "this came from a fitted
 * guess" and "this came from your own benchmark" should never look alike.
 */
export type Confidence = 'vendor-spec' | 'prior' | 'calibrated'

/**
 * Kernel-launch and framework overhead per kernel.
 *
 * A decode step launches on the order of a dozen kernels per layer, so at 80
 * layers this is real: ~2 us x hundreds of kernels is milliseconds, comparable
 * to the memory time it sits beside. CUDA graph capture removes most of it,
 * which is why the graph-captured figure is a separate entry rather than a
 * blanket reduction.
 *
 * [prior] - not a vendor specification.
 */
export interface OverheadModel {
  /** Per kernel launch, no CUDA graph. */
  readonly launchSeconds: number
  /** Per kernel launch under a captured CUDA graph. */
  readonly graphLaunchSeconds: number
}

export const H100_OVERHEAD: OverheadModel = {
  launchSeconds: 2.5e-6,
  graphLaunchSeconds: 0.4e-6,
}
