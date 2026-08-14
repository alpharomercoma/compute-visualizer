/**
 * Physical units and scopes, as types.
 *
 * The reason this file exists ahead of any physics: during design review, an
 * independent reviewer reported a decode step's per-GPU KV-cache traffic as
 * 21.47 GB. The true per-GPU figure is 2.68 GB — the 21.47 was the cluster-wide
 * total, quoted as if it were per-GPU. An 8x error, in a number that then
 * propagated into a 2x wrong step time and a 2x wrong throughput.
 *
 * That class of mistake is not carelessness, it is the predictable result of
 * passing bare `number` around a codebase where a number can mean bytes or
 * seconds, per-GPU or cluster-wide. So both facts live in the type:
 *
 *   Bytes                 — a quantity of bytes, scope unspecified
 *   PerGpu<Bytes>         — bytes on one GPU
 *   Cluster<Bytes>        — bytes summed across every GPU
 *
 * `PerGpu<Bytes>` is assignable to `Bytes` (widening to "some bytes" is always
 * sound), but `Cluster<Bytes>` is not assignable to `PerGpu<Bytes>`. The exact
 * substitution that produced the 8x error will not compile.
 */

declare const unitBrand: unique symbol
declare const scopeBrand: unique symbol

type Unit<U extends string> = { readonly [unitBrand]: U }
type Scope<S extends string> = { readonly [scopeBrand]: S }

/** Floating-point operations. Always a count of operations, never a rate. */
export type Flops = number & Unit<'flops'>

/** Bytes. Always bytes — never bits, never elements. See `bitsToBytes`. */
export type Bytes = number & Unit<'bytes'>

/** Seconds. The only time unit in the core; format to ms/us at the edges. */
export type Seconds = number & Unit<'seconds'>

/** Floating-point operations per second. */
export type FlopsPerSecond = number & Unit<'flops/s'>

/** Bytes per second. */
export type BytesPerSecond = number & Unit<'bytes/s'>

/**
 * Arithmetic (operational) intensity: FLOPs per byte moved.
 *
 * Which bytes is the whole question — DRAM traffic for the classic roofline,
 * network traffic for the communication analogue. The type deliberately does
 * not distinguish them, because the call sites name which one they mean and
 * conflating the two is caught by review, not by the compiler.
 */
export type FlopsPerByte = number & Unit<'flops/byte'>

/** A quantity measured on a single accelerator. */
export type PerGpu<T> = T & Scope<'per-gpu'>

/** A quantity summed over every accelerator taking part. */
export type Cluster<T> = T & Scope<'cluster'>

/**
 * Rejects values that cannot describe a physical quantity.
 *
 * A calculator that silently returns NaN teaches its user nothing; a calculator
 * that throws at the point the nonsense entered names the bug. Negative FLOPs or
 * negative bytes always mean an upstream sign error, so they are refused here
 * rather than propagated into a plausible-looking time.
 */
function checked(value: number, what: string): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${what} must be finite, got ${value}`)
  }
  if (value < 0) {
    throw new RangeError(`${what} must be non-negative, got ${value}`)
  }
  return value
}

export const flops = (n: number): Flops => checked(n, 'flops') as Flops
export const bytes = (n: number): Bytes => checked(n, 'bytes') as Bytes
export const seconds = (n: number): Seconds => checked(n, 'seconds') as Seconds
export const flopsPerSecond = (n: number): FlopsPerSecond =>
  checked(n, 'flops/s') as FlopsPerSecond
export const bytesPerSecond = (n: number): BytesPerSecond =>
  checked(n, 'bytes/s') as BytesPerSecond

/** Asserts a scope on a quantity. Use at the point the scope becomes known. */
export const perGpu = <T>(value: T): PerGpu<T> => value as PerGpu<T>
export const cluster = <T>(value: T): Cluster<T> => value as Cluster<T>

/**
 * Converts a cluster-wide total into the share held by one of `n` equal ranks.
 *
 * Named rather than inlined so that dividing by the world size is a visible,
 * greppable act. Every per-GPU quantity in this codebase should be traceable to
 * either a direct per-GPU derivation or a call to this function.
 */
export function share<T extends number>(total: Cluster<T>, ranks: number): PerGpu<T> {
  if (!Number.isInteger(ranks) || ranks < 1) {
    throw new RangeError(`ranks must be a positive integer, got ${ranks}`)
  }
  return (total / ranks) as PerGpu<T>
}

/** Sums the per-rank quantities of `n` equal ranks into a cluster total. */
export function total<T extends number>(each: PerGpu<T>, ranks: number): Cluster<T> {
  if (!Number.isInteger(ranks) || ranks < 1) {
    throw new RangeError(`ranks must be a positive integer, got ${ranks}`)
  }
  return (each * ranks) as Cluster<T>
}

// --- Arithmetic -------------------------------------------------------------
//
// TypeScript's `+` and `*` erase brands, so every operation that should preserve
// one is spelled out. The friction is deliberate: an expression that does not
// fit these helpers is usually an expression whose units do not balance.

export const addBytes = (...xs: readonly Bytes[]): Bytes =>
  xs.reduce((a, b) => (a + b) as Bytes, 0 as Bytes)

export const addFlops = (...xs: readonly Flops[]): Flops =>
  xs.reduce((a, b) => (a + b) as Flops, 0 as Flops)

export const addSeconds = (...xs: readonly Seconds[]): Seconds =>
  xs.reduce((a, b) => (a + b) as Seconds, 0 as Seconds)

export const maxSeconds = (...xs: readonly Seconds[]): Seconds =>
  xs.reduce((a, b) => (a > b ? a : b), 0 as Seconds)

export const scaleBytes = (b: Bytes, k: number): Bytes => (b * checked(k, 'factor')) as Bytes
export const scaleFlops = (f: Flops, k: number): Flops => (f * checked(k, 'factor')) as Flops
export const scaleSeconds = (s: Seconds, k: number): Seconds =>
  (s * checked(k, 'factor')) as Seconds

/** Time to move `b` bytes at `rate`. Zero bytes takes zero time, at any rate. */
export function transferTime(b: Bytes, rate: BytesPerSecond): Seconds {
  if (b === 0) return 0 as Seconds
  if (rate <= 0) throw new RangeError('bandwidth must be positive to move bytes')
  return (b / rate) as Seconds
}

/** Time to execute `f` FLOPs at `rate`. Zero work takes zero time. */
export function computeTime(f: Flops, rate: FlopsPerSecond): Seconds {
  if (f === 0) return 0 as Seconds
  if (rate <= 0) throw new RangeError('throughput must be positive to do work')
  return (f / rate) as Seconds
}

/**
 * Arithmetic intensity.
 *
 * Zero bytes with non-zero FLOPs is infinitely intense — genuinely, not as an
 * error: a kernel that touches no memory is unboundedly compute-bound. Zero
 * FLOPs over zero bytes has no meaningful intensity and is reported as NaN, so
 * a caller that plots it sees a hole rather than a fabricated point.
 */
export function intensity(f: Flops, b: Bytes): FlopsPerByte {
  if (b === 0) return (f === 0 ? Number.NaN : Number.POSITIVE_INFINITY) as FlopsPerByte
  return (f / b) as FlopsPerByte
}

/** Achieved rate implied by doing `f` FLOPs in `t`. */
export function achievedRate(f: Flops, t: Seconds): FlopsPerSecond {
  if (t <= 0) throw new RangeError('elapsed time must be positive to imply a rate')
  return (f / t) as FlopsPerSecond
}

/** Achieved bandwidth implied by moving `b` bytes in `t`. */
export function achievedBandwidth(b: Bytes, t: Seconds): BytesPerSecond {
  if (t <= 0) throw new RangeError('elapsed time must be positive to imply a bandwidth')
  return (b / t) as BytesPerSecond
}

// --- Scale helpers ----------------------------------------------------------
//
// NVIDIA quotes memory and bandwidth in DECIMAL units: an "80 GB" H100 holds
// 80 x 10^9 bytes, and "3.35 TB/s" is 3.35 x 10^12 bytes/s. Model checkpoints
// and `nvidia-smi`, meanwhile, are reported in binary units. Mixing the two is a
// silent 7.4% error at the GB scale, so both are provided and neither is default.

export const GB = (n: number): Bytes => bytes(n * 1e9)
export const TB = (n: number): Bytes => bytes(n * 1e12)
export const GiB = (n: number): Bytes => bytes(n * 1024 ** 3)
export const MiB = (n: number): Bytes => bytes(n * 1024 ** 2)
export const KiB = (n: number): Bytes => bytes(n * 1024)

export const TFLOPS = (n: number): FlopsPerSecond => flopsPerSecond(n * 1e12)
export const PFLOPS = (n: number): FlopsPerSecond => flopsPerSecond(n * 1e15)
export const TFLOP = (n: number): Flops => flops(n * 1e12)
export const PFLOP = (n: number): Flops => flops(n * 1e15)

export const GBps = (n: number): BytesPerSecond => bytesPerSecond(n * 1e9)
export const TBps = (n: number): BytesPerSecond => bytesPerSecond(n * 1e12)

export const microseconds = (n: number): Seconds => seconds(n * 1e-6)
export const milliseconds = (n: number): Seconds => seconds(n * 1e-3)

export const toMilliseconds = (s: Seconds): number => s * 1e3
export const toMicroseconds = (s: Seconds): number => s * 1e6

/**
 * Interconnect capacity is quoted in bits per second (400 Gb/s NDR, 25 GB/s per
 * NVLink lane), and confusing the two is the other classic factor-of-8.
 */
export const bitsToBytes = (bitsPerSecond: number): BytesPerSecond =>
  bytesPerSecond(bitsPerSecond / 8)
