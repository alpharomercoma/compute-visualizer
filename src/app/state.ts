/**
 * Application state, and its encoding into the URL.
 *
 * The whole configuration lives in the hash, which buys two things at once: a
 * scenario is shareable by copying the address bar, and a single-route static
 * site never needs SPA 404 handling. A calculator whose interesting cases cannot
 * be linked to is a calculator nobody quotes.
 */

import type { Dtype } from '../core/hardware'
import type { OptimizerKind, ZeroStage } from '../core/capacity'
import { DEFAULT_PRESET } from '../data/presets'

export type Phase = 'inference' | 'training'

export interface AppState {
  readonly phase: Phase
  readonly modelId: string
  readonly topologyId: string

  readonly tensorParallel: number
  readonly pipelineParallel: number
  readonly dataParallel: number

  readonly batch: number
  readonly promptLength: number
  readonly generateLength: number

  readonly weightDtype: Dtype
  readonly kvDtype: Dtype

  readonly cudaGraph: boolean
  readonly flashAttention: boolean

  // Training only.
  readonly sequenceLength: number
  readonly gradientAccumulation: number
  readonly optimizer: OptimizerKind
  readonly zeroStage: ZeroStage
  readonly sequenceParallel: boolean
  readonly recompute: 'none' | 'selective' | 'full'

  /** Dollars per GPU-hour, for the cost readout. */
  readonly gpuHourCost: number
  /** Draw the vendor's peak alongside the derated roof, for comparison. */
  readonly showTheoretical: boolean
}

export const DEFAULT_STATE: AppState = {
  phase: 'inference',
  modelId: DEFAULT_PRESET.id,
  topologyId: 'h100-x8',
  tensorParallel: 8,
  pipelineParallel: 1,
  dataParallel: 1,
  batch: 32,
  promptLength: 2048,
  generateLength: 256,
  weightDtype: 'bf16',
  kvDtype: 'bf16',
  cudaGraph: true,
  flashAttention: true,
  sequenceLength: 8192,
  gradientAccumulation: 16,
  optimizer: 'adam-mixed',
  /**
   * ZeRO-3: the stage that makes large-model training possible at all.
   *
   * Note this does NOT rescue the default scenario, and nothing does. ZeRO
   * partitions along the data-parallel dimension, so at DP 1 every stage is
   * inert, and once resharded the result is the same either way — a 70B in
   * mixed-precision Adam is 16 bytes per parameter, which is 141 GiB on each of
   * eight H100s however it is divided. Pressing "Training" on the default model
   * correctly reports that it will not fit; what makes that a usable answer is
   * that the controls stay on screen and the search names the models that do.
   */
  zeroStage: 3,
  sequenceParallel: true,
  recompute: 'selective',
  gpuHourCost: 2.5,
  showTheoretical: false,
}

/**
 * Short keys, because the hash is meant to be pasted into a chat window.
 * Anything matching the default is omitted, so a lightly-modified scenario stays
 * short and a reader can see at a glance what was changed.
 */
const KEYS: Record<keyof AppState, string> = {
  phase: 'ph',
  modelId: 'm',
  topologyId: 'hw',
  tensorParallel: 'tp',
  pipelineParallel: 'pp',
  dataParallel: 'dp',
  batch: 'b',
  promptLength: 'in',
  generateLength: 'out',
  weightDtype: 'wd',
  kvDtype: 'kd',
  cudaGraph: 'cg',
  flashAttention: 'fa',
  sequenceLength: 'sl',
  gradientAccumulation: 'ga',
  optimizer: 'opt',
  zeroStage: 'z',
  sequenceParallel: 'sp',
  recompute: 'rc',
  gpuHourCost: 'cost',
  showTheoretical: 'th',
}

export function encodeState(state: AppState): string {
  const params = new URLSearchParams()
  for (const [field, key] of Object.entries(KEYS) as Array<[keyof AppState, string]>) {
    const value = state[field]
    const fallback = DEFAULT_STATE[field]
    if (value === undefined || value === fallback) continue
    params.set(key, typeof value === 'boolean' ? (value ? '1' : '0') : String(value))
  }
  return params.toString()
}

/**
 * The legal values of every enumerated field.
 *
 * Strings from the hash used to be assigned straight through, so `#ph=nope` put
 * an unknown phase into state — which is neither 'inference' nor 'training', so
 * both analyses declined to run and the page showed a verdict-shaped hole. The
 * dtypes are worse: an unrecognised one indexes DTYPE_BYTES to undefined and
 * NaN spreads through the whole model.
 */
const ENUMS: Partial<Record<keyof AppState, readonly string[]>> = {
  phase: ['inference', 'training'],
  topologyId: ['h100-x1', 'h100-x8'],
  weightDtype: ['fp32', 'bf16', 'fp16', 'fp8', 'int8', 'int4'],
  kvDtype: ['fp32', 'bf16', 'fp16', 'fp8', 'int8', 'int4'],
  optimizer: ['adam-mixed', 'adam-fp32-grads', 'sgd-momentum', 'adam-8bit'],
  recompute: ['none', 'selective', 'full'],
}

/**
 * The legal range of every numeric field.
 *
 * `>= 0` was not a range. It let `#b=0` through, which is a batch of nothing,
 * and `#in=1e9`, which is a context nobody has. A link is untrusted input like
 * any other, and every one of these numbers is a divisor somewhere downstream.
 */
const LIMITS: Partial<Record<keyof AppState, { min: number; max: number }>> = {
  tensorParallel: { min: 1, max: 1024 },
  pipelineParallel: { min: 1, max: 1024 },
  dataParallel: { min: 1, max: 1024 },
  batch: { min: 1, max: 512 },
  promptLength: { min: 1, max: 1_048_576 },
  generateLength: { min: 1, max: 1_048_576 },
  sequenceLength: { min: 1, max: 1_048_576 },
  gradientAccumulation: { min: 1, max: 4096 },
  gpuHourCost: { min: 0, max: 10_000 },
  zeroStage: { min: 0, max: 3 },
}

export function decodeState(hash: string): AppState {
  const params = new URLSearchParams(hash.replace(/^#/, ''))
  const next: Record<string, unknown> = { ...DEFAULT_STATE }

  for (const [field, key] of Object.entries(KEYS) as Array<[keyof AppState, string]>) {
    const raw = params.get(key)
    if (raw === null) continue

    const fallback = DEFAULT_STATE[field]
    if (typeof fallback === 'boolean') {
      next[field] = raw === '1'
    } else if (typeof fallback === 'number') {
      const parsed = Number(raw)
      // A malformed link should degrade to the default, not to NaN propagating
      // through the whole model.
      if (!Number.isFinite(parsed)) continue
      const limit = LIMITS[field]
      if (limit && (parsed < limit.min || parsed > limit.max)) continue
      next[field] = parsed
    } else {
      const allowed = ENUMS[field]
      // modelId is the one free-form string: any Hub repo is legitimate, and a
      // bad one is reported by the fetch rather than guessed at here.
      if (allowed && !allowed.includes(raw)) continue
      next[field] = raw
    }
  }

  return normalise(next as unknown as AppState)
}

/**
 * Makes the parallelism describe the hardware it claims to run on.
 *
 * `#hw=h100-x1` used to produce a single H100 running the DEFAULT tensor
 * parallelism of 8 — and because nothing checked the product against the world
 * size, the capacity model dutifully divided the weights by eight and announced
 * that a 70B fits comfortably on one GPU with memory bandwidth as its limit.
 * A confidently wrong answer from a plain URL is the worst failure this tool
 * has, worse than the dead end, because nothing about it looks broken.
 *
 * The hardware select clamped on change, so this was only ever reachable by
 * link — which is exactly how a scenario gets shared.
 */
export function normalise(state: AppState): AppState {
  const gpus = state.topologyId === 'h100-x1' ? 1 : 8

  const tensor = Math.max(1, Math.floor(state.tensorParallel))
  const pipeline = Math.max(1, Math.floor(state.pipelineParallel))
  const data = Math.max(1, Math.floor(state.dataParallel))

  if (tensor * pipeline * data === gpus) return { ...state, tensorParallel: tensor, pipelineParallel: pipeline, dataParallel: data }

  /**
   * Anything that does not account for exactly the GPUs present falls back to
   * pure tensor parallelism.
   *
   * Not to pure data parallelism, which is the other split valid at every world
   * size: data parallelism replicates the model, so going 8 GPUs -> 1 -> 8 would
   * have handed back TP 1 · DP 8 and a 70B that no longer fits. Tensor
   * parallelism is the split that makes a large model fit, so it is the safe
   * thing to land on when the previous choice has stopped making sense.
   */
  return { ...state, tensorParallel: gpus, pipelineParallel: 1, dataParallel: 1 }
}

export function readStateFromLocation(): AppState {
  if (typeof window === 'undefined') return DEFAULT_STATE
  return decodeState(window.location.hash)
}

export function writeStateToLocation(state: AppState): void {
  if (typeof window === 'undefined') return
  const encoded = encodeState(state)
  const url = `${window.location.pathname}${window.location.search}${encoded ? `#${encoded}` : ''}`
  // replaceState rather than pushState: dragging a slider should not fill the
  // back button with every intermediate value.
  window.history.replaceState(null, '', url)
}

/** Valid parallelism splits of a given world size, for the sharding search. */
export function shardingOptions(worldSize: number): Array<{
  tensor: number
  pipeline: number
  data: number
}> {
  const divisors = (n: number) =>
    Array.from({ length: n }, (_, i) => i + 1).filter((d) => n % d === 0)

  const options: Array<{ tensor: number; pipeline: number; data: number }> = []
  for (const tensor of divisors(worldSize)) {
    for (const pipeline of divisors(worldSize / tensor)) {
      const data = worldSize / (tensor * pipeline)
      if (Number.isInteger(data)) options.push({ tensor, pipeline, data })
    }
  }
  return options
}
