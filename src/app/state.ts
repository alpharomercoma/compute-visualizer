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
  /** A config pasted or fetched for a model outside the bundled catalog. */
  readonly customConfig?: string
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
  zeroStage: 1,
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
  customConfig: 'cfg',
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
      if (Number.isFinite(parsed) && parsed >= 0) next[field] = parsed
    } else {
      next[field] = raw
    }
  }

  return next as unknown as AppState
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
