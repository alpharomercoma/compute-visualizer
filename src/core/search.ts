/**
 * Searching the configuration space.
 *
 * Two jobs, both of which turn the calculator into an explainer:
 *
 *   - the best way to shard a model across the GPUs you have, and why the
 *     runner-up loses;
 *   - the crossovers — the batch size where decode stops being memory-bound,
 *     the tensor-parallel degree past which latency eats the gains.
 *
 * A number is useful; the threshold where the number changes character is what
 * someone actually remembers.
 */

import { type InferenceConfig, type InferenceResult, analyseInference } from './inference'
import type { BottleneckClass } from './schedule'

export interface ShardingCandidate {
  readonly tensor: number
  readonly pipeline: number
  readonly data: number
  readonly feasible: boolean
  readonly tokensPerSecond?: number
  readonly timePerOutputToken?: number
  readonly bottleneck?: BottleneckClass
}

/** Every split of `worldSize` into tensor x pipeline x data. */
export function shardingSplits(worldSize: number): Array<{
  tensor: number
  pipeline: number
  data: number
}> {
  const divisors = (n: number) =>
    Array.from({ length: n }, (_, i) => i + 1).filter((d) => n % d === 0)

  const splits: Array<{ tensor: number; pipeline: number; data: number }> = []
  for (const tensor of divisors(worldSize)) {
    for (const pipeline of divisors(worldSize / tensor)) {
      splits.push({ tensor, pipeline, data: worldSize / (tensor * pipeline) })
    }
  }
  return splits
}

export type Objective = 'throughput' | 'latency'

/**
 * Ranks every valid sharding of a fixed number of GPUs.
 *
 * The search space is tiny — 8 GPUs admit ten splits — so this is exhaustive
 * rather than clever. Infeasible splits are kept in the results rather than
 * filtered out, because "that one does not fit" is an answer too.
 */
export function rankSharding(
  base: InferenceConfig,
  worldSize: number,
  objective: Objective = 'throughput',
): ShardingCandidate[] {
  const candidates = shardingSplits(worldSize).map((split): ShardingCandidate => {
    let result: InferenceResult
    try {
      result = analyseInference({ ...base, parallelism: split })
    } catch {
      return { ...split, feasible: false }
    }

    if (!result.feasible) return { ...split, feasible: false }

    return {
      ...split,
      feasible: true,
      tokensPerSecond: result.aggregateTokensPerSecond ?? 0,
      timePerOutputToken: result.timePerOutputToken ?? 0,
      bottleneck: result.decode?.bottleneck ?? 'compute',
    }
  })

  return candidates.sort((a, b) => {
    if (a.feasible !== b.feasible) return a.feasible ? -1 : 1
    if (!a.feasible) return 0
    return objective === 'throughput'
      ? (b.tokensPerSecond ?? 0) - (a.tokensPerSecond ?? 0)
      : (a.timePerOutputToken ?? 0) - (b.timePerOutputToken ?? 0)
  })
}

export interface Crossover {
  readonly label: string
  readonly value: number
  readonly unit: string
  readonly explanation: string
}

/**
 * Finds the batch size where the decode bottleneck changes class.
 *
 * Binary search over a monotone-ish predicate. Not perfectly monotone in
 * pathological corners — capacity can fail before the bottleneck flips — so the
 * search reports failure rather than a fabricated crossing when the endpoints do
 * not bracket a change.
 */
export function findBottleneckCrossover(
  base: InferenceConfig,
  from: BottleneckClass,
  maxBatch = 4096,
): number | undefined {
  const classOf = (batch: number): BottleneckClass | undefined => {
    try {
      const result = analyseInference({ ...base, batch })
      return result.feasible ? result.decode?.bottleneck : undefined
    } catch {
      return undefined
    }
  }

  let low = 1
  let high = maxBatch

  if (classOf(low) !== from) return undefined
  const highClass = classOf(high)
  if (highClass === from || highClass === undefined) return undefined

  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2)
    if (classOf(mid) === from) low = mid
    else high = mid
  }
  return high
}

/**
 * The thresholds worth knowing for a configuration.
 *
 * Closed forms where one exists; a search only where it does not.
 */
export function crossovers(
  base: InferenceConfig,
  ridgePointFlopsPerByte: number,
  weightBytes: number,
  queryHeads: number,
  kvHeads: number,
  kvBytes: number,
): Crossover[] {
  const out: Crossover[] = []

  // Weight-stationary GEMM intensity is 2b/w, so it meets the ridge at
  // ridge * w / 2.
  out.push({
    label: 'Decode leaves the memory roof',
    value: (ridgePointFlopsPerByte * weightBytes) / 2,
    unit: 'batch',
    explanation:
      'Where the linear layers reach the machine balance. Attention does not follow: its intensity is fixed by the head ratio and never moves with batch.',
  })

  out.push({
    label: 'Decode attention intensity',
    value: (2 * queryHeads) / (kvHeads * kvBytes),
    unit: 'FLOP/byte',
    explanation:
      'Fixed for the architecture. Only GQA, MLA or a smaller KV dtype move it — batching cannot, because every sequence carries its own cache.',
  })

  const memoryToCompute = findBottleneckCrossover(base, 'memory-bandwidth')
  if (memoryToCompute !== undefined) {
    out.push({
      label: 'Bottleneck changes class',
      value: memoryToCompute,
      unit: 'batch',
      explanation: 'Measured by search rather than closed form, because capacity and collectives both move with batch.',
    })
  }

  return out
}
