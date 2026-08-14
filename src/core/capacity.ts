/**
 * Memory feasibility, checked before any timing.
 *
 * An out-of-memory configuration has no step time, and reporting a fast one for
 * it is the worst failure this tool could have — it is the exact shape of
 * "confidently wrong" that makes a calculator worse than no calculator. So
 * capacity runs first, and an infeasible configuration returns a verdict rather
 * than a number.
 */

import { DTYPE_BYTES, type Dtype, type Accelerator } from './hardware'
import { type TransformerSpec, kvBytesPerTokenPerLayer, kvHeadsPerRank } from './modelspec'
import { countParameters } from './params'
import { type Bytes, GiB, bytes } from './units'

export type Verdict = 'comfortable' | 'tight' | 'will-not-fit'

export interface MemoryBreakdown {
  readonly weights: Bytes
  readonly kvCache: Bytes
  readonly activations: Bytes
  readonly gradients: Bytes
  readonly optimizerStates: Bytes
  readonly reserve: Bytes
  readonly total: Bytes
}

export interface CapacityReport {
  readonly verdict: Verdict
  readonly perGpu: MemoryBreakdown
  readonly available: Bytes
  readonly headroom: Bytes
  /** Largest context that fits at this batch size, or 0 if nothing does. */
  readonly maxContextLength: number
  /** Largest batch that fits at this context length. */
  readonly maxBatch: number
}

/**
 * CUDA context, cuBLAS workspace, the allocator's slack and the framework
 * itself. Underestimating this is how a tool promises a fit and then the process
 * dies, so it is rounded up.
 */
const RUNTIME_RESERVE = GiB(2)

/** Above this share of available memory, allocator fragmentation starts to bite. */
const TIGHT_FRACTION = 0.9

export interface Parallelism {
  readonly tensor: number
  readonly pipeline: number
  readonly data: number
}

export interface InferenceCapacityInput {
  readonly spec: TransformerSpec
  readonly accelerator: Accelerator
  readonly parallelism: Parallelism
  readonly batch: number
  readonly contextLength: number
  readonly weightDtype: Dtype
  readonly kvDtype: Dtype
}

/**
 * Layers held by the largest pipeline stage.
 *
 * Stages are only equal when `p` divides `L`, and it is the biggest stage that
 * runs out of memory first, so that is what gets sized.
 */
const layersPerStage = (numLayers: number, pipeline: number) => Math.ceil(numLayers / pipeline)

export function inferenceCapacity(input: InferenceCapacityInput): CapacityReport {
  const { spec, accelerator, parallelism: par, batch, contextLength } = input
  const stageLayers = layersPerStage(spec.numLayers, par.pipeline)
  const layerFraction = stageLayers / spec.numLayers

  const counted = countParameters(spec)
  // Weights shard across BOTH tensor and pipeline parallelism: tensor splits
  // each matrix, pipeline splits the layer stack. Omitting the pipeline divisor
  // falsely rejects configurations that fit comfortably.
  const weights = bytes(
    (counted.stored * DTYPE_BYTES[input.weightDtype] * layerFraction) / par.tensor,
  )

  const kvPerTokenPerLayer = kvBytesPerTokenPerLayer(spec, DTYPE_BYTES[input.kvDtype])
  // KV shards by attention head under tensor parallelism, and by layer under
  // pipeline parallelism.
  const kvShard =
    spec.attention.kind === 'mla'
      ? 1 // the latent is not head-sharded
      : kvHeadsPerRank(spec, par.tensor) / spec.attention.numKvHeads
  const kvCache = bytes(kvPerTokenPerLayer * kvShard * contextLength * batch * stageLayers)

  // Inference activations are transient: a few layers' worth of the residual
  // stream plus attention workspace, not the whole stack.
  const activations = bytes(
    4 * batch * Math.max(1, contextLength / 64) * spec.hiddenSize * DTYPE_BYTES[input.kvDtype],
  )

  const zero = bytes(0)
  const total = bytes(weights + kvCache + activations + RUNTIME_RESERVE)
  const available = accelerator.memoryAddressable

  return {
    verdict: verdictFor(total, available),
    perGpu: {
      weights,
      kvCache,
      activations,
      gradients: zero,
      optimizerStates: zero,
      reserve: RUNTIME_RESERVE,
      total,
    },
    available,
    headroom: bytes(Math.max(0, available - total)),
    maxContextLength: solveMaxContext(input, weights, activations, available),
    maxBatch: solveMaxBatch(input, weights, activations, available),
  }
}

function verdictFor(total: Bytes, available: Bytes): Verdict {
  if (total > available) return 'will-not-fit'
  if (total > available * TIGHT_FRACTION) return 'tight'
  return 'comfortable'
}

function kvBytesPerToken(input: InferenceCapacityInput): number {
  const { spec, parallelism: par } = input
  const stageLayers = layersPerStage(spec.numLayers, par.pipeline)
  const perTokenPerLayer = kvBytesPerTokenPerLayer(spec, DTYPE_BYTES[input.kvDtype])
  const shard =
    spec.attention.kind === 'mla'
      ? 1
      : kvHeadsPerRank(spec, par.tensor) / spec.attention.numKvHeads
  return perTokenPerLayer * shard * stageLayers
}

function solveMaxContext(
  input: InferenceCapacityInput,
  weights: Bytes,
  activations: Bytes,
  available: Bytes,
): number {
  const spare = available - weights - activations - RUNTIME_RESERVE
  if (spare <= 0) return 0
  const perToken = kvBytesPerToken(input) * input.batch
  if (perToken <= 0) return input.spec.maxPositionEmbeddings

  const affordable = Math.floor(spare / perToken)
  // Offering more context than the model was trained for produces gibberish,
  // not a longer memory, so the trained limit caps the answer.
  const trained = input.spec.maxPositionEmbeddings || affordable
  return Math.max(0, Math.min(affordable, trained))
}

function solveMaxBatch(
  input: InferenceCapacityInput,
  weights: Bytes,
  activations: Bytes,
  available: Bytes,
): number {
  const spare = available - weights - activations - RUNTIME_RESERVE
  if (spare <= 0) return 0
  const perSequence = kvBytesPerToken(input) * input.contextLength
  if (perSequence <= 0) return Number.POSITIVE_INFINITY
  return Math.max(0, Math.floor(spare / perSequence))
}

export type OptimizerKind = 'adam-mixed' | 'adam-fp32-grads' | 'sgd-momentum' | 'adam-8bit'
export type ZeroStage = 0 | 1 | 2 | 3

/**
 * Bytes per parameter, decomposed as ZeRO does it.
 *
 * Only the first row is paper-sourced (ZeRO section 3.1: 2 + 2 + K where K = 12
 * for mixed-precision Adam, giving 16). The others are framework configurations
 * that circulate widely as if they were citable constants and are not — fp32
 * gradients give 18, an fp32 master plus a bf16 shadow copy gives 20. They are
 * offered as named variants so a user can pick the one matching their setup,
 * and labelled so nobody mistakes them for a specification.
 */
export const OPTIMIZER_BYTES: Record<
  OptimizerKind,
  { weights: number; gradients: number; states: number; source: string }
> = {
  'adam-mixed': {
    weights: 2,
    gradients: 2,
    states: 12,
    source: 'ZeRO section 3.1, verbatim: 2 + 2 + K with K = 12',
  },
  'adam-fp32-grads': {
    weights: 2,
    gradients: 4,
    states: 12,
    source: 'framework configuration (Megatron --accumulate-allreduce-grads-in-fp32), not paper-sourced',
  },
  'sgd-momentum': {
    weights: 2,
    gradients: 2,
    states: 4,
    source: 'ZeRO decomposition with K = 4',
  },
  'adam-8bit': {
    weights: 2,
    gradients: 2,
    states: 6,
    source: 'bitsandbytes 8-bit Adam: 2 bytes of state plus fp32 master, approximate',
  },
}

export interface TrainingCapacityInput {
  readonly spec: TransformerSpec
  readonly accelerator: Accelerator
  readonly parallelism: Parallelism
  readonly microBatch: number
  readonly sequenceLength: number
  readonly optimizer: OptimizerKind
  readonly zeroStage: ZeroStage
  readonly sequenceParallel: boolean
  /**
   * Selective recomputation (Korthikanti eq. 6) discards the attention-square
   * tensors and recomputes them, leaving 34*s*b*h*L/t.
   */
  readonly selectiveRecompute: boolean
  /**
   * Full recomputation keeps only each layer's INPUT and recomputes the entire
   * layer forward during backward. Far cheaper in memory than selective, and it
   * costs an extra forward pass — the classic 1/3 FLOPs surcharge.
   */
  readonly fullRecompute?: boolean
  /**
   * Korthikanti's activation constant. 34 is specific to the GPT-2 shape the
   * paper analysed — dropout enabled, a 2-matrix GeLU MLP, LayerNorm, and no
   * FlashAttention. Modern models differ, so it is a parameter.
   */
  readonly activationConstant?: number
}

export function trainingCapacity(input: TrainingCapacityInput): CapacityReport {
  const { spec, accelerator, parallelism: par, microBatch: b, sequenceLength: s } = input
  const stageLayers = layersPerStage(spec.numLayers, par.pipeline)
  const layerFraction = stageLayers / spec.numLayers

  const counted = countParameters(spec)
  const perParam = OPTIMIZER_BYTES[input.optimizer]
  const shardedParams = (counted.unique * layerFraction) / par.tensor

  // ZeRO partitions across the data-parallel dimension only.
  const dp = par.data
  const stateDivisor = input.zeroStage >= 1 ? dp : 1
  const gradDivisor = input.zeroStage >= 2 ? dp : 1
  const weightDivisor = input.zeroStage >= 3 ? dp : 1

  const weights = bytes((shardedParams * perParam.weights) / weightDivisor)
  const gradients = bytes((shardedParams * perParam.gradients) / gradDivisor)
  const optimizerStates = bytes((shardedParams * perParam.states) / stateDivisor)

  // Korthikanti equation 1, with tensor+sequence parallelism dividing the WHOLE
  // bracket by t (equation 4), not merely the leading constant.
  const c = input.activationConstant ?? 34
  const a = spec.attention.numQueryHeads
  const h = spec.hiddenSize
  const perLayerElements = input.fullRecompute
    ? // Only the layer input survives; everything else is recomputed.
      (s * b * h) / (input.sequenceParallel ? par.tensor : 1)
    : input.selectiveRecompute
      ? (s * b * h * c) / par.tensor
      : input.sequenceParallel
        ? ((s * b * h) / par.tensor) * (c + (5 * a * s) / h)
        : s * b * h * (c / par.tensor + (5 * a * s) / (h * par.tensor))

  // The first pipeline stage must keep activations for `p` in-flight microbatches
  // to stay pressurised, even though it owns only L/p layers.
  const pipelineFactor = par.pipeline > 1 ? par.pipeline : 1
  const activations = bytes(perLayerElements * stageLayers * pipelineFactor)

  const total = bytes(weights + gradients + optimizerStates + activations + RUNTIME_RESERVE)
  const available = accelerator.memoryAddressable

  return {
    verdict: verdictFor(total, available),
    perGpu: {
      weights,
      kvCache: bytes(0),
      activations,
      gradients,
      optimizerStates,
      reserve: RUNTIME_RESERVE,
      total,
    },
    available,
    headroom: bytes(Math.max(0, available - total)),
    maxContextLength: 0,
    maxBatch: 0,
  }
}

/** A short, honest sentence about a verdict. */
export function explainVerdict(report: CapacityReport): string {
  const gb = (b: number) => `${(b / GiB(1)).toFixed(1)} GiB`
  switch (report.verdict) {
    case 'will-not-fit':
      return `Needs ${gb(report.perGpu.total)} per GPU against ${gb(report.available)} available. This configuration will not run.`
    case 'tight':
      return `Needs ${gb(report.perGpu.total)} of ${gb(report.available)}, leaving ${gb(report.headroom)}. Allocator fragmentation may still defeat it.`
    case 'comfortable':
      return `Needs ${gb(report.perGpu.total)} of ${gb(report.available)}, leaving ${gb(report.headroom)}.`
  }
}

export const RESERVE_BYTES = RUNTIME_RESERVE
