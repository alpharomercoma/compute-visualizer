/**
 * Training: step time, MFU versus HFU, and time to train.
 *
 * The structural difference from inference is not that backward costs 2x — it is
 * that data-parallel gradient reduction genuinely overlaps compute. A bucket's
 * all-reduce depends only on its own backward kernel, so it runs while lower
 * layers are still computing and only the tail is exposed. That falls out of the
 * scheduler from the dependency edges; it is not a special case here.
 */

import {
  type OptimizerKind,
  type Parallelism,
  type ZeroStage,
  type CapacityReport,
  OPTIMIZER_BYTES,
  trainingCapacity,
} from './capacity'
import {
  type CollectiveProfile,
  algorithmicMessage,
  chooseCollective,
  defaultProfiles,
} from './collectives'
import type { EfficiencyModel, OverheadModel } from './efficiency'
import { DTYPE_BYTES, type Dtype, type TensorPrecision, type Topology, DTYPE_COMPUTES_AS } from './hardware'
import { largestStage } from './inference'
import { type Workload, layerKernels, lmHead, modelFlops } from './kernels'
import type { TransformerSpec } from './modelspec'
import { countParameters, nonEmbeddingParameters } from './params'
import { type BottleneckClass, type LimitedBy, type ScheduleResult, type Task, schedule } from './schedule'
import { type Flops, type Seconds, bytes, flops, seconds } from './units'

export interface TrainingConfig {
  readonly spec: TransformerSpec
  readonly topology: Topology
  readonly parallelism: Parallelism
  readonly microBatch: number
  readonly sequenceLength: number
  /** Microbatches per optimizer step, per data-parallel replica. */
  readonly gradientAccumulation: number
  readonly weightDtype: Dtype
  readonly activationDtype: Dtype
  readonly optimizer: OptimizerKind
  readonly zeroStage: ZeroStage
  readonly sequenceParallel: boolean
  readonly selectiveRecompute: boolean
  readonly fullRecompute: boolean
  readonly efficiency: EfficiencyModel
  readonly overhead: OverheadModel
  readonly collectiveProfiles?: readonly CollectiveProfile[]
  /** Interleaved pipeline chunks per device. Reduces the bubble by this factor. */
  readonly pipelineChunks?: number
}

export interface TrainingResult {
  readonly capacity: CapacityReport
  readonly feasible: boolean
  readonly stepTime?: Seconds
  readonly globalBatchTokens?: number
  /** Model FLOPs: what the architecture requires, ignoring recomputation. */
  readonly modelFlopsPerStep?: Flops
  /** Hardware FLOPs: what the GPUs actually execute, including recomputation. */
  readonly hardwareFlopsPerStep?: Flops
  readonly mfu?: number
  readonly hfu?: number
  readonly tokensPerSecond?: number
  readonly tokensPerDay?: number
  readonly bottleneck?: BottleneckClass
  /** The scheduled microbatch, for the time-breakdown readout. */
  readonly schedule?: ScheduleResult
  /** Fraction of the step lost to the pipeline bubble. */
  readonly bubbleFraction?: number
  readonly exposedCommunication?: Seconds
}

const computePrecision = (weight: Dtype, activation: Dtype): TensorPrecision =>
  weight === 'int4' || weight === 'int8'
    ? DTYPE_COMPUTES_AS[activation]
    : DTYPE_COMPUTES_AS[weight]

/**
 * Backward costs twice forward.
 *
 * Every GEMM runs three times per step: once forward, then once for the data
 * gradient and once for the weight gradient. Megatron-LM's own accounting states
 * it exactly that way.
 */
const BACKWARD_MULTIPLIER = 3

export function analyseTraining(config: TrainingConfig): TrainingResult {
  const capacity = trainingCapacity({
    spec: config.spec,
    accelerator: config.topology.accelerator,
    parallelism: config.parallelism,
    microBatch: config.microBatch,
    sequenceLength: config.sequenceLength,
    optimizer: config.optimizer,
    zeroStage: config.zeroStage,
    sequenceParallel: config.sequenceParallel,
    selectiveRecompute: config.selectiveRecompute,
    fullRecompute: config.fullRecompute,
  })

  if (capacity.verdict === 'will-not-fit') {
    return { capacity, feasible: false }
  }

  const { spec, parallelism: par, topology } = config
  const { accelerator } = topology
  const precision = computePrecision(config.weightDtype, config.activationDtype)
  const peak = accelerator.peak[precision]

  const workload: Workload = {
    spec,
    batch: config.microBatch,
    queryLength: config.sequenceLength,
    contextLength: config.sequenceLength,
    tensorParallel: par.tensor,
    weightDtype: config.weightDtype,
    kvDtype: config.activationDtype,
    activationDtype: config.activationDtype,
    flashAttention: true,
    causal: true,
  }

  const profiles =
    config.collectiveProfiles ?? defaultProfiles(par.tensor, topology.fabric === 'nvswitch')

  // The largest stage sets the pace; ceil() for every stage would model layers
  // that do not exist whenever the pipeline degree does not divide the model.
  const stageLayers = largestStage(spec.numLayers, par.pipeline)
  const tokens = config.microBatch * config.sequenceLength

  // Full recomputation runs every layer's forward pass a second time, so it
  // multiplies: 3x becomes 4x.
  //
  // SELECTIVE recomputation does not multiply. It re-runs only the attention
  // tensors it discarded, which is an additive term of 4·B·s²·h per layer and
  // is applied to the attention kernel alone below. An earlier version charged
  // a blanket 1.1x across every kernel including the MLP, norms and LM head —
  // an invented constant, wrong at every s/h, and exactly the kind of number
  // this project refuses to ship elsewhere.
  const forwardRepeats = config.fullRecompute ? 4 / 3 : 1

  const tasks: Task[] = []
  let previous: string | undefined
  let forwardBackwardFlops = 0
  let hbmBytes = 0

  for (let layer = 0; layer < stageLayers; layer++) {
    for (const kernel of layerKernels(workload, layer)) {
      // Selective recompute adds one extra forward of the attention kernel and
      // nothing else.
      const selectiveExtra =
        config.selectiveRecompute && !config.fullRecompute && kernel.kernelClass === 'attention'
          ? kernel.flops
          : 0

      const scaled = kernel.flops * BACKWARD_MULTIPLIER * forwardRepeats + selectiveExtra
      const scaledBytes = kernel.hbmBytes * BACKWARD_MULTIPLIER
      forwardBackwardFlops += scaled
      hbmBytes += scaledBytes

      const gemmEfficiency =
        tokens < 512 ? config.efficiency.gemmSkinny.value : config.efficiency.gemmLarge.value
      const bandwidthEfficiency = kernel.isGather
        ? config.efficiency.hbmGather.value
        : config.efficiency.hbmStreaming.value

      const computeTime = scaled / (peak * gemmEfficiency)
      const memoryTime = scaledBytes / (accelerator.memoryBandwidth * bandwidthEfficiency)
      const launch = config.overhead.launchSeconds
      const duration = Math.max(computeTime, memoryTime, launch)
      const limitedBy: LimitedBy =
        duration === launch ? 'overhead' : computeTime >= memoryTime ? 'compute' : 'memory'

      const id = `L${layer}.${kernel.kernelClass}`
      tasks.push({
        id,
        label: `layer ${layer} ${kernel.name}`,
        deps: previous ? [previous] : [],
        resource: 'gpu',
        duration: seconds(duration),
        limitedBy,
      })
      previous = id
    }

    // Tensor-parallel all-reduces, twice forward and twice backward.
    if (par.tensor > 1) {
      const message = algorithmicMessage(
        bytes(tokens * spec.hiddenSize * DTYPE_BYTES[config.activationDtype]),
      )
      for (let i = 0; i < 4; i++) {
        const choice = chooseCollective(message, profiles)
        const id = `L${layer}.tp${i}`
        tasks.push({
          id,
          label: `layer ${layer} TP all-reduce`,
          deps: previous ? [previous] : [],
          resource: 'network',
          duration: choice.time,
          limitedBy: choice.latencyBound ? 'network-latency' : 'network-bandwidth',
        })
        previous = id
      }
    }
  }

  const head = lmHead(workload)
  const headFlops = head.flops * BACKWARD_MULTIPLIER
  forwardBackwardFlops += headFlops
  hbmBytes += head.hbmBytes * BACKWARD_MULTIPLIER
  tasks.push({
    id: 'lm-head',
    label: head.name,
    deps: previous ? [previous] : [],
    resource: 'gpu',
    duration: seconds(headFlops / (peak * config.efficiency.gemmLarge.value)),
    limitedBy: 'compute',
  })

  const scheduled = schedule(tasks)

  /**
   * The data-parallel gradient reduction.
   *
   * It runs ONCE per optimizer step, not once per microbatch: gradient
   * accumulation sums locally across microbatches and only the total is
   * reduced. Putting it inside the microbatch schedule — which an earlier
   * version did — charged it `m` times over.
   *
   * It is also the reason DP training is tolerable while TP decode is not.
   * Gradients for the upper layers are ready before backward reaches the bottom,
   * so their buckets reduce while lower layers are still computing. Only the
   * part that outlives the final microbatch's backward pass extends the step.
   */
  let exposedCommunication = seconds(0)
  if (par.data > 1) {
    const counted = countParameters(spec)
    const shardedParams = (counted.stored * (stageLayers / spec.numLayers)) / par.tensor
    // ZeRO-3 adds parameter all-gathers in both passes: 3 psi against 2 psi.
    const volumeFactor = config.zeroStage >= 3 ? 1.5 : 1
    const totalGradientBytes = shardedParams * OPTIMIZER_BYTES[config.optimizer].gradients

    const buckets = 8
    const perBucket = algorithmicMessage(bytes((totalGradientBytes * volumeFactor) / buckets))
    const bucketTime = chooseCollective(perBucket, profiles).time
    const allBuckets = bucketTime * buckets

    // Every bucket but the last has a backward pass still running behind it to
    // hide under; the last one has nothing left and is exposed in full.
    const hideableWindow = scheduled.makespan * ((buckets - 1) / buckets)
    exposedCommunication = seconds(Math.max(bucketTime, allBuckets - hideableWindow))
  }

  // Pipeline bubble: (p-1)/m, reduced by the interleaving factor when used.
  const microbatches = config.gradientAccumulation
  const chunks = config.pipelineChunks ?? 1
  const bubbleFraction =
    par.pipeline > 1 ? (par.pipeline - 1) / (microbatches * chunks) : 0

  const perMicrobatch = scheduled.makespan
  // The gradient reduction is added once, after every microbatch has run.
  const stepTime = seconds(
    perMicrobatch * microbatches * (1 + bubbleFraction) + exposedCommunication,
  )

  const nonEmbedding = nonEmbeddingParameters(spec)
  // Model FLOPs: forward plus backward, no recomputation, whole cluster.
  const perMicrobatchModelFlops = modelFlops(
    spec,
    nonEmbedding,
    {
      batch: config.microBatch,
      queryLength: config.sequenceLength,
      contextLength: config.sequenceLength,
      causal: true,
    },
    tokens,
  )
  const replicas = par.data
  const modelFlopsPerStep = flops(
    perMicrobatchModelFlops * BACKWARD_MULTIPLIER * microbatches * replicas,
  )
  // Hardware FLOPs: what actually ran, recomputation included.
  const hardwareFlopsPerStep = flops(
    forwardBackwardFlops * par.tensor * par.pipeline * microbatches * replicas,
  )

  const totalGpus = par.tensor * par.pipeline * par.data
  const clusterPeak = peak * totalGpus
  const globalBatchTokens = tokens * microbatches * replicas

  return {
    capacity,
    feasible: true,
    stepTime,
    globalBatchTokens,
    modelFlopsPerStep,
    hardwareFlopsPerStep,
    // MFU and HFU share a denominator and differ only in the numerator. HFU is
    // always the larger, and full recomputation raises HFU while lowering MFU --
    // which is why PaLM argued MFU is the throughput-honest metric.
    mfu: modelFlopsPerStep / (clusterPeak * stepTime),
    hfu: hardwareFlopsPerStep / (clusterPeak * stepTime),
    tokensPerSecond: globalBatchTokens / stepTime,
    tokensPerDay: (globalBatchTokens / stepTime) * 86400,
    bottleneck: scheduled.bottleneck,
    schedule: scheduled,
    bubbleFraction,
    exposedCommunication,
  }
}

/**
 * Days to train a given token budget at a measured or modelled throughput.
 *
 * Chinchilla-optimal is roughly 20 tokens per parameter; the caller supplies the
 * budget rather than having one assumed for them.
 */
export function timeToTrain(tokensPerSecond: number, tokenBudget: number): number {
  if (!(tokensPerSecond > 0)) return Number.POSITIVE_INFINITY
  return tokenBudget / tokensPerSecond / 86400
}

/** The 6ND estimate, for comparison against the structural model. */
export function sixNDFlops(nonEmbeddingParams: number, tokens: number): Flops {
  return flops(6 * nonEmbeddingParams * tokens)
}
