/**
 * Inference: prefill, decode, and the metrics people actually quote.
 *
 * Both phases are built as task DAGs and handed to the scheduler, so the
 * familiar asymmetry — prefill compute-bound, decode memory-bound — is an
 * OUTPUT of the model rather than an assumption baked into it.
 */

import {
  type CapacityReport,
  type Parallelism,
  inferenceCapacity,
} from './capacity'
import {
  type CollectiveProfile,
  algorithmicMessage,
  chooseCollective,
  defaultProfiles,
} from './collectives'
import { type EfficiencyModel, type OverheadModel } from './efficiency'
import {
  DTYPE_BYTES,
  type Dtype,
  type TensorPrecision,
  type Topology,
  DTYPE_COMPUTES_AS,
} from './hardware'
import { type Kernel, type Workload, layerKernels, lmHead, modelFlops } from './kernels'
import type { TransformerSpec } from './modelspec'
import { nonEmbeddingParameters } from './params'
import {
  type BottleneckClass,
  type LimitedBy,
  type ScheduleResult,
  type Task,
  schedule,
} from './schedule'
import {
  type Bytes,
  type Flops,
  type FlopsPerByte,
  type Seconds,
  bytes,
  bytesPerSecond,
  flops,
  flopsPerSecond,
  intensity,
  seconds,
} from './units'

export interface InferenceConfig {
  readonly spec: TransformerSpec
  readonly topology: Topology
  readonly parallelism: Parallelism
  readonly batch: number
  readonly promptLength: number
  readonly generateLength: number
  readonly weightDtype: Dtype
  readonly kvDtype: Dtype
  readonly activationDtype: Dtype
  readonly efficiency: EfficiencyModel
  readonly overhead: OverheadModel
  /** CUDA graph capture removes most per-kernel launch cost. */
  readonly cudaGraph: boolean
  readonly flashAttention: boolean
  /** Override the collective priors, e.g. from the calibration panel. */
  readonly collectiveProfiles?: readonly CollectiveProfile[]
}

/**
 * One kernel class, aggregated over the whole step.
 *
 * The roofline plots these, never a single whole-step point: aggregating FLOPs
 * and bytes across classes of differing intensity lands the result near the
 * ridge and implies a utilisation that never occurs.
 */
export interface ClassAggregate {
  readonly label: string
  readonly kernelClass: string
  readonly flops: Flops
  readonly hbmBytes: Bytes
  readonly seconds: Seconds
}

export interface PhaseResult {
  readonly schedule: ScheduleResult
  readonly flopsPerRank: Flops
  readonly hbmBytesPerRank: Bytes
  readonly collectiveBytes: Bytes
  readonly arithmeticIntensity: FlopsPerByte
  readonly bottleneck: BottleneckClass
  readonly classes: readonly ClassAggregate[]
}

export interface InferenceResult {
  readonly capacity: CapacityReport
  readonly feasible: boolean
  readonly prefill?: PhaseResult
  readonly decode?: PhaseResult

  /** Time to first token: one prefill of the whole batch. */
  readonly timeToFirstToken?: Seconds
  /** Time per output token, once generating. NVIDIA calls this ITL. */
  readonly timePerOutputToken?: Seconds
  readonly endToEndLatency?: Seconds
  readonly tokensPerSecondPerUser?: number
  readonly aggregateTokensPerSecond?: number

  /** Model FLOPs utilisation over the decode phase. Low here is correct. */
  readonly decodeMfu?: number
  /** Memory bandwidth utilisation. The metric that carries information in decode. */
  readonly decodeMbu?: number
  readonly prefillMfu?: number
}

/**
 * The precision the tensor cores actually run at.
 *
 * Weight-only 4-bit dequantizes to 16-bit before the tensor core sees it, so
 * W4A16 moves a quarter of the bytes at unchanged FLOPs. That asymmetry is
 * exactly what shifts decode off the memory roof, and it only shows up if
 * storage precision and compute precision are tracked separately.
 */
const computePrecision = (weight: Dtype, activation: Dtype): TensorPrecision =>
  weight === 'int4' || weight === 'int8'
    ? DTYPE_COMPUTES_AS[activation]
    : DTYPE_COMPUTES_AS[weight]

/**
 * Skinny GEMMs reach a materially lower fraction of peak than large square ones:
 * tiles are poorly filled and tail waves dominate. Decode is always skinny.
 */
const SKINNY_TOKEN_THRESHOLD = 512

function kernelDuration(
  kernel: Kernel,
  config: InferenceConfig,
  tokens: number,
): { duration: Seconds; limitedBy: LimitedBy } {
  const { accelerator } = config.topology
  const precision = computePrecision(config.weightDtype, config.activationDtype)
  const peak = accelerator.peak[precision]

  const gemmEfficiency =
    tokens < SKINNY_TOKEN_THRESHOLD
      ? config.efficiency.gemmSkinny.value
      : config.efficiency.gemmLarge.value
  const bandwidthEfficiency = kernel.isGather
    ? config.efficiency.hbmGather.value
    : config.efficiency.hbmStreaming.value

  const computeTime = kernel.flops / (peak * gemmEfficiency)
  const memoryTime = kernel.hbmBytes / (accelerator.memoryBandwidth * bandwidthEfficiency)
  const launch = config.cudaGraph
    ? config.overhead.graphLaunchSeconds
    : config.overhead.launchSeconds

  // Within one kernel, compute and memory genuinely overlap in hardware, so the
  // duration is the larger of the two -- never their sum.
  const duration = Math.max(computeTime, memoryTime, launch)
  const limitedBy: LimitedBy =
    duration === launch ? 'overhead' : computeTime >= memoryTime ? 'compute' : 'memory'

  return { duration: seconds(duration), limitedBy }
}

/**
 * Builds one forward pass as a task DAG.
 *
 * Tensor parallelism inserts two all-reduces per layer — after the attention
 * output projection and after the MLP down projection — and each feeds the next
 * kernel directly. That dependency is what puts them on the critical path in
 * decode, and it is stated here rather than assumed anywhere downstream.
 */
function buildForwardPass(config: InferenceConfig, queryLength: number, contextLength: number) {
  const { spec, parallelism: par } = config
  const tokens = config.batch * queryLength

  const workload: Workload = {
    spec,
    batch: config.batch,
    queryLength,
    contextLength,
    tensorParallel: par.tensor,
    weightDtype: config.weightDtype,
    kvDtype: config.kvDtype,
    activationDtype: config.activationDtype,
    flashAttention: config.flashAttention,
    // Prefill benefits from the causal mask; a decode query attends everything.
    causal: queryLength > 1,
  }

  const profiles =
    config.collectiveProfiles ??
    defaultProfiles(par.tensor, config.topology.fabric === 'nvswitch')

  // The all-reduce payload is the full hidden state for every token in flight.
  const messageBytes = algorithmicMessage(
    bytes(tokens * spec.hiddenSize * DTYPE_BYTES[config.activationDtype]),
  )
  const needsCollectives = par.tensor > 1

  const tasks: Task[] = []
  const kernelsSeen: Kernel[] = []
  let previous: string | undefined
  let collectiveBytes = 0

  // The largest stage: it sets the pipeline's pace and is the one that OOMs.
  const stageLayers = largestStage(spec.numLayers, par.pipeline)

  for (let layer = 0; layer < stageLayers; layer++) {
    const kernels = layerKernels(workload, layer)

    kernels.forEach((kernel, index) => {
      const id = `L${layer}.${kernel.kernelClass}`
      const { duration, limitedBy } = kernelDuration(kernel, config, tokens)
      tasks.push({
        id,
        label: `layer ${layer} ${kernel.name}`,
        deps: previous ? [previous] : [],
        resource: 'gpu',
        duration,
        limitedBy,
      })
      kernelsSeen.push(kernel)
      previous = id

      // An all-reduce follows the output projection and the MLP, because both
      // are row-parallel and produce partial sums.
      const needsReduce =
        needsCollectives &&
        (kernel.kernelClass === 'output-projection' || kernel.kernelClass === 'mlp')
      if (needsReduce) {
        const choice = chooseCollective(messageBytes, profiles)
        const id2 = `L${layer}.allreduce${index}`
        tasks.push({
          id: id2,
          label: `layer ${layer} all-reduce`,
          deps: [id],
          resource: 'network',
          duration: choice.time,
          limitedBy: choice.latencyBound ? 'network-latency' : 'network-bandwidth',
        })
        collectiveBytes += messageBytes
        previous = id2
      }
    })
  }

  // Serving needs one logit row per sequence, not one per prompt token. And the
  // head lives on the FINAL pipeline stage only — running it on every stage and
  // then multiplying by the stage count charges it `p` times over.
  const head = lmHead(workload, config.batch)
  const headTiming = kernelDuration(head, config, tokens)
  tasks.push({
    id: 'lm-head',
    label: head.name,
    deps: previous ? [previous] : [],
    resource: 'gpu',
    // Charged once across the whole traversal: pipelineTraversal multiplies the
    // stage time by p, so the head is divided by p here to survive that
    // multiplication exactly once.
    duration: seconds(headTiming.duration / par.pipeline),
    limitedBy: headTiming.limitedBy,
  })
  kernelsSeen.push(head)

  return { tasks, kernels: kernelsSeen, collectiveBytes: bytes(collectiveBytes) }
}

/**
 * Wall-clock for one token to traverse the whole model.
 *
 * `buildForwardPass` covers only the layers one pipeline stage owns, so its
 * makespan is a stage time, not a step time. A token must pass through every
 * stage in turn, with a point-to-point send of the hidden state between each
 * pair. Treating the stage time as the step time underpredicts latency by
 * roughly the pipeline degree.
 *
 * This is latency, not throughput: microbatching lets stages work concurrently
 * and recovers throughput, but no token gets its answer before crossing all of
 * them.
 */
function pipelineTraversal(
  stageTime: Seconds,
  config: InferenceConfig,
  tokens: number,
): Seconds {
  const p = config.parallelism.pipeline
  if (p <= 1) return stageTime

  // Each boundary ships the hidden state for the tokens in flight.
  const payload = bytes(tokens * config.spec.hiddenSize * DTYPE_BYTES[config.activationDtype])
  const link = config.topology.accelerator.interconnect.perDirection
  const hop = payload / link + config.overhead.launchSeconds

  return seconds(stageTime * p + hop * (p - 1))
}

/**
 * Layers owned by a given pipeline stage.
 *
 * `ceil(L/p)` for every stage overcounts whenever `p` does not divide `L` — 80
 * layers over 6 stages becomes 6 × 14 = 84 — which inflates latency, FLOPs and
 * capacity together. This partition sums to exactly `L`.
 */
export function layersInStage(numLayers: number, pipeline: number, stage: number): number {
  return (
    Math.floor((numLayers * (stage + 1)) / pipeline) - Math.floor((numLayers * stage) / pipeline)
  )
}

/** The largest stage, which is the one that runs out of memory first. */
export function largestStage(numLayers: number, pipeline: number): number {
  let largest = 0
  for (let stage = 0; stage < pipeline; stage++) {
    largest = Math.max(largest, layersInStage(numLayers, pipeline, stage))
  }
  return largest
}

function runPhase(config: InferenceConfig, queryLength: number, contextLength: number): PhaseResult {
  const { tasks, kernels, collectiveBytes } = buildForwardPass(config, queryLength, contextLength)
  const result = schedule(tasks)

  const totalFlops = flops(kernels.reduce((sum, k) => sum + k.flops, 0))
  const totalBytes = bytes(kernels.reduce((sum, k) => sum + k.hbmBytes, 0))

  // Roll the per-layer kernels up by class, and pair each with the time the
  // schedule actually gave it.
  const durations = new Map<string, number>()
  for (const task of result.tasks) {
    if (task.resource !== 'gpu') continue
    const key = task.id.includes('.') ? task.id.slice(task.id.indexOf('.') + 1) : task.id
    durations.set(key, (durations.get(key) ?? 0) + task.duration)
  }

  const totals = new Map<string, { label: string; flops: number; bytes: number }>()
  for (const kernel of kernels) {
    const entry = totals.get(kernel.kernelClass) ?? {
      label: kernel.name,
      flops: 0,
      bytes: 0,
    }
    entry.flops += kernel.flops
    entry.bytes += kernel.hbmBytes
    totals.set(kernel.kernelClass, entry)
  }

  const classes: ClassAggregate[] = [...totals.entries()].map(([kernelClass, entry]) => ({
    label: entry.label,
    kernelClass,
    flops: flops(entry.flops),
    hbmBytes: bytes(entry.bytes),
    seconds: seconds(durations.get(kernelClass) ?? 0),
  }))

  return {
    schedule: result,
    flopsPerRank: totalFlops,
    hbmBytesPerRank: totalBytes,
    collectiveBytes,
    arithmeticIntensity: intensity(totalFlops, totalBytes),
    bottleneck: result.bottleneck,
    classes,
  }
}

export function analyseInference(config: InferenceConfig): InferenceResult {
  const capacity = inferenceCapacity({
    spec: config.spec,
    accelerator: config.topology.accelerator,
    parallelism: config.parallelism,
    batch: config.batch,
    contextLength: config.promptLength + config.generateLength,
    weightDtype: config.weightDtype,
    kvDtype: config.kvDtype,
    activationDtype: config.activationDtype,
  })

  // An infeasible configuration gets a verdict, not a time. Reporting a fast
  // step for a model that cannot load is the worst thing this tool could do.
  if (capacity.verdict === 'will-not-fit') {
    return { capacity, feasible: false }
  }

  const prefill = runPhase(config, config.promptLength, config.promptLength)

  // Decode cost grows with context, so the midpoint of the generation is the
  // representative step. Prefill produces token 1; the decode steps counted in
  // end-to-end latency produce tokens 2..n at contexts prompt+1..prompt+n-1,
  // whose mean is prompt + n/2.
  const meanContext = config.promptLength + Math.max(0, config.generateLength) / 2
  const decode = runPhase(config, 1, Math.round(meanContext))

  // A token must cross every pipeline stage; a stage makespan is not a step time.
  const ttft = pipelineTraversal(
    prefill.schedule.makespan,
    config,
    config.batch * config.promptLength,
  )
  const tpot = pipelineTraversal(decode.schedule.makespan, config, config.batch)
  const outputTokens = Math.max(1, config.generateLength)
  const endToEnd = seconds(ttft + tpot * (outputTokens - 1))

  const { accelerator } = config.topology
  const precision = computePrecision(config.weightDtype, config.activationDtype)
  // Ranks cooperating on one replica: tensor and pipeline degrees.
  const replicaRanks = config.parallelism.tensor * config.parallelism.pipeline
  const replicaPeak = flopsPerSecond(accelerator.peak[precision] * replicaRanks)
  const replicaBandwidth = bytesPerSecond(accelerator.memoryBandwidth * replicaRanks)

  // MFU's numerator is MODEL FLOPs — what the architecture requires — not the
  // per-rank kernel sum scaled by rank count. The latter is hardware FLOPs, and
  // includes work every rank redundantly repeats, so using it would credit a
  // distributed system for its own duplication.
  const nonEmbedding = nonEmbeddingParameters(config.spec)
  const decodeModelFlops = modelFlops(
    config.spec,
    nonEmbedding,
    { batch: config.batch, queryLength: 1, contextLength: Math.round(meanContext), causal: false },
    config.batch,
  )
  const prefillModelFlops = modelFlops(
    config.spec,
    nonEmbedding,
    {
      batch: config.batch,
      queryLength: config.promptLength,
      contextLength: config.promptLength,
      causal: true,
    },
    config.batch,
  )
  // MBU, by contrast, is about real traffic, so hardware bytes are correct here.
  const decodeBytes = bytes(decode.hbmBytesPerRank * replicaRanks)

  const replicas = Math.max(1, config.parallelism.data)

  return {
    capacity,
    feasible: true,
    prefill,
    decode,
    timeToFirstToken: ttft,
    timePerOutputToken: tpot,
    endToEndLatency: endToEnd,
    tokensPerSecondPerUser: 1 / tpot,
    // Every sequence in the batch emits one token per step, across every replica.
    aggregateTokensPerSecond: (config.batch * replicas) / tpot,
    decodeMfu: decodeModelFlops / (replicaPeak * tpot),
    decodeMbu: decodeBytes / (replicaBandwidth * tpot),
    prefillMfu: prefillModelFlops / (replicaPeak * ttft),
  }
}

/**
 * The batch size at which decode's linear layers stop being memory-bound.
 *
 * A weight-stationary GEMM's intensity is 2b/w_weight, so it crosses the machine
 * balance when b reaches (peak/bandwidth) * w/2. For an H100 in bf16 that is
 * around 295 * 2/2 = 295 tokens — which is why almost every real serving
 * deployment runs deep in the memory-bound regime.
 *
 * Note this moves the LINEAR layers only. Attention sits at 2*n_q/(n_kv*w_kv)
 * regardless of batch and never crosses at all.
 */
export function computeBoundBatch(
  topology: Topology,
  precision: TensorPrecision,
  weightDtype: Dtype,
): number {
  const { accelerator } = topology
  const balance = accelerator.peak[precision] / accelerator.memoryBandwidth
  return (balance * DTYPE_BYTES[weightDtype]) / 2
}
