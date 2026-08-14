/**
 * Per-kernel-class FLOPs and HBM traffic.
 *
 * Everything here is stated for ONE layer on ONE rank. The scheduler
 * instantiates `L` copies. Folding the layer count in here and then summing over
 * layers downstream would overstate traffic by `L` — an 80x error on a 70B
 * model — so the boundary is worth being pedantic about.
 *
 * Why per-kernel at all: a transformer step is many kernels of wildly different
 * arithmetic intensity, and a roofline evaluated on the aggregate is a category
 * error. Within one kernel, memory and compute genuinely overlap, so
 * `max(t_compute, t_memory)` is right. Across kernels it is not, because by
 * Jensen's inequality `sum(max) >= max(sum)`: averaging a compute-bound MLP with
 * a memory-bound attention parks the aggregate intensity at the ridge point and
 * predicts a peak utilization that never happens.
 */

import { DTYPE_BYTES, type Dtype } from './hardware'
import { type TransformerSpec, isMoeLayer, kvHeadsPerRank } from './modelspec'
import { type Bytes, type Flops, bytes, flops } from './units'

export type KernelClass =
  | 'qkv-projection'
  | 'attention'
  | 'output-projection'
  | 'mlp'
  | 'norm'
  | 'lm-head'

/** One kernel invocation's cost, on one rank. */
export interface Kernel {
  readonly name: string
  readonly kernelClass: KernelClass
  readonly flops: Flops
  readonly hbmBytes: Bytes
  /** True when the traffic is a scattered gather rather than a stream. */
  readonly isGather: boolean
}

export interface Workload {
  readonly spec: TransformerSpec
  /** Concurrent sequences. */
  readonly batch: number
  /** New query positions this step: `contextLength` in prefill, 1 in decode. */
  readonly queryLength: number
  /** Total context each sequence holds, prompt plus generated so far. */
  readonly contextLength: number
  readonly tensorParallel: number
  readonly weightDtype: Dtype
  readonly kvDtype: Dtype
  readonly activationDtype: Dtype
  /** FlashAttention keeps attention traffic linear in `s`. Default and realistic. */
  readonly flashAttention: boolean
  /** Causal masking. Prefill gets a discount from it; decode does not. */
  readonly causal: boolean
}

/**
 * Query-key pairs actually evaluated.
 *
 * The familiar "causal halves attention" rule is an average over a FULL prefill,
 * where the mean query attends s/2 keys. It does not hold for decode, where the
 * single new query attends the entire existing context, nor for a chunk of a
 * chunked prefill, where the discount depends on where the chunk sits.
 *
 * The general count for `q` new queries after a prefix of `s - q`:
 *
 *   pairs = q(s - q) + q(q + 1)/2
 *
 * which gives s(s+1)/2 at q = s (recovering the halving) and exactly s at
 * q = 1 (no discount at all). Applying a blanket /2 understates decode
 * attention by 2x.
 */
export function attentionPairs(queryLength: number, contextLength: number, causal: boolean): number {
  const q = queryLength
  const s = contextLength

  // Without this, q > s silently yields a NEGATIVE pair count — q=4, s=1 gives
  // -2 — which then reaches flops() as an impossible quantity. New queries are
  // part of the context they attend to, so q > s describes nothing.
  if (!Number.isFinite(q) || !Number.isFinite(s) || q < 0 || s < 0) {
    throw new RangeError(`attention lengths must be non-negative, got q=${q}, s=${s}`)
  }
  if (q > s) {
    throw new RangeError(
      `queryLength cannot exceed contextLength: the new queries are part of the context they attend to (q=${q}, s=${s})`,
    )
  }

  if (!causal) return q * s
  const prefix = s - q
  return q * prefix + (q * (q + 1)) / 2
}

const FMA = 2 // a multiply-add is two floating-point operations

export function qkvProjection(w: Workload): Kernel {
  const { spec, tensorParallel: t } = w
  const h = spec.hiddenSize
  const a = spec.attention
  const tokens = w.batch * w.queryLength
  const wW = DTYPE_BYTES[w.weightDtype]
  const wA = DTYPE_BYTES[w.activationDtype]

  // Q is n_q heads wide; K and V are n_kv heads wide each. GQA shrinks only the
  // latter two, which is why it cuts KV cache far more than it cuts FLOPs.
  const outDim = ((a.numQueryHeads + 2 * a.numKvHeads) * a.headDim) / t
  const weightElements = h * outDim

  return {
    name: 'QKV projection',
    kernelClass: 'qkv-projection',
    flops: flops(FMA * tokens * h * outDim),
    hbmBytes: bytes(weightElements * wW + tokens * h * wA + tokens * outDim * wA),
    isGather: false,
  }
}

export function outputProjection(w: Workload): Kernel {
  const { spec, tensorParallel: t } = w
  const h = spec.hiddenSize
  const a = spec.attention
  const tokens = w.batch * w.queryLength
  const wW = DTYPE_BYTES[w.weightDtype]
  const wA = DTYPE_BYTES[w.activationDtype]

  const inDim = (a.numQueryHeads * a.headDim) / t
  return {
    name: 'output projection',
    kernelClass: 'output-projection',
    flops: flops(FMA * tokens * inDim * h),
    hbmBytes: bytes(inDim * h * wW + tokens * inDim * wA + tokens * h * wA),
    isGather: false,
  }
}

export function attention(w: Workload): Kernel {
  const { spec, tensorParallel: t } = w
  const a = spec.attention
  const wKv = DTYPE_BYTES[w.kvDtype]
  const wA = DTYPE_BYTES[w.activationDtype]

  const qHeadsPerRank = a.numQueryHeads / t
  const pairs = attentionPairs(w.queryLength, w.contextLength, w.causal)

  // Two matmuls of equal size: scores (Q x K^T) and the weighted sum (A x V).
  const attnFlops = 2 * FMA * w.batch * qHeadsPerRank * a.headDim * pairs

  // The KV cache read: every sequence's whole history, every step.
  const kvDimPerRank = kvHeadsPerRank(spec, t) * a.headDim
  const kvRead = 2 * w.batch * w.contextLength * kvDimPerRank * wKv

  // Q in and O out are only queryLength wide, which is why decode attention is
  // essentially pure KV streaming.
  const queryTraffic = 2 * w.batch * w.queryLength * qHeadsPerRank * a.headDim * wA

  // FlashAttention never materialises the score matrix. Modelling it as if it
  // did overstates long-context traffic by orders of magnitude.
  const scoreTraffic = w.flashAttention
    ? 0
    : 2 * w.batch * qHeadsPerRank * pairs * DTYPE_BYTES[w.activationDtype]

  return {
    name: w.queryLength === 1 ? 'decode attention' : 'attention',
    kernelClass: 'attention',
    flops: flops(attnFlops),
    hbmBytes: bytes(kvRead + queryTraffic + scoreTraffic),
    // Paged per-sequence KV is a gather, not a stream, and reaches a materially
    // lower fraction of peak bandwidth.
    isGather: true,
  }
}

export function mlp(w: Workload, layerIndex: number): Kernel {
  const { spec, tensorParallel: t } = w
  const h = spec.hiddenSize
  const tokens = w.batch * w.queryLength
  const wW = DTYPE_BYTES[w.weightDtype]
  const wA = DTYPE_BYTES[w.activationDtype]
  const m = spec.mlpMatrices

  const moe = spec.moe
  const isMoe = isMoeLayer(spec, layerIndex) && moe !== undefined

  // MoE FLOPs scale with top-k, not with the expert count — that is the whole
  // point of the architecture. Weight BYTES, however, scale with how many
  // experts a rank holds, because at low batch every routed expert still has to
  // be read from HBM.
  const width = isMoe ? moe.expertIntermediateSize : spec.intermediateSize
  const activeUnits = isMoe ? moe.expertsPerToken + moe.sharedExperts : 1
  const residentUnits = isMoe ? (moe.numExperts + moe.sharedExperts) / t : 1

  // Tensor parallelism splits every MLP matrix column-wise then row-wise, so the
  // arithmetic divides by `t` just as the weights do. Sharding the bytes but not
  // the FLOPs inflates decode intensity by `t` and flips the bottleneck
  // classification from memory to compute.
  const flopsPerUnit = (FMA * tokens * h * width * m) / t
  const weightElementsPerUnit = (m * h * width) / (isMoe ? 1 : t)

  return {
    name: isMoe ? 'MoE experts' : 'MLP',
    kernelClass: 'mlp',
    flops: flops(flopsPerUnit * activeUnits),
    hbmBytes: bytes(
      weightElementsPerUnit * residentUnits * wW +
        // The residual-width input and output are full width on every rank:
        // the input arrives replicated and the output is a partial sum awaiting
        // the all-reduce.
        tokens * h * wA * 2 +
        // The intermediate activation is column-sharded, so a rank only ever
        // materialises width/t of it. Charging every rank for the full width
        // overstates this term by `t`.
        (tokens * (width / t) * wA * activeUnits),
    ),
    isGather: false,
  }
}

/**
 * Norms, RoPE and residual adds.
 *
 * Negligible in FLOPs and emphatically not in bytes: they read and write the
 * whole activation tensor several times over. At batch 1 they are a real share
 * of a decode step, which is why they are modelled rather than waved away.
 */
export function elementwise(w: Workload): Kernel {
  const { spec } = w
  const tokens = w.batch * w.queryLength
  const wA = DTYPE_BYTES[w.activationDtype]
  // Two RMSNorms and two residual adds per layer, each reading and writing.
  const passes = 8
  return {
    name: 'norms, RoPE, residual',
    kernelClass: 'norm',
    flops: flops(tokens * spec.hiddenSize * 10),
    hbmBytes: bytes(passes * tokens * spec.hiddenSize * wA),
    isGather: false,
  }
}

/**
 * The language-model head. Once per step, not per layer.
 *
 * Worth 30-60% of total FLOPs for a small model with a large vocabulary, and
 * under 2% for a 400B one, so it is a term that has to be present rather than
 * folded into a constant.
 */
/**
 * The language-model head.
 *
 * `logitPositions` matters more than it looks. Serving a prompt needs logits for
 * the LAST position of each sequence only — one per sequence, not one per token.
 * Charging a 65,536-token prefill for 65,536 logit rows against a 128k vocabulary
 * adds ~17 TFLOP of work that no inference engine performs. Training does need
 * every position, so the caller says which.
 */
export function lmHead(w: Workload, logitPositions?: number): Kernel {
  const { spec, tensorParallel: t } = w
  const h = spec.hiddenSize
  const wW = DTYPE_BYTES[w.weightDtype]
  const wA = DTYPE_BYTES[w.activationDtype]
  const tokens = logitPositions ?? w.batch * w.queryLength
  const vocabPerRank = spec.vocabSize / t

  return {
    name: 'LM head',
    kernelClass: 'lm-head',
    flops: flops(FMA * tokens * h * vocabPerRank),
    hbmBytes: bytes(h * vocabPerRank * wW + tokens * h * wA + tokens * vocabPerRank * wA),
    isGather: false,
  }
}

/** Every kernel in one transformer layer, forward, on one rank. */
export function layerKernels(w: Workload, layerIndex: number): Kernel[] {
  return [qkvProjection(w), attention(w), outputProjection(w), mlp(w, layerIndex), elementwise(w)]
}

/**
 * Cluster-wide ALGORITHMIC FLOPs for one forward pass — the numerator of MFU.
 *
 * Deliberately computed from the architecture rather than by summing per-rank
 * kernels and scaling by rank count. Those are hardware FLOPs: they include work
 * every rank redundantly repeats (norms and residuals act on the unsharded
 * hidden state, so `t` ranks each do all of it), and crediting a distributed
 * system for its own redundancy inflates MFU. Model FLOPs are what the model
 * mathematically requires, independent of how it was sharded.
 */
export function modelFlops(
  spec: TransformerSpec,
  nonEmbeddingParams: number,
  w: Pick<Workload, 'batch' | 'queryLength' | 'contextLength' | 'causal'>,
  logitPositions: number,
): Flops {
  const tokens = w.batch * w.queryLength
  const a = spec.attention

  const linear = FMA * nonEmbeddingParams * tokens
  const attn =
    2 *
    FMA *
    w.batch *
    a.numQueryHeads *
    a.headDim *
    attentionPairs(w.queryLength, w.contextLength, w.causal) *
    spec.numLayers
  const head = FMA * logitPositions * spec.hiddenSize * spec.vocabSize

  return flops(linear + attn + head)
}

export const totalFlops = (kernels: readonly Kernel[]): Flops =>
  flops(kernels.reduce((sum, k) => sum + k.flops, 0))

export const totalBytes = (kernels: readonly Kernel[]): Bytes =>
  bytes(kernels.reduce((sum, k) => sum + k.hbmBytes, 0))
