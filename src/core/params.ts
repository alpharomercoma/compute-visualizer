/**
 * Parameter counting, from shapes rather than from a scalar.
 *
 * Two different counts matter and they are not the same number:
 *
 *   unique  - what the architecture defines. Drives FLOPs per token.
 *   stored  - what the checkpoint actually contains. Drives memory footprint.
 *
 * They diverge when a model declares tied embeddings but ships an `lm_head`
 * tensor anyway. Qwen3-0.6B does exactly this: HuggingFace reports 752 M from
 * safetensors metadata against a true 596 M, a 20% gap that is precisely one
 * embedding matrix. Loading it really does cost the larger figure, and computing
 * with it really does cost the smaller. So both are computed and, where they
 * disagree, both are shown.
 */

import { type Bytes, bytes } from './units'
import { DTYPE_BYTES, type Dtype } from './hardware'
import { type TransformerSpec, isMoeLayer } from './modelspec'

export interface ParameterCount {
  /** Parameters the architecture defines, counting a tied embedding once. */
  readonly unique: number
  /** Parameters actually materialised in the checkpoint. */
  readonly stored: number
  /** Parameters a token's forward pass touches — differs from `unique` for MoE. */
  readonly active: number

  readonly breakdown: {
    readonly embedding: number
    readonly attention: number
    readonly mlp: number
    readonly norms: number
    readonly lmHead: number
  }
}

/** Attention projection parameters for one layer. */
function attentionParams(spec: TransformerSpec): number {
  const { hiddenSize: h, attention: a } = spec

  if (a.kind === 'mla' && a.mla) {
    const { kvLoraRank, qkRopeHeadDim, qkNopeHeadDim, vHeadDim, qLoraRank } = a.mla
    const qHead = qkNopeHeadDim + qkRopeHeadDim
    // Query path is optionally low-rank; DeepSeek-V3 compresses it, V2-Lite does not.
    const q =
      qLoraRank !== undefined && qLoraRank > 0
        ? h * qLoraRank + qLoraRank + qLoraRank * (a.numQueryHeads * qHead)
        : h * (a.numQueryHeads * qHead)
    // Down-projection produces the cached latent plus the decoupled RoPE key.
    const kvDown = h * (kvLoraRank + qkRopeHeadDim) + kvLoraRank
    // Up-projection reconstructs per-head K (nope part) and V.
    const kvUp = kvLoraRank * (a.numQueryHeads * (qkNopeHeadDim + vHeadDim))
    const o = a.numQueryHeads * vHeadDim * h
    return q + kvDown + kvUp + o
  }

  const qDim = a.numQueryHeads * a.headDim
  const kvDim = a.numKvHeads * a.headDim
  // Q, K, V, O. K and V shrink with GQA; Q and O do not.
  const projections = h * qDim + 2 * (h * kvDim) + qDim * h
  // Qwen3-style RMSNorm on Q and K, one gain vector of head_dim each.
  const qkNormParams = spec.qkNorm ? 2 * a.headDim : 0
  return projections + qkNormParams
}

/** MLP parameters for one layer, dense or MoE depending on the layer index. */
function mlpParams(spec: TransformerSpec, layerIndex: number): number {
  const { hiddenSize: h, mlpMatrices: m } = spec

  if (!isMoeLayer(spec, layerIndex) || !spec.moe) {
    return m * h * spec.intermediateSize
  }

  const { numExperts, expertIntermediateSize: fe, sharedExperts } = spec.moe
  const routed = numExperts * m * h * fe
  const shared = sharedExperts * m * h * fe
  const router = h * numExperts
  return routed + shared + router
}

/** MLP parameters a single token actually activates. */
function activeMlpParams(spec: TransformerSpec, layerIndex: number): number {
  const { hiddenSize: h, mlpMatrices: m } = spec

  if (!isMoeLayer(spec, layerIndex) || !spec.moe) {
    return m * h * spec.intermediateSize
  }

  const { expertsPerToken: k, expertIntermediateSize: fe, sharedExperts, numExperts } = spec.moe
  // FLOPs scale with top-k, not with the expert count. This is the whole point
  // of MoE and the reason DeepSeek-V3 costs 37B per token rather than 671B.
  const routed = k * m * h * fe
  const shared = sharedExperts * m * h * fe
  const router = h * numExperts
  return routed + shared + router
}

export function countParameters(spec: TransformerSpec): ParameterCount {
  const { hiddenSize: h, numLayers: L, vocabSize: V } = spec

  const embedding = V * h
  const attention = attentionParams(spec) * L
  // Two RMSNorms per layer (pre-attention, pre-MLP) plus one final norm.
  const norms = 2 * h * L + h

  let mlp = 0
  let activeMlp = 0
  for (let i = 0; i < L; i++) {
    mlp += mlpParams(spec, i)
    activeMlp += activeMlpParams(spec, i)
  }

  const lmHead = spec.tiedEmbeddings ? 0 : V * h
  const unique = embedding + attention + mlp + norms + lmHead

  // A tied model that ships lm_head anyway stores one extra embedding matrix.
  const stored = unique + (spec.tiedEmbeddings ? V * h : 0)

  const active = embedding + attention + activeMlp + norms + lmHead

  return {
    unique,
    stored,
    active,
    breakdown: { embedding, attention, mlp, norms, lmHead },
  }
}

/**
 * Non-embedding parameter count, which is what `6ND` means by N.
 *
 * Kaplan et al. exclude embeddings not because they would be double-counted but
 * because an embedding lookup is a gather: it costs essentially no FLOPs, unlike
 * the 2N per token a dense matrix costs. Feeding a full HuggingFace parameter
 * count into 6ND therefore charges matrix-multiply time for a table lookup.
 */
export function nonEmbeddingParameters(spec: TransformerSpec): number {
  const counted = countParameters(spec)
  return counted.active - counted.breakdown.embedding - counted.breakdown.lmHead
}

/** Bytes the weights occupy, at a given storage precision. */
export function weightBytes(count: number, dtype: Dtype): Bytes {
  return bytes(count * DTYPE_BYTES[dtype])
}

/**
 * Reconciles the analytic count against HuggingFace's safetensors metadata.
 *
 * `safetensors.total` counts stored tensor elements. When it exceeds the unique
 * count by exactly one embedding matrix, the checkpoint has a redundant
 * `lm_head` despite declaring tied embeddings — a real condition worth
 * reporting rather than silently resolving in either direction.
 */
export function reconcile(
  spec: TransformerSpec,
  counted: ParameterCount,
  safetensorsTotal: number | undefined,
): { agrees: boolean; note?: string } {
  if (safetensorsTotal === undefined) return { agrees: true }

  const difference = safetensorsTotal - counted.unique
  const embeddingSize = spec.vocabSize * spec.hiddenSize
  // Checkpoints round differently across shards; a fraction of a percent is noise.
  const tolerance = Math.max(1e6, counted.unique * 0.001)

  if (Math.abs(difference) <= tolerance) return { agrees: true }

  if (Math.abs(difference - embeddingSize) <= tolerance && spec.tiedEmbeddings) {
    return {
      agrees: false,
      note:
        `declares tied embeddings but stores a separate lm_head: ` +
        `${formatCount(counted.unique)} unique, ${formatCount(safetensorsTotal)} stored`,
    }
  }

  return {
    agrees: false,
    note:
      `analytic count ${formatCount(counted.unique)} differs from safetensors ` +
      `${formatCount(safetensorsTotal)} by ${formatCount(Math.abs(difference))}`,
  }
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n < 1e10 ? 2 : 1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`
  return n.toLocaleString()
}
