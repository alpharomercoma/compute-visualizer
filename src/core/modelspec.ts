/**
 * The workload: a transformer described precisely enough to compute its cost.
 *
 * `TransformerSpec` is the normalized form. Getting here from a HuggingFace
 * `config.json` is the messy part, because the same concept is spelled
 * differently by every model family, and three of those spellings will silently
 * produce wrong numbers rather than an error:
 *
 *   - `head_dim` is sometimes present and NOT equal to hidden/heads. Qwen3-30B-A3B
 *     has head_dim 128 with hidden 2048 and 32 heads, so deriving it gives 64 and
 *     halves the KV cache.
 *   - expert count is `num_local_experts` (Mixtral), `num_experts` (Qwen3-MoE),
 *     or `n_routed_experts` (DeepSeek).
 *   - MLA does not cache K and V at all, so the standard formula overstates
 *     DeepSeek's KV cache by ~57x.
 *
 * Each is handled explicitly below and guarded by a test.
 */

import type { Dtype } from './hardware'

export type AttentionKind = 'mha' | 'gqa' | 'mqa' | 'mla'

export interface AttentionSpec {
  readonly kind: AttentionKind
  readonly numQueryHeads: number
  /** Equals `numQueryHeads` for MHA, 1 for MQA, a divisor of it for GQA. */
  readonly numKvHeads: number
  readonly headDim: number

  /** Multi-head latent attention (DeepSeek). Present only when kind is 'mla'. */
  readonly mla?: {
    readonly kvLoraRank: number
    readonly qkRopeHeadDim: number
    readonly qkNopeHeadDim: number
    readonly vHeadDim: number
    readonly qLoraRank?: number
  }

  /** Sliding-window size, when the model uses local attention. */
  readonly slidingWindow?: number
  /** Layers using global attention; the rest are local. Defaults to all global. */
  readonly globalLayers?: number
}

export interface MoeSpec {
  readonly numExperts: number
  readonly expertsPerToken: number
  readonly expertIntermediateSize: number
  readonly sharedExperts: number
  /** Leading layers that are dense rather than MoE (DeepSeek first_k_dense_replace). */
  readonly denseLayerPrefix: number
}

export interface TransformerSpec {
  readonly id: string
  readonly name: string
  readonly hiddenSize: number
  readonly numLayers: number
  readonly intermediateSize: number
  readonly vocabSize: number
  readonly maxPositionEmbeddings: number
  readonly attention: AttentionSpec
  readonly moe?: MoeSpec
  /**
   * Matrices in the MLP. SwiGLU and friends use three (gate, up, down); the
   * classic GeLU MLP uses two. This is the difference the Megatron closed form
   * bakes in as two and cannot express.
   */
  readonly mlpMatrices: 2 | 3
  readonly tiedEmbeddings: boolean
  /** Extra RMSNorms on Q and K (Qwen3). Adds 2*head_dim params per layer. */
  readonly qkNorm: boolean
  readonly weightDtype: Dtype
}

/** The subset of a HuggingFace config.json this tool reads. */
export interface HuggingFaceConfig {
  model_type?: string
  hidden_size?: number
  num_hidden_layers?: number
  num_attention_heads?: number
  num_key_value_heads?: number
  head_dim?: number
  intermediate_size?: number
  vocab_size?: number
  max_position_embeddings?: number
  tie_word_embeddings?: boolean
  torch_dtype?: string
  sliding_window?: number | null

  // MoE, spelled three different ways.
  num_local_experts?: number
  num_experts?: number
  n_routed_experts?: number
  num_experts_per_tok?: number
  moe_intermediate_size?: number
  n_shared_experts?: number
  first_k_dense_replace?: number
  mlp_only_layers?: readonly number[]

  // MLA (DeepSeek).
  kv_lora_rank?: number
  q_lora_rank?: number
  qk_rope_head_dim?: number
  qk_nope_head_dim?: number
  v_head_dim?: number
}

/** Model families whose MLP is a 2-matrix GeLU rather than a 3-matrix gated unit. */
const TWO_MATRIX_MLP = new Set(['gpt2', 'gpt_neox', 'bloom', 'opt', 'falcon'])

/** Families that add RMSNorm to Q and K inside attention. */
const QK_NORM_FAMILIES = new Set(['qwen3', 'qwen3_moe'])

const DTYPE_ALIASES: Record<string, Dtype> = {
  bfloat16: 'bf16',
  float16: 'fp16',
  half: 'fp16',
  float32: 'fp32',
  float: 'fp32',
  float8_e4m3fn: 'fp8',
  int8: 'int8',
}

function required(value: number | undefined, field: string): number {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    throw new Error(`config.json is missing a usable "${field}"`)
  }
  return value
}

/**
 * Normalizes a HuggingFace config into a `TransformerSpec`.
 *
 * Throws rather than guessing when a load-bearing field is absent. A model whose
 * config cannot be read is a model this tool should decline to cost, not one it
 * should cost with invented dimensions.
 */
export function fromHuggingFaceConfig(
  config: HuggingFaceConfig,
  identity: { id: string; name?: string },
): TransformerSpec {
  const modelType = config.model_type ?? ''
  const hiddenSize = required(config.hidden_size, 'hidden_size')
  const numLayers = required(config.num_hidden_layers, 'num_hidden_layers')
  const numQueryHeads = required(config.num_attention_heads, 'num_attention_heads')

  const isMlaConfig = config.kv_lora_rank !== undefined && config.qk_rope_head_dim !== undefined

  // Prefer the explicit head_dim. Qwen3-30B-A3B declares 128 while hidden/heads
  // is 64; deriving it there halves every KV-cache figure downstream.
  //
  // MLA has no single head dimension: its query/key heads are
  // qk_nope + qk_rope wide while its value heads are v_head_dim wide. Deriving
  // hidden/heads for DeepSeek-V3 yields 56, which corresponds to nothing in the
  // architecture. So `headDim` carries v_head_dim there — the width the output
  // projection actually sees — and the true per-path dims live in `mla`.
  const headDim = isMlaConfig
    ? (config.v_head_dim ?? required(config.qk_nope_head_dim, 'v_head_dim'))
    : (config.head_dim ?? hiddenSize / numQueryHeads)
  if (!Number.isFinite(headDim) || headDim <= 0) {
    throw new Error('config.json yields no usable head dimension')
  }

  // Absent num_key_value_heads means the model predates GQA, so every query head
  // has its own KV head.
  const numKvHeads = config.num_key_value_heads ?? numQueryHeads

  const isMla = isMlaConfig

  const attention: AttentionSpec = {
    kind: isMla ? 'mla' : numKvHeads === numQueryHeads ? 'mha' : numKvHeads === 1 ? 'mqa' : 'gqa',
    numQueryHeads,
    numKvHeads,
    headDim,
    ...(isMla
      ? {
          mla: {
            kvLoraRank: required(config.kv_lora_rank, 'kv_lora_rank'),
            qkRopeHeadDim: required(config.qk_rope_head_dim, 'qk_rope_head_dim'),
            qkNopeHeadDim: config.qk_nope_head_dim ?? 0,
            vHeadDim: config.v_head_dim ?? headDim,
            ...(config.q_lora_rank !== undefined ? { qLoraRank: config.q_lora_rank } : {}),
          },
        }
      : {}),
    ...(config.sliding_window ? { slidingWindow: config.sliding_window } : {}),
  }

  const numExperts =
    config.num_local_experts ?? config.num_experts ?? config.n_routed_experts ?? 0

  const moe: MoeSpec | undefined =
    numExperts > 0
      ? {
          numExperts,
          expertsPerToken: required(config.num_experts_per_tok, 'num_experts_per_tok'),
          // Mixtral reuses intermediate_size for its experts; Qwen3-MoE and
          // DeepSeek give experts their own, much smaller, width.
          expertIntermediateSize:
            config.moe_intermediate_size ?? required(config.intermediate_size, 'intermediate_size'),
          sharedExperts: config.n_shared_experts ?? 0,
          denseLayerPrefix: config.first_k_dense_replace ?? config.mlp_only_layers?.length ?? 0,
        }
      : undefined

  return {
    id: identity.id,
    name: identity.name ?? identity.id,
    hiddenSize,
    numLayers,
    intermediateSize: required(config.intermediate_size, 'intermediate_size'),
    vocabSize: required(config.vocab_size, 'vocab_size'),
    maxPositionEmbeddings: config.max_position_embeddings ?? 0,
    attention,
    ...(moe ? { moe } : {}),
    mlpMatrices: TWO_MATRIX_MLP.has(modelType) ? 2 : 3,
    tiedEmbeddings: config.tie_word_embeddings ?? false,
    qkNorm: QK_NORM_FAMILIES.has(modelType),
    weightDtype: DTYPE_ALIASES[config.torch_dtype ?? ''] ?? 'bf16',
  }
}

/**
 * Bytes of KV cache per token per layer, before any parallelism.
 *
 * MLA is genuinely a different mechanism, not a parameter change: it caches one
 * low-rank latent plus a shared decoupled RoPE key, and there is no separate V
 * to store. So there is no factor of 2. For DeepSeek-V3 this is 576 elements
 * against the 32,768 the standard formula would predict.
 */
export function kvBytesPerTokenPerLayer(spec: TransformerSpec, kvDtypeBytes: number): number {
  const a = spec.attention
  if (a.kind === 'mla' && a.mla) {
    return (a.mla.kvLoraRank + a.mla.qkRopeHeadDim) * kvDtypeBytes
  }
  return 2 * a.numKvHeads * a.headDim * kvDtypeBytes
}

/**
 * KV heads left on one rank under tensor parallelism.
 *
 * Heads cannot be split below one, so past `t = n_kv` implementations replicate
 * the cache rather than shard it further, and per-rank KV stops shrinking.
 */
export function kvHeadsPerRank(spec: TransformerSpec, tensorParallel: number): number {
  return spec.attention.numKvHeads / Math.min(tensorParallel, spec.attention.numKvHeads)
}

/** Whether a given layer index is an MoE layer or a dense one. */
export function isMoeLayer(spec: TransformerSpec, layerIndex: number): boolean {
  return spec.moe !== undefined && layerIndex >= spec.moe.denseLayerPrefix
}
