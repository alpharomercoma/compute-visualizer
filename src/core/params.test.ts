import { describe, expect, it } from 'vitest'
import { countParameters, nonEmbeddingParameters, reconcile } from './params'
import {
  type HuggingFaceConfig,
  fromHuggingFaceConfig,
  kvBytesPerTokenPerLayer,
  kvHeadsPerRank,
} from './modelspec'

/**
 * Verbatim excerpts from real config.json files on the Hub. The expected
 * parameter counts are HuggingFace's own safetensors totals, so these tests
 * check the analytic model against ground truth rather than against itself.
 */

const MIXTRAL_8X7B: HuggingFaceConfig = {
  model_type: 'mixtral',
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  intermediate_size: 14336,
  vocab_size: 32000,
  max_position_embeddings: 32768,
  num_local_experts: 8,
  num_experts_per_tok: 2,
  tie_word_embeddings: false,
  torch_dtype: 'bfloat16',
}

const QWEN3_30B_A3B: HuggingFaceConfig = {
  model_type: 'qwen3_moe',
  hidden_size: 2048,
  num_hidden_layers: 48,
  num_attention_heads: 32,
  num_key_value_heads: 4,
  head_dim: 128,
  intermediate_size: 6144,
  moe_intermediate_size: 768,
  num_experts: 128,
  num_experts_per_tok: 8,
  vocab_size: 151936,
  max_position_embeddings: 40960,
  tie_word_embeddings: false,
  torch_dtype: 'bfloat16',
}

const QWEN3_0_6B: HuggingFaceConfig = {
  model_type: 'qwen3',
  hidden_size: 1024,
  num_hidden_layers: 28,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  head_dim: 128,
  intermediate_size: 3072,
  vocab_size: 151936,
  max_position_embeddings: 40960,
  tie_word_embeddings: true,
  torch_dtype: 'bfloat16',
}

const LLAMA_3_70B: HuggingFaceConfig = {
  model_type: 'llama',
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_key_value_heads: 8,
  intermediate_size: 28672,
  vocab_size: 128256,
  max_position_embeddings: 8192,
  tie_word_embeddings: false,
  torch_dtype: 'bfloat16',
}

const DEEPSEEK_V3: HuggingFaceConfig = {
  model_type: 'deepseek_v3',
  hidden_size: 7168,
  num_hidden_layers: 61,
  num_attention_heads: 128,
  num_key_value_heads: 128,
  intermediate_size: 18432,
  moe_intermediate_size: 2048,
  vocab_size: 129280,
  max_position_embeddings: 163840,
  q_lora_rank: 1536,
  kv_lora_rank: 512,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
  n_routed_experts: 256,
  n_shared_experts: 1,
  num_experts_per_tok: 8,
  first_k_dense_replace: 3,
  tie_word_embeddings: false,
  torch_dtype: 'bfloat16',
}

const spec = (config: HuggingFaceConfig, id: string) => fromHuggingFaceConfig(config, { id })

describe('parameter count against real checkpoints', () => {
  it('matches Mixtral-8x7B exactly', () => {
    expect(countParameters(spec(MIXTRAL_8X7B, 'mixtral')).unique).toBe(46_702_792_704)
  })

  it('matches Qwen3-30B-A3B exactly', () => {
    expect(countParameters(spec(QWEN3_30B_A3B, 'qwen3-moe')).unique).toBe(30_532_122_624)
  })

  it('matches Qwen3-0.6B exactly, counting a tied embedding once', () => {
    expect(countParameters(spec(QWEN3_0_6B, 'qwen3-0.6b')).unique).toBe(596_049_920)
  })

  it('lands within a fraction of a percent of Llama-3-70B', () => {
    // Meta ships 70,553,706,496; the analytic count omits nothing structural, so
    // any gap here is a shape misreading rather than rounding.
    const got = countParameters(spec(LLAMA_3_70B, 'llama-3-70b')).unique
    expect(Math.abs(got / 70_553_706_496 - 1)).toBeLessThan(0.005)
  })
})

describe('the stored-versus-unique divergence', () => {
  it('reports both counts when a tied model ships a separate lm_head', () => {
    const s = spec(QWEN3_0_6B, 'qwen3-0.6b')
    const counted = countParameters(s)

    expect(counted.unique).toBe(596_049_920)
    // HuggingFace reports this larger figure from safetensors metadata.
    expect(counted.stored).toBe(751_632_384)
    // The gap is exactly one embedding matrix, which is why it is 20%.
    expect(counted.stored - counted.unique).toBe(151936 * 1024)

    const check = reconcile(s, counted, 751_632_384)
    expect(check.agrees).toBe(false)
    expect(check.note).toContain('tied embeddings but stores a separate lm_head')
  })

  it('agrees when the checkpoint matches the architecture', () => {
    const s = spec(MIXTRAL_8X7B, 'mixtral')
    expect(reconcile(s, countParameters(s), 46_702_792_704).agrees).toBe(true)
  })
})

describe('MoE active parameters', () => {
  it('activates only top-k experts, not all of them', () => {
    const counted = countParameters(spec(QWEN3_30B_A3B, 'qwen3-moe'))
    // 128 experts declared, 8 routed per token: active must be far below total.
    expect(counted.active).toBeLessThan(counted.unique / 5)
    expect(counted.active / 1e9).toBeCloseTo(3.3, 0)
  })

  it('costs DeepSeek-V3 at its activated size, not its total', () => {
    const counted = countParameters(spec(DEEPSEEK_V3, 'deepseek-v3'))
    // Published as 671B total / 37B activated per token.
    expect(counted.unique / 1e9).toBeGreaterThan(600)
    expect(counted.active / 1e9).toBeGreaterThan(30)
    expect(counted.active / 1e9).toBeLessThan(45)
  })

  it('leaves a dense model with active equal to unique', () => {
    const counted = countParameters(spec(LLAMA_3_70B, 'llama-3-70b'))
    expect(counted.active).toBe(counted.unique)
  })
})

describe('the config.json landmines', () => {
  it('prefers an explicit head_dim over hidden/heads', () => {
    // Qwen3-30B-A3B: head_dim 128, but hidden/heads = 2048/32 = 64. Deriving it
    // halves the KV cache.
    const s = spec(QWEN3_30B_A3B, 'qwen3-moe')
    expect(s.attention.headDim).toBe(128)
    expect(s.hiddenSize / s.attention.numQueryHeads).toBe(64)
  })

  it('reads all three spellings of the expert count', () => {
    expect(spec(MIXTRAL_8X7B, 'a').moe?.numExperts).toBe(8) // num_local_experts
    expect(spec(QWEN3_30B_A3B, 'b').moe?.numExperts).toBe(128) // num_experts
    expect(spec(DEEPSEEK_V3, 'c').moe?.numExperts).toBe(256) // n_routed_experts
  })

  it('gives MoE experts their own width where the family declares one', () => {
    // Mixtral reuses intermediate_size; Qwen3-MoE and DeepSeek do not.
    expect(spec(MIXTRAL_8X7B, 'a').moe?.expertIntermediateSize).toBe(14336)
    expect(spec(QWEN3_30B_A3B, 'b').moe?.expertIntermediateSize).toBe(768)
  })

  it('detects MLA and drops the factor of two', () => {
    const s = spec(DEEPSEEK_V3, 'deepseek-v3')
    expect(s.attention.kind).toBe('mla')
    // kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576 elements per token/layer.
    expect(kvBytesPerTokenPerLayer(s, 2)).toBe(576 * 2)
  })

  it('carries v_head_dim rather than a meaningless derived head_dim', () => {
    // hidden/heads = 7168/128 = 56 for DeepSeek-V3, which matches no dimension
    // in the architecture. MLA heads are qk_nope+qk_rope = 192 wide on the
    // query side and v_head_dim = 128 on the value side.
    const s = spec(DEEPSEEK_V3, 'deepseek-v3')
    expect(s.attention.headDim).toBe(128)
    expect(s.hiddenSize / s.attention.numQueryHeads).toBe(56)
  })

  it('does not apply the standard KV formula to MLA', () => {
    const s = spec(DEEPSEEK_V3, 'deepseek-v3')
    const mla = kvBytesPerTokenPerLayer(s, 2)
    // What an equivalent MHA would cache: K and V for every head at v_head_dim.
    const asIfMha = 2 * s.attention.numQueryHeads * s.attention.headDim * 2
    // DeepSeek-V2 Table 1's headline claim, reproduced: MLA caches roughly what
    // GQA with 2.25 groups would, some 57x less than the MHA equivalent.
    expect(asIfMha / mla).toBeCloseTo(56.9, 0)
  })

  it('classifies attention by the head ratio', () => {
    expect(spec(LLAMA_3_70B, 'a').attention.kind).toBe('gqa')
    expect(spec({ ...LLAMA_3_70B, num_key_value_heads: 64 }, 'b').attention.kind).toBe('mha')
    expect(spec({ ...LLAMA_3_70B, num_key_value_heads: 1 }, 'c').attention.kind).toBe('mqa')
  })

  it('refuses a config it cannot read rather than inventing dimensions', () => {
    expect(() => spec({ model_type: 'llama' }, 'broken')).toThrow(/hidden_size/)

    const { vocab_size: _omitted, ...withoutVocab } = LLAMA_3_70B
    expect(() => spec(withoutVocab, 'broken')).toThrow(/vocab_size/)
  })
})

describe('KV sharding under tensor parallelism', () => {
  it('shards KV heads across ranks', () => {
    const s = spec(LLAMA_3_70B, 'llama-3-70b')
    expect(kvHeadsPerRank(s, 8)).toBe(1) // 8 KV heads over TP=8
  })

  it('stops sharding once heads run out', () => {
    const s = spec(LLAMA_3_70B, 'llama-3-70b')
    // TP=16 cannot split 8 KV heads further; implementations replicate instead,
    // so per-rank KV stops shrinking rather than going fractional.
    expect(kvHeadsPerRank(s, 16)).toBe(1)
  })
})

describe('6ND inputs', () => {
  it('excludes embeddings and the head from the non-embedding count', () => {
    const s = spec(LLAMA_3_70B, 'llama-3-70b')
    const counted = countParameters(s)
    const nonEmbedding = nonEmbeddingParameters(s)

    expect(nonEmbedding).toBeLessThan(counted.unique)
    expect(counted.unique - nonEmbedding).toBe(
      counted.breakdown.embedding + counted.breakdown.lmHead,
    )
  })
})
