/**
 * A curated catalog, bundled with the app.
 *
 * The Hub is the point of this tool, but three things make an offline catalog
 * necessary rather than decorative: anonymous rate limits are per-IP and shared
 * with everyone behind the same address, gated repositories refuse anonymous
 * config reads, and a tool that shows nothing until a network round-trip
 * completes is a tool nobody explores.
 *
 * Every config below is a verbatim excerpt of the model's published
 * `config.json`, trimmed to the fields the model reads.
 */

import type { HuggingFaceConfig } from '../core/modelspec'

export interface Preset {
  readonly id: string
  readonly label: string
  readonly family: string
  readonly config: HuggingFaceConfig
  /** Stored element count from safetensors metadata, where published. */
  readonly storedParameters?: number
  readonly note?: string
}

export const PRESETS: readonly Preset[] = [
  {
    id: 'meta-llama/Llama-3.1-8B-Instruct',
    label: 'Llama 3.1 8B Instruct',
    family: 'Llama',
    storedParameters: 8_030_261_248,
    config: {
      model_type: 'llama',
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      intermediate_size: 14336,
      vocab_size: 128256,
      max_position_embeddings: 131072,
      tie_word_embeddings: false,
      torch_dtype: 'bfloat16',
    },
  },
  {
    id: 'meta-llama/Llama-3.1-70B-Instruct',
    label: 'Llama 3.1 70B Instruct',
    family: 'Llama',
    storedParameters: 70_553_706_496,
    config: {
      model_type: 'llama',
      hidden_size: 8192,
      num_hidden_layers: 80,
      num_attention_heads: 64,
      num_key_value_heads: 8,
      intermediate_size: 28672,
      vocab_size: 128256,
      max_position_embeddings: 131072,
      tie_word_embeddings: false,
      torch_dtype: 'bfloat16',
    },
    note: 'The reference scenario. GQA cuts its KV cache eightfold against MHA.',
  },
  {
    id: 'meta-llama/Llama-3.1-405B',
    label: 'Llama 3.1 405B',
    family: 'Llama',
    config: {
      model_type: 'llama',
      hidden_size: 16384,
      num_hidden_layers: 126,
      num_attention_heads: 128,
      num_key_value_heads: 8,
      intermediate_size: 53248,
      vocab_size: 128256,
      max_position_embeddings: 131072,
      tie_word_embeddings: false,
      torch_dtype: 'bfloat16',
    },
    note: 'Trained on 16,384 H100s at 400 TFLOP/s per GPU, BF16 MFU 38-43%.',
  },
  {
    id: 'mistralai/Mistral-7B-Instruct-v0.3',
    label: 'Mistral 7B Instruct v0.3',
    family: 'Mistral',
    storedParameters: 7_248_023_552,
    config: {
      model_type: 'mistral',
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      intermediate_size: 14336,
      vocab_size: 32768,
      max_position_embeddings: 32768,
      tie_word_embeddings: false,
      torch_dtype: 'bfloat16',
    },
  },
  {
    id: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    label: 'Mixtral 8x7B Instruct',
    family: 'Mistral',
    storedParameters: 46_702_792_704,
    config: {
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
    },
    note: '47B stored, ~13B active per token. Spells its expert count num_local_experts.',
  },
  {
    id: 'Qwen/Qwen3-8B',
    label: 'Qwen3 8B',
    family: 'Qwen',
    storedParameters: 8_190_735_360,
    config: {
      model_type: 'qwen3',
      hidden_size: 4096,
      num_hidden_layers: 36,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
      intermediate_size: 12288,
      vocab_size: 151936,
      max_position_embeddings: 40960,
      tie_word_embeddings: false,
      torch_dtype: 'bfloat16',
    },
  },
  {
    id: 'Qwen/Qwen3-30B-A3B',
    label: 'Qwen3 30B-A3B',
    family: 'Qwen',
    storedParameters: 30_532_122_624,
    config: {
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
    },
    note: 'head_dim is 128 while hidden/heads is 64. Deriving it halves the KV cache.',
  },
  {
    id: 'Qwen/Qwen3-0.6B',
    label: 'Qwen3 0.6B',
    family: 'Qwen',
    storedParameters: 751_632_384,
    config: {
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
    },
    note: 'Declares tied embeddings but stores lm_head anyway: 596M unique, 752M stored.',
  },
  {
    id: 'deepseek-ai/DeepSeek-V3',
    label: 'DeepSeek V3',
    family: 'DeepSeek',
    config: {
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
    },
    note: '671B stored, 37B active. MLA caches 576 elements per token per layer, not 32,768.',
  },
  {
    id: 'google/gemma-2-9b',
    label: 'Gemma 2 9B',
    family: 'Gemma',
    config: {
      model_type: 'gemma2',
      hidden_size: 3584,
      num_hidden_layers: 42,
      num_attention_heads: 16,
      num_key_value_heads: 8,
      head_dim: 256,
      intermediate_size: 14336,
      vocab_size: 256000,
      max_position_embeddings: 8192,
      sliding_window: 4096,
      tie_word_embeddings: true,
      torch_dtype: 'bfloat16',
    },
    note: 'A 256k vocabulary makes the LM head a large share of a small model.',
  },
]

export const DEFAULT_PRESET = PRESETS[1]!

export const presetById = (id: string): Preset | undefined => PRESETS.find((p) => p.id === id)
